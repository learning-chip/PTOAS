# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tcmps

Note: A5 hardware implements tcmps as packed comparison with scalar:
  dst = packed_mask(src cmp scalar)

Uses vcmps + psts to produce packed predicate mask output.
Implementation:
  - 32B types (f32, i32): 64 elements per repeat, 32 bytes per iteration (NORM mode).
  - 16B types (f16, i16): 128 elements per repeat, 16 bytes per iteration (PK mode).
  - 8B types (i8, u8): 256 elements per repeat, 32 bytes per iteration (NORM mode).

Supported comparison modes (via cmp_mode op attribute):
  eq, ne, lt, gt, ge, le
"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tcmps",
    dtypes=[
        (pto.f32, pto.f32, pto.ui8), (pto.i32, pto.i32, pto.ui8),
        (pto.f16, pto.f16, pto.ui8), (pto.i16, pto.i16, pto.ui8),
        (pto.i8, pto.i8, pto.ui8), (pto.ui8, pto.ui8, pto.ui8),
    ],
    advanced=True,
)
def template_tcmps(src: pto.Tile, scalar: pto.AnyType, dst: pto.Tile):
    """src cmp scalar -> packed mask in dst (ui8)

    - 32B: 1 repeat per iteration, 32 bytes/store (NORM mode, 1 bit per element)
    - 16B: 1 repeat per iteration, 16 bytes/store (PK mode)
    - 8B: 1 repeat per iteration, 32 bytes/store (NORM mode)
    """
    dtype = src.element_type
    valid_rows, valid_cols = src.valid_shape
    cmp_mode = pto.get_op_attr("cmp_mode")

    lanes = pto.get_lanes(dtype)
    dst_ptr = dst.as_ptr()

    if pto.constexpr(dtype == pto.f32 or dtype == pto.i32):
        # 32B path: 2 vcmps + pbitcast + dintlv_b8 -> psts(PK)
        # Use 2D slicing for safety, convert linear offset to (row, col)
        bytes_per_iter = 16
        elem_size = 4
        total_elm = valid_rows * valid_cols
        repeat_elm = lanes
        
        # Calculate repeat times matching ISA: CeilDivision + 1
        # But limit iterations to avoid complete out-of-bounds access
        repeat_times = (total_elm + repeat_elm - 1) // repeat_elm + 1
        
        # Safety: limit iterations to avoid elem_offset beyond total_elm + repeat_elm
        # ISA allows one extra repeat for odd elements, but we need to protect DSL slicing
        iterations_needed = repeat_times // 2
        
        for i in range(0, iterations_needed, 1):
            # Convert linear element offsets to (row, col) coordinates
            elem_offset0 = i * 2 * repeat_elm
            elem_offset1 = (i * 2 + 1) * repeat_elm
            
            row0 = elem_offset0 // valid_cols
            col0 = elem_offset0 % valid_cols
            row1 = elem_offset1 // valid_cols
            col1 = elem_offset1 % valid_cols
            
            # Remaining elements for each position (clamp to >= 0)
            # When remaining <= 0, make_mask returns all-zero mask (safe)
            remaining0 = total_elm - elem_offset0
            if remaining0 < 0:
                remaining0 = 0
            remaining1 = total_elm - elem_offset1
            if remaining1 < 0:
                remaining1 = 0
            
            # Predicate for each compare
            mask0, _ = pto.make_mask(dtype, remaining0)
            mask1, _ = pto.make_mask(dtype, remaining1)
            
            # Load using 2D slicing (safer than pointer+offset)
            # When row/col exceeds valid_shape, mask ensures no invalid data is used
            vec0 = pto.vlds(src[row0, col0:])
            vec1 = pto.vlds(src[row1, col1:])
            
            cmp0 = pto.vcmps(vec0, scalar, mask0, cmp_mode)
            cmp1 = pto.vcmps(vec1, scalar, mask1, cmp_mode)
            
            # Convert mask_b32 to mask_b8 and interleave
            cmp0_b8 = pto.pbitcast(cmp0, pto.mask_b8)
            cmp1_b8 = pto.pbitcast(cmp1, pto.mask_b8)
            cmp_interleaved, _ = pto.pdintlv_b8(cmp0_b8, cmp1_b8)
            
            # Linear byte offset for output
            byte_offset = i * bytes_per_iter
            pto.psts(cmp_interleaved, dst_ptr, byte_offset, pto.PredicateDist.PK)
    elif pto.constexpr(dtype == pto.f16 or dtype == pto.i16):
        # 16B path: 128 elements per repeat, 16 bytes per iteration (PK mode).
        # Each vcmps produces 128 bits; PK mode packs them into 16 bytes,
        # achieving 1 bit per element.
        bytes_per_iter = 16
        iters_per_row = (valid_cols + lanes - 1) // lanes

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                cmp = pto.vcmps(vec, scalar, mask, cmp_mode)
                byte_offset = (row * iters_per_row + col // lanes) * bytes_per_iter
                pto.psts(cmp, dst_ptr, byte_offset, pto.PredicateDist.PK)
    else:
        # 8B path: 256 elements per repeat, 32 bytes packed per iteration
        bytes_per_iter = 32
        iters_per_row = (valid_cols + lanes - 1) // lanes

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                cmp = pto.vcmps(vec, scalar, mask, cmp_mode)
                byte_offset = (row * iters_per_row + col // lanes) * bytes_per_iter
                pto.psts(cmp, dst_ptr, byte_offset, pto.PredicateDist.NORM)

    return
