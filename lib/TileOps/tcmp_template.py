# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tcmp

Note: A5 hardware implements tcmp as packed comparison between two tiles:
  dst = packed_mask(src0 cmp src1)

Dst is i8 type with same shape as src (packed predicate mask bytes).
Uses vcmp + psts to produce packed predicate mask output.

Implementation per TCmp.hpp:
  - 32B types (f32, i32): uses TCmp_32B path with pdintlv_b8 + PK storage
  - 16B types (f16, i16): uses TCmp_8B_16B path with PK storage
  - 8B types (i8, u8): uses TCmp_8B_16B path with NORM storage

Supported comparison modes (via cmp_mode attribute):
  eq, ne, lt, gt, ge, le
"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tcmp",
    dtypes=[
        (pto.f32, pto.f32, pto.i8),
        (pto.i32, pto.i32, pto.i8),
        (pto.f16, pto.f16, pto.i8),
        (pto.i16, pto.i16, pto.i8),
        (pto.i8, pto.i8, pto.i8),
        (pto.ui8, pto.ui8, pto.i8),
    ],
    advanced=True,
)
def template_tcmp(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """src0 cmp src1 -> packed mask in dst (i8, same shape as src)
    
    TCmp.hpp structure:
      - 32B: TCmp_32B with double iteration + pdintlv_b8 + PK
      - 16B: TCmp_8B_16B with PK
      - 8B: TCmp_8B_16B with NORM
    """
    dtype = src0.element_type
    valid_rows, valid_cols = src0.valid_shape
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")

    lanes = pto.get_lanes(dtype)
    dst_ptr = dst.as_ptr()
    dst_stride = dst.shape[1]

    if pto.constexpr(dtype == pto.f32 or dtype == pto.i32):
        # 32B path: TCmp_32B implementation per TCmp.hpp
        # repeatElm = CCE_VL / sizeof(uint32_t) = 64
        # repeatTimes = CeilDivision(validCol, repeatElm) + 1
        # iterations = repeatTimes // 2
        # Each iteration loads 2*64 elements (4 vlds), uses plt_b32 to split into
        # two 32-lane comparisons, then pdintlv_b8 to interleave
        repeat_times = (valid_cols + lanes - 1) // lanes + 1
        iterations = repeat_times // 2

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for j in range(0, iterations, 1):
                # Load 4 vector registers per TCmp.hpp structure
                # src0Reg0, src0Reg1 from j*2*repeatElm offset
                # src1Reg0, src1Reg1 from (j*2+1)*repeatElm offset
                vec_src0_first = pto.vlds(src0[row, j * lanes * 2:])
                vec_src1_first = pto.vlds(src1[row, j * lanes * 2:])
                vec_src0_second = pto.vlds(src0[row, (j * 2 + 1) * lanes:])
                vec_src1_second = pto.vlds(src1[row, (j * 2 + 1) * lanes:])
                
                # Use plt_b32 to create 32-lane masks (POST_UPDATE semantics)
                mask_first, remained = pto.make_mask(dtype, remained)
                cmp_first = pto.vcmp(vec_src0_first, vec_src1_first, mask_first, cmp_mode)
                cmp_first_b8 = pto.pbitcast(cmp_first, pto.mask_b8)
                
                mask_second, remained = pto.make_mask(dtype, remained)
                cmp_second = pto.vcmp(vec_src0_second, vec_src1_second, mask_second, cmp_mode)
                cmp_second_b8 = pto.pbitcast(cmp_second, pto.mask_b8)
                
                # pdintlv_b8 interleave two mask_b8 results
                packed_low, packed_high = pto.pdintlv_b8(cmp_first_b8, cmp_second_b8)
                
                # Store to dst: offset = (row * dstStride + j * 4) in uint32 units
                # byte_offset = row * dst_stride + j * 16
                # For i8 dst, dstStride = RowStride / 4 = 16 uint32 units
                byte_offset = row * dst_stride + j * 16
                pto.psts(packed_low, dst_ptr, byte_offset, pto.PredicateDist.PK)

    elif pto.constexpr(dtype == pto.f16 or dtype == pto.i16):
        # 16B path: TCmp_8B_16B with PK
        # vcmp returns mask_b16, cast to mask_b8 for psts PK
        iters_per_row = (valid_cols + lanes - 1) // lanes

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for j in range(0, iters_per_row, 1):
                mask, remained = pto.make_mask(dtype, remained)
                vec0 = pto.vlds(src0[row, j * lanes:])
                vec1 = pto.vlds(src1[row, j * lanes:])
                cmp = pto.vcmp(vec0, vec1, mask, cmp_mode)
                cmp_b8 = pto.pbitcast(cmp, pto.mask_b8)
                byte_offset = row * dst_stride + j * 16
                pto.psts(cmp_b8, dst_ptr, byte_offset, pto.PredicateDist.PK)

    else:
        # 8B path: TCmp_8B_16B with NORM
        # vcmp returns mask_b8 directly, no cast needed
        iters_per_row = (valid_cols + lanes - 1) // lanes

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for j in range(0, iters_per_row, 1):
                mask, remained = pto.make_mask(dtype, remained)
                vec0 = pto.vlds(src0[row, j * lanes:])
                vec1 = pto.vlds(src1[row, j * lanes:])
                cmp = pto.vcmp(vec0, vec1, mask, cmp_mode)
                byte_offset = row * dst_stride + j * 32
                pto.psts(cmp, dst_ptr, byte_offset, pto.PredicateDist.NORM)

    return