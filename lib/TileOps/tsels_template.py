# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tsels

NOTE: This template uses pto.plds for mask loading which directly
loads predicate mask from UB without vcmps comparison.

TSels: Select between source tile and scalar based on mask.
- mask=true: select from src
- mask=false: select scalar value

Mask tile format:
- Packed predicate bytes in UB.
- Each row stores ceil(valid_cols / 8) valid bytes; tile row stride may be padded.
- mask_dtype determines the storage format (i8/i16/i32), but the actual
  predicate bits are packed and accessed as bytes.

IMPORTANT: mask_row_stride is always mask.shape[1] (element count), 
because mask tile stride equals cols in element units regardless of mask_dtype.
Byte offset for plds is col // 8 (one byte covers 8 elements).
"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tsels",
    dtypes=[
        (pto.i8, pto.i8, pto.i8, pto.i8, pto.i8),
        (pto.i16, pto.i8, pto.i8, pto.i8, pto.i8),
        (pto.i32, pto.i8, pto.i8, pto.i8, pto.i8),
        (pto.i8, pto.i16, pto.i16, pto.i16, pto.i16),
        (pto.i16, pto.i16, pto.i16, pto.i16, pto.i16),
        (pto.i32, pto.i16, pto.i16, pto.i16, pto.i16),
        (pto.i8, pto.i32, pto.i32, pto.i32, pto.i32),
        (pto.i16, pto.i32, pto.i32, pto.i32, pto.i32),
        (pto.i32, pto.i32, pto.i32, pto.i32, pto.i32),
        (pto.i8, pto.f16, pto.f16, pto.f16, pto.f16),
        (pto.i16, pto.f16, pto.f16, pto.f16, pto.f16),
        (pto.i32, pto.f16, pto.f16, pto.f16, pto.f16),
        (pto.i8, pto.f32, pto.f32, pto.f32, pto.f32),
        (pto.i16, pto.f32, pto.f32, pto.f32, pto.f32),
        (pto.i32, pto.f32, pto.f32, pto.f32, pto.f32),
    ],
    advanced=True
)
def template_tsels(mask: pto.Tile, src: pto.Tile, tmp: pto.Tile, scalar: pto.AnyType, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    mask_dtype = mask.element_type

    lanes = pto.get_lanes(dtype)
    mask_row_stride = mask.shape[1] * pto.bytewidth(mask_dtype)
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, pto.MemorySpace.UB))

    scalar_mask, _ = pto.make_mask(dtype, lanes)
    vreg_scalar = pto.vdup(scalar, scalar_mask)

    if pto.constexpr(lanes == 64):
        full_mask_b16 = pto.pset_b16(pto.MaskPattern.ALL)
        pair_width = lanes * 2
        paired_cols = (valid_cols // pair_width) * pair_width
        for row in range(0, valid_rows, 1):
            for col in range(0, paired_cols, pair_width):
                mask_offset = row * mask_row_stride + col // 8
                select_mask_raw = pto.plds(mask_ptr, mask_offset, pto.PredicateDist.US)
                select_mask = select_mask_raw.astype(pto.mask_b16)
                pred0, _ = pto.make_mask(dtype, pair_width)
                pred1, _ = pto.make_mask(dtype, lanes)
                select_mask0, select_mask1 = pto.pintlv_b16(select_mask, full_mask_b16)
                select_mask0 = select_mask0.astype(pto.mask_b32)
                select_mask1 = select_mask1.astype(pto.mask_b32)
                src0 = pto.vlds(src[row, col:])
                src1 = pto.vlds(src[row, col + lanes:])
                selected0 = pto.vsel(src0, vreg_scalar, select_mask0)
                selected1 = pto.vsel(src1, vreg_scalar, select_mask1)
                pto.vsts(selected0, dst[row, col:], pred0)
                pto.vsts(selected1, dst[row, col + lanes:], pred1)
            tail_cols = valid_cols - paired_cols
            if tail_cols > 0:
                col = paired_cols
                mask_offset = row * mask_row_stride + col // 8
                select_mask_raw = pto.plds(mask_ptr, mask_offset, pto.PredicateDist.US)
                select_mask = select_mask_raw.astype(pto.mask_b16)
                select_mask0 = pto.punpack(select_mask, pto.PredicatePart.LOWER)
                select_mask0 = select_mask0.astype(pto.mask_b32)
                pred0, _ = pto.make_mask(dtype, tail_cols)
                src0 = pto.vlds(src[row, col:])
                selected0 = pto.vsel(src0, vreg_scalar, select_mask0)
                pto.vsts(selected0, dst[row, col:], pred0)
    elif pto.constexpr(lanes == 128):
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                pred_mask, remained = pto.make_mask(dtype, remained)
                mask_offset = row * mask_row_stride + col // 8
                select_mask_raw = pto.plds(mask_ptr, mask_offset, pto.PredicateDist.US)
                select_mask = select_mask_raw.astype(pto.mask_b16)
                src_vec = pto.vlds(src[row, col:])
                selected = pto.vsel(src_vec, vreg_scalar, select_mask)
                pto.vsts(selected, dst[row, col:], pred_mask)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                pred_mask, remained = pto.make_mask(dtype, remained)
                mask_offset = row * mask_row_stride + col // 8
                select_mask = pto.plds(mask_ptr, mask_offset, pto.PredicateDist.NORM)
                src_vec = pto.vlds(src[row, col:])
                selected = pto.vsel(src_vec, vreg_scalar, select_mask)
                pto.vsts(selected, dst[row, col:], pred_mask)
    return