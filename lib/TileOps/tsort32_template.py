# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tsort32"""

import tilelang_dsl as pto

BLOCK_SIZE = 32
FLOAT_DST_STRIDE_COEF = 2
HALF_DST_STRIDE_COEF = 4
MAX_UB_TMP = 32 * 255  # 8160 bytes
REPEAT_MAX = 255


def _constraint_aligned(
    src_shape=(),
    src_valid_shape=(),
    idx_shape=(),
    idx_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_config=None,
    idx_config=None,
    dst_config=None,
) -> bool:
    """Constraint for Format1: valid_cols % 32 == 0 (aligned, no tmp needed)."""
    if len(src_valid_shape) != 2:
        return False
    valid_cols = src_valid_shape[1]
    return valid_cols % BLOCK_SIZE == 0


def _constraint_unaligned(
    src_shape=(),
    src_valid_shape=(),
    idx_shape=(),
    idx_valid_shape=(),
    tmp_shape=(),
    tmp_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_config=None,
    idx_config=None,
    tmp_config=None,
    dst_config=None,
) -> bool:
    """Constraint for Format2: valid_cols % 32 != 0 (unaligned, tmp needed)."""
    if len(src_valid_shape) != 2:
        return False
    valid_cols = src_valid_shape[1]
    return valid_cols % BLOCK_SIZE != 0


@pto.vkernel(
    target="a5",
    advanced=True,
    op="pto.tsort32",
    constraints=[_constraint_aligned]
)
def template_tsort32(src: pto.Tile, idx: pto.Tile, dst: pto.Tile):
    """
    TSort32 Format1: Bitonic sort for aligned cols (valid_cols % 32 == 0).
    
    Semantics (matching pto-isa TSort32.hpp Format1):
    - Sorts src values into dst, generating indices in idx
    - Direct sort without tmp buffer when src.valid_cols % 32 == 0
    - No padding needed
    """
    dtype = dst.element_type
    valid_rows = dst.valid_shape[0]
    valid_cols = src.valid_shape[1]

    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()
    idx_ptr = idx.as_ptr()

    elem_bytes = pto.bytewidth(dtype)
    dst_stride = ((dst.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    src_stride = ((src.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    idx_stride = ((idx.shape[1] * 4 + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // 4
    if idx.valid_shape[0] == 1:
        idx_stride = 0

    type_coef = HALF_DST_STRIDE_COEF
    if pto.constexpr(dtype == pto.f32):
        type_coef = FLOAT_DST_STRIDE_COEF

    repeat_num_per_row = (valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    if repeat_num_per_row <= REPEAT_MAX:
        for i in range(0, valid_rows, 1):
            pto.vbitsort(
                pto.addptr(dst_ptr, i * dst_stride),
                pto.addptr(src_ptr, i * src_stride),
                pto.addptr(idx_ptr, i * idx_stride),
                repeat_num_per_row
            )
    else:
        loop_num = (repeat_num_per_row + REPEAT_MAX - 1) // REPEAT_MAX
        tail_repeat_num = repeat_num_per_row % REPEAT_MAX
        for i in range(0, valid_rows, 1):
            for j in range(0, loop_num, 1):
                repeat_num = REPEAT_MAX
                if j == loop_num - 1:
                    repeat_num = tail_repeat_num
                    
                pto.vbitsort(
                    pto.addptr(dst_ptr, i * dst_stride + j * REPEAT_MAX * BLOCK_SIZE * type_coef),
                    pto.addptr(src_ptr, i * src_stride + j * REPEAT_MAX * BLOCK_SIZE),
                    pto.addptr(idx_ptr, i * idx_stride + j * REPEAT_MAX * BLOCK_SIZE),
                    repeat_num
                )
    return


@pto.vkernel(
    target="a5",
    advanced=True,
    op="pto.tsort32",
    constraints=[_constraint_unaligned]
)
def template_tsort32_with_tmp(src: pto.Tile, idx: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """
    TSort32 Format2: Bitonic sort with tmp buffer for unaligned cols.
    
    Semantics (matching pto-isa TSort32.hpp Format2):
    - Sorts src values into dst, generating indices in idx
    - Uses tmp buffer when src.valid_cols % 32 != 0 (padding needed)
    - Pads unaligned tail with NaN to ensure correct sorting
    """
    dtype = dst.element_type
    valid_rows = dst.valid_shape[0]
    valid_cols = src.valid_shape[1]

    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()
    idx_ptr = idx.as_ptr()
    tmp_ptr = tmp.as_ptr()

    elem_bytes = pto.bytewidth(dtype)
    dst_stride = ((dst.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    src_stride = ((src.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    idx_stride = ((idx.shape[1] * 4 + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // 4
    if idx.valid_shape[0] == 1:
        idx_stride = 0

    type_coef = HALF_DST_STRIDE_COEF
    if pto.constexpr(dtype == pto.f32):
        type_coef = FLOAT_DST_STRIDE_COEF

    repeat_num_per_row = (valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    src_tail_per_row = valid_cols % BLOCK_SIZE
    src_tail_repeat_num = ((valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE) % REPEAT_MAX

    if pto.constexpr(dtype == pto.f16):
        min_val = pto.f16(0xFC00)
    elif pto.constexpr(dtype == pto.bf16):
        min_val = pto.bf16(0xFF80)
    else:
        min_val = pto.f32(0xFF800000)

    src_shape_bytes_per_row = valid_cols * elem_bytes

    if src_shape_bytes_per_row <= MAX_UB_TMP:
        len_burst = (src_shape_bytes_per_row + BLOCK_SIZE - 1) // BLOCK_SIZE

        for i in range(0, valid_rows, 1):
            pto.copy_ubuf_to_ubuf(
                pto.addptr(src_ptr, i * src_stride),
                tmp_ptr,
                0, 1, len_burst, 0, 0
            )

            tmp_last_offset = ((valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) - BLOCK_SIZE

            vec = pto.vlds(tmp[0, tmp_last_offset:])
            pad_mask, _ = pto.make_mask(dtype, BLOCK_SIZE - src_tail_per_row)
            vec = pto.vdup(min_val, pad_mask)
            pto.vsts(vec, tmp[0, tmp_last_offset:], pad_mask)

            pto.vbitsort(
                pto.addptr(dst_ptr, i * dst_stride),
                tmp_ptr,
                pto.addptr(idx_ptr, i * idx_stride),
                repeat_num_per_row
            )
    else:
        loop_num = (repeat_num_per_row + REPEAT_MAX - 1) // REPEAT_MAX

        for i in range(0, valid_rows, 1):
            for j in range(0, loop_num, 1):
                if j < loop_num - 1:
                    pto.vbitsort(
                        pto.addptr(dst_ptr, i * dst_stride + j * REPEAT_MAX * BLOCK_SIZE * type_coef),
                        pto.addptr(src_ptr, i * src_stride + j * REPEAT_MAX * BLOCK_SIZE),
                        pto.addptr(idx_ptr, i * idx_stride + j * REPEAT_MAX * BLOCK_SIZE),
                        REPEAT_MAX
                    )
                else:
                    if src_tail_repeat_num > 0:
                        sort_repeat_num = 0
                        if src_tail_repeat_num > 1:
                            sort_repeat_num = src_tail_repeat_num - 1
                            
                        pto.vbitsort(
                            pto.addptr(dst_ptr, i * dst_stride + j * REPEAT_MAX * BLOCK_SIZE * type_coef),
                            pto.addptr(src_ptr, i * src_stride + j * REPEAT_MAX * BLOCK_SIZE),
                            pto.addptr(idx_ptr, i * idx_stride + j * REPEAT_MAX * BLOCK_SIZE),
                            sort_repeat_num
                        )

                    tail_src_offset = (j * REPEAT_MAX + (src_tail_repeat_num - 1)) * BLOCK_SIZE
                    tail_dst_offset = (j * REPEAT_MAX + (src_tail_repeat_num - 1)) * BLOCK_SIZE * type_coef
                    len_burst = (src_tail_per_row * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

                    pto.copy_ubuf_to_ubuf(
                        pto.addptr(src_ptr, i * src_stride + tail_src_offset),
                        tmp_ptr,
                        0, 1, len_burst, 0, 0
                    )

                    tmp_last_offset = ((src_tail_per_row + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) - BLOCK_SIZE

                    vec = pto.vlds(tmp[0, tmp_last_offset:])
                    pad_mask, _ = pto.make_mask(dtype, BLOCK_SIZE - src_tail_per_row)
                    vec = pto.vdup(min_val, pad_mask)
                    pto.vsts(vec, tmp[0, tmp_last_offset:], pad_mask)

                    pto.vbitsort(
                        pto.addptr(dst_ptr, i * dst_stride + tail_dst_offset),
                        tmp_ptr,
                        pto.addptr(idx_ptr, i * idx_stride + tail_src_offset),
                        1
                    )

    return