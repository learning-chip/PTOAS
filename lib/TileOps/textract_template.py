# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.textract and pto.textract_fp (A5).

TEXTRACT extracts a sub-tile window from src into dst at offset (indexRow, indexCol):
    dst[i, j] = src[i + indexRow, j + indexCol]

Supported data-flow directions (A5):

Cube path (Mat -> Left/Right):
  The mte_l1_l0a/l0b transpose flag is set based on src tile layout:
    - Same fractal (src.col_major blayout + src.row_major slayout):
      transpose=False — direct extraction (TExtractToA / TExtractToACompact)
    - Cross fractal (src.row_major blayout + src.col_major slayout):
      transpose=True — transposed extraction (TExtractToATransCompact / TExtractToA<true>)
  Offset extraction (indexRow/indexCol != 0) is supported via start_row/start_col
  keywords on mte_l1_l0a/l0b (PR #469).

Fix-pipe path (Acc -> Mat with FP quantization):
  - textract_fp   : PIPE_FIX, uses pto.mte_l0c_l1 with pre_quant keyword
  Currently constrained to indexRow=0, indexCol=0. The index_row/index_col
  parameters are accepted but unused in the lowering because the start
  position for fix-pipe copy is implicit (whole-tile extraction from L0C).
  Future work: pass index offset to mte_l0c_l1 when PTO-ISA supports it.

Vector path (Vec -> Vec):
  - Vec -> Vec ND : PIPE_V, uses pto.vlds/vsts subscript syntax
  Note: Uses vector load/store instead of DMA (copy_ubuf_to_ubuf) because
  textract requires arbitrary offset support (index_row/index_col), which
  DMA block copy cannot provide.

Blocked paths:
  - Mat -> ScaleLeft/ScaleRight : mte_l1_l0a_mx/l0b_mx reject memory_space="scaling" dst
  - Vec -> Mat                  : now available via pto.mte_ub_l1 (PR #405)
"""

import tilelang_dsl as pto

_FP_QUANT_MODE_MAP = {
    (pto.f32, pto.si8): "qf322b8_pre_vec",
    (pto.f32, pto.ui8): "qf322b8_pre_vec",
    (pto.f32, pto.f16): "qf322f16_pre_vec",
    (pto.f32, pto.bf16): "qf322bf16_pre_vec",
    (pto.f32, pto.f32): "qf322f32_pre_vec",
    (pto.si32, pto.si8): "req8_vec",
    (pto.si32, pto.ui8): "req8_vec",
    (pto.si32, pto.f16): "deqf16_vec",
    (pto.si32, pto.bf16): "qs322bf16_pre_vec",
}


def _is_same_fractal_as_left(src) -> bool:
    if src.config is None:
        return True
    return (src.config.b_layout != pto.BLayout.ROW_MAJOR
            and src.config.s_layout == pto.SLayout.ROW_MAJOR)


def _is_cross_fractal_as_left(src) -> bool:
    if src.config is None:
        return False
    return (src.config.b_layout == pto.BLayout.ROW_MAJOR
            and src.config.s_layout == pto.SLayout.COL_MAJOR)


def _textract_cube_dst_is_left_same_fractal(src, index_row, index_col, dst) -> bool:
    dst_ms = dst.memory_space
    if not (dst_ms == "left" if isinstance(dst_ms, str)
            else dst_ms.value == "left"):
        return False
    if not _is_same_fractal_as_left(src):
        return False
    return True


def _textract_cube_dst_is_left_cross_fractal(src, index_row, index_col, dst) -> bool:
    dst_ms = dst.memory_space
    if not (dst_ms == "left" if isinstance(dst_ms, str)
            else dst_ms.value == "left"):
        return False
    if not _is_cross_fractal_as_left(src):
        return False
    return True


def _is_same_fractal_as_right(src) -> bool:
    if src.config is None:
        return True
    return (src.config.b_layout != pto.BLayout.ROW_MAJOR
            and src.config.s_layout == pto.SLayout.ROW_MAJOR)


def _is_cross_fractal_as_right(src) -> bool:
    if src.config is None:
        return False
    return (src.config.b_layout == pto.BLayout.ROW_MAJOR
            and src.config.s_layout == pto.SLayout.COL_MAJOR)


def _textract_cube_dst_is_right_same_fractal(src, index_row, index_col, dst) -> bool:
    dst_ms = dst.memory_space
    if not (dst_ms == "right" if isinstance(dst_ms, str)
            else dst_ms.value == "right"):
        return False
    if not _is_same_fractal_as_right(src):
        return False
    return True


def _textract_cube_dst_is_right_cross_fractal(src, index_row, index_col, dst) -> bool:
    dst_ms = dst.memory_space
    if not (dst_ms == "right" if isinstance(dst_ms, str)
            else dst_ms.value == "right"):
        return False
    if not _is_cross_fractal_as_right(src):
        return False
    return True


def _textract_vec2vec_nd_constraint(src, index_row, index_col, dst) -> bool:
    src_ms = src.memory_space
    dst_ms = dst.memory_space
    src_is_ub = (src_ms == "ub" if isinstance(src_ms, str)
                 else src_ms.value == "ub")
    dst_is_ub = (dst_ms == "ub" if isinstance(dst_ms, str)
                 else dst_ms.value == "ub")
    if not (src_is_ub and dst_is_ub):
        return False
    if src.config is None or dst.config is None:
        return False
    if src.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src.config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if dst.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if dst.config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if src.dtype != dst.dtype:
        return False
    return True


def _textract_fp_acc2mat_constraint(src, fp, index_row, index_col, dst) -> bool:
    src_ms = src.memory_space
    fp_ms = fp.memory_space
    dst_ms = dst.memory_space
    if not (src_ms == "acc" if isinstance(src_ms, str)
            else src_ms.value == "acc"):
        return False
    if not (fp_ms == "scaling" if isinstance(fp_ms, str)
            else fp_ms.value == "scaling"):
        return False
    if not (dst_ms == "mat" if isinstance(dst_ms, str)
            else dst_ms.value == "mat"):
        return False
    return (src.dtype, dst.dtype) in _FP_QUANT_MODE_MAP


def _make_fp_constraint(src_dtype, dst_dtype):
    def _fp_constraint(src, fp, index_row, index_col, dst) -> bool:
        src_ms = src.memory_space
        fp_ms = fp.memory_space
        dst_ms = dst.memory_space
        if not (src_ms == "acc" if isinstance(src_ms, str)
                else src_ms.value == "acc"):
            return False
        if not (fp_ms == "scaling" if isinstance(fp_ms, str)
                else fp_ms.value == "scaling"):
            return False
        if not (dst_ms == "mat" if isinstance(dst_ms, str)
                else dst_ms.value == "mat"):
            return False
        return src.dtype == src_dtype and dst.dtype == dst_dtype
    return _fp_constraint


@pto.ckernel(
    target="a5",
    op="pto.textract",
    constraints=[_textract_cube_dst_is_left_same_fractal],
)
def template_textract_mat2left(src: pto.Tile,
                                index_row: pto.i32, index_col: pto.i32,
                                dst: pto.Tile):
    m, k = dst.valid_shape
    pto.mte_l1_l0a(src.as_ptr(), dst.as_ptr(), m, k,
                    start_row=index_row, start_col=index_col)
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract",
    constraints=[_textract_cube_dst_is_left_cross_fractal],
)
def template_textract_mat2left_trans(src: pto.Tile,
                                      index_row: pto.i32, index_col: pto.i32,
                                      dst: pto.Tile):
    m, k = dst.valid_shape
    pto.mte_l1_l0a(src.as_ptr(), dst.as_ptr(), m, k,
                    start_row=index_row, start_col=index_col, transpose=True)
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract",
    constraints=[_textract_cube_dst_is_right_same_fractal],
)
def template_textract_mat2right(src: pto.Tile,
                                 index_row: pto.i32, index_col: pto.i32,
                                 dst: pto.Tile):
    k, n = dst.valid_shape
    pto.mte_l1_l0b(src.as_ptr(), dst.as_ptr(), k, n,
                    start_row=index_row, start_col=index_col)
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract",
    constraints=[_textract_cube_dst_is_right_cross_fractal],
)
def template_textract_mat2right_trans(src: pto.Tile,
                                       index_row: pto.i32, index_col: pto.i32,
                                       dst: pto.Tile):
    k, n = dst.valid_shape
    pto.mte_l1_l0b(src.as_ptr(), dst.as_ptr(), k, n,
                    start_row=index_row, start_col=index_col, transpose=True)
    return None


@pto.vkernel(
    target="a5",
    op="pto.textract",
    advanced=True,
    constraints=[_textract_vec2vec_nd_constraint],
)
def template_textract_vec2vec_nd(src: pto.Tile,
                                  index_row: pto.i32, index_col: pto.i32,
                                  dst: pto.Tile):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[index_row + row, index_col + col:])
            pto.vsts(data, dst[row, col:], mask)

    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.f32, pto.f32, pto.i32, pto.i32, pto.si8),),
    constraints=[_make_fp_constraint(pto.f32, pto.si8)],
)
def template_textract_fp_f32_si8(src: pto.Tile, fp: pto.Tile,
                                  index_row: pto.i32, index_col: pto.i32,
                                  dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "qf322b8_pre_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.f32, pto.f32, pto.i32, pto.i32, pto.ui8),),
    constraints=[_make_fp_constraint(pto.f32, pto.ui8)],
)
def template_textract_fp_f32_ui8(src: pto.Tile, fp: pto.Tile,
                                  index_row: pto.i32, index_col: pto.i32,
                                  dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "qf322b8_pre_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.f32, pto.f32, pto.i32, pto.i32, pto.f16),),
    constraints=[_make_fp_constraint(pto.f32, pto.f16)],
)
def template_textract_fp_f32_f16(src: pto.Tile, fp: pto.Tile,
                                  index_row: pto.i32, index_col: pto.i32,
                                  dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "qf322f16_pre_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.f32, pto.f32, pto.i32, pto.i32, pto.bf16),),
    constraints=[_make_fp_constraint(pto.f32, pto.bf16)],
)
def template_textract_fp_f32_bf16(src: pto.Tile, fp: pto.Tile,
                                   index_row: pto.i32, index_col: pto.i32,
                                   dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "qf322bf16_pre_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.f32, pto.f32, pto.i32, pto.i32, pto.f32),),
    constraints=[_make_fp_constraint(pto.f32, pto.f32)],
)
def template_textract_fp_f32_f32(src: pto.Tile, fp: pto.Tile,
                                  index_row: pto.i32, index_col: pto.i32,
                                  dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "qf322f32_pre_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.si32, pto.f32, pto.i32, pto.i32, pto.si8),),
    constraints=[_make_fp_constraint(pto.si32, pto.si8)],
)
def template_textract_fp_si32_si8(src: pto.Tile, fp: pto.Tile,
                                   index_row: pto.i32, index_col: pto.i32,
                                   dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "req8_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.si32, pto.f32, pto.i32, pto.i32, pto.ui8),),
    constraints=[_make_fp_constraint(pto.si32, pto.ui8)],
)
def template_textract_fp_si32_ui8(src: pto.Tile, fp: pto.Tile,
                                   index_row: pto.i32, index_col: pto.i32,
                                   dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "req8_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.si32, pto.f32, pto.i32, pto.i32, pto.f16),),
    constraints=[_make_fp_constraint(pto.si32, pto.f16)],
)
def template_textract_fp_si32_f16(src: pto.Tile, fp: pto.Tile,
                                   index_row: pto.i32, index_col: pto.i32,
                                   dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "deqf16_vec"))
    return None


@pto.ckernel(
    target="a5",
    op="pto.textract_fp",
    dtypes=((pto.si32, pto.f32, pto.i32, pto.i32, pto.bf16),),
    constraints=[_make_fp_constraint(pto.si32, pto.bf16)],
)
def template_textract_fp_si32_bf16(src: pto.Tile, fp: pto.Tile,
                                    index_row: pto.i32, index_col: pto.i32,
                                    dst: pto.Tile):
    m, n = dst.valid_shape
    src_stride = src.shape[0]
    dst_stride = dst.shape[1]
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
                    pre_quant=(fp.as_ptr(), "qs322bf16_pre_vec"))
    return None