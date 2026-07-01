# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowexpandmax"""

import sys
from pathlib import Path
import tilelang_dsl as pto


def _constraint_trowexpandmax_row_major(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint for RowMajor layout trowexpandmax template."""
    # All tiles must be RowMajor layout
    src0_row_major = src0.config.b_layout == pto.BLayout.ROW_MAJOR
    src1_row_major = src1.config.b_layout == pto.BLayout.ROW_MAJOR
    dst_row_major = dst.config.b_layout == pto.BLayout.ROW_MAJOR
    return src0_row_major and src1_row_major and dst_row_major


@pto.vkernel(
    target="a5",
    op="pto.trowexpandmax",
    dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat), (pto.AnyInt, pto.AnyInt, pto.AnyInt)],
    constraints=[_constraint_trowexpandmax_row_major],
)
def template_trowexpandmax(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpandmax.

    Compute element-wise max of each row of src0 with a per-row scalar from src1[row, 0].
    Semantics: dst[row, col] = max(src0[row, col], src1[row, 0])
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            # Load the scalar vector from src1[row, :]
            # For row-major src1, valid_shape[1] is 32/sizeof(dtype) (e.g., 8 for f32)
            # vdup broadcasts the first element to the full vector width
            scalar_vec = pto.vlds(src1[row, :])
            broadcasted = pto.vdup(scalar_vec, mask)
            lhs = pto.vlds(src0[row, col:])
            result = pto.vmax(lhs, broadcasted, mask)
            pto.vsts(result, dst[row, col:], mask)
    return