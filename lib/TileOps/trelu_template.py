# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trelu (Elementwise ReLU)"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.trelu",
    dtypes=[(pto.f16, pto.f16), (pto.f32, pto.f32), (pto.i32, pto.i32)],
    advanced=True
)
def template_trelu(src: pto.Tile, dst: pto.Tile):
    """Elementwise ReLU: dst = max(0, src).
    
    Semantics:
    For each element (i, j):
        dst[i, j] = max(0, src[i, j])
    
    Supported data types: f16, f32, i32
    """
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    
    lanes = pto.get_lanes(dtype)
    
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            src_vec = pto.vlds(src[row, col:])
            result = pto.vrelu(src_vec, mask)
            pto.vsts(result, dst[row, col:], mask)
    return