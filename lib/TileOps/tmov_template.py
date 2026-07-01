# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmov - tile data movement

This template implements UB2UB ND2ND tile data movement:
  - UB2UB: Both src and dst must be in Unified Buffer (memory_space="ub")
  - ND2ND: Both tiles must have N-dimensional layout (s_layout=NONE_BOX)

For other transfer scenarios (GM2UB, UB2GM, or specialized layouts),
different templates/implementation paths should be used.
"""

import tilelang_dsl as pto


def _tmov_ub2ub_nd2nd_constraint(src: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint: Both src and dst must be UB location with ND layout.

    Supported scenario:
      - UB2UB: src and dst both have memory_space="ub"
      - ND2ND: src and dst both have s_layout=NONE_BOX (N-dimensional format)

    Unsupported scenarios (require different implementation paths):
      - GM2UB: src from Global Memory, dst to Unified Buffer
      - UB2GM: src from Unified Buffer, dst to Global Memory
      - Specialized layouts (e.g., cube formats with non-NONE_BOX s_layout)
    """
    # Check memory_space for both tiles (UB2UB constraint)
    src_ms = src.memory_space
    dst_ms = dst.memory_space
    if isinstance(src_ms, str):
        src_is_ub = src_ms == "ub"
    else:
        src_is_ub = hasattr(src_ms, "value") and src_ms.value == "ub"
    if isinstance(dst_ms, str):
        dst_is_ub = dst_ms == "ub"
    else:
        dst_is_ub = hasattr(dst_ms, "value") and dst_ms.value == "ub"

    if not (src_is_ub and dst_is_ub):
        return False

    # Check s_layout for both tiles (ND2ND constraint)
    # ND layout uses NONE_BOX, specialized layouts (cube, etc.) use different values
    src_config = src.config
    dst_config = dst.config
    if src_config is None or dst_config is None:
        return False
    if src_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if dst_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if pto.bytewidth(src.dtype) != pto.bytewidth(dst.dtype):
        return False

    return True


@pto.vkernel(
    target="a5",
    op="pto.tmov",
    constraints=[_tmov_ub2ub_nd2nd_constraint],
    advanced=True,
)
def template_tmov_basic(src: pto.Tile, dst: pto.Tile):
    """Basic tile-to-tile data movement using vlds/vsts.

    Based on TMovVecToVec in TMov.hpp (lines 378-405):
    - Iterate over valid_row rows
    - Each row processed in chunks of nRepeatElem elements
    - Use predicate mask for partial chunks

    Args:
        src: Source tile (Vec location)
        dst: Destination tile (Vec location)
    """
    src_dtype = src.element_type
    dst_dtype = dst.element_type
    lanes = pto.get_lanes(dst_dtype)

    # Use dst.valid_shape as the copy dimensions
    # The dst tile defines how many elements to write
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dst_dtype, remained)
            data = pto.vlds(src[row, col:])
            if pto.constexpr(src_dtype != dst_dtype):
                data = pto.vbitcast(data, dst_dtype)
            pto.vsts(data, dst[row, col:], mask)

    return None
