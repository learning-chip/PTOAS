# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
``pto`` – the public DSL namespace.

Import as::

    import pto

or as the sub-namespace ``pto`` from the ptodsl package::

    from ptodsl import pto

All user-facing symbols live here.  Low-level MLIR bindings are accessed
internally as ``_pto`` (``from mlir.dialects import pto as _pto``).
"""

# ── Types ─────────────────────────────────────────────────────────────────────
from ._types import (           # noqa: F401
    float32, float16,
    int8, int16, int32, int64,
    index,
    ptr, vreg_type, mask_type,
    tile_buf_type, tensor_view_type, part_tensor_view_type,
    _resolve,
)

# ── Operations ────────────────────────────────────────────────────────────────
from ._ops import (             # noqa: F401
    const,
    castptr, addptr,
    vlds, vbrc_load, vsts, vsts_1pt,
    plt_b32, pset_b32,
    vadd, vmul, vmax, vdiv,
    vcmax, vcadd, vdup, vexpdif,
    make_tensor_view, partition_view,
    alloc_tile, tload, tstore, tile_ptr,
    get_block_idx, barrier_all,
    set_flag, wait_flag,
)

# ── Control flow ──────────────────────────────────────────────────────────────
from ._control_flow import (    # noqa: F401
    vecscope,
    for_, fori_loop,
    if_, cond,
    yield_,
)

# ── Decorator ─────────────────────────────────────────────────────────────────
from ._module import to_ir, KernelHandle      # noqa: F401

# ── Scalar sub-namespace ──────────────────────────────────────────────────────
from . import scalar            # noqa: F401
