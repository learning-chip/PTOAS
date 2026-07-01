# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tlog"""

import tilelang_dsl as pto


@pto.inline_proc
def _tlog_high_precision(src: pto.Tile, dst: pto.Tile, dtype, valid_rows, valid_cols):
    if pto.constexpr(dtype == pto.f16):
        subnormal_threshold = pto.f16("0x03FF")
        mul_factor = pto.f16("0x6400")
        compensation = pto.f16(-6.931471805599453094172)
    elif pto.constexpr(dtype == pto.f32):
        subnormal_threshold = pto.f32("0x007FFFFF")
        mul_factor = pto.f32("0x4B000000")
        compensation = pto.f32(-15.9423851528787421)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vinput = pto.vlds(src[row, col:])
            cmp_mask = pto.vcmps(vinput, subnormal_threshold, mask, pto.CmpMode.LT)
            scaled = pto.vmuls(vinput, mul_factor, mask)
            selected_input = pto.vsel(scaled, vinput, cmp_mask)
            log_result = pto.vln(selected_input, mask)
            compensated = pto.vadds(log_result, compensation, mask)
            result = pto.vsel(compensated, log_result, cmp_mask)
            pto.vsts(result, dst[row, col:], mask)
    return None


@pto.inline_proc
def _tlog_default(src: pto.Tile, dst: pto.Tile, dtype, valid_rows, valid_cols):
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vinput = pto.vlds(src[row, col:])
            result = pto.vln(vinput, mask)
            pto.vsts(result, dst[row, col:], mask)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tlog",
    advanced=True
)
def template_tlog(src: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    precision_type = pto.get_op_attr("precisionType", "default")

    if pto.constexpr(precision_type == "high_precision"):
        _tlog_high_precision(src, dst, dtype, valid_rows, valid_cols)
    else:
        _tlog_default(src, dst, dtype, valid_rows, valid_cols)
    return