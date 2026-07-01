# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tilelang_dsl as pto

@pto.inline_proc
def _tl_exp_precision(src, mask, dtype):
    if pto.constexpr(dtype == pto.f16):
        subnormal_threshold = pto.f16("0x03ff")
        two_val = pto.f16(2.0)
    else:
        subnormal_threshold = pto.f32("0x007FFFFF")
        two_val = pto.f32(2.0)

    dst = pto.vexp(src, mask)

    subnormal_mask = pto.vcmps(dst, subnormal_threshold, mask, pto.CmpMode.LE)

    reg_two = pto.vbr(two_val)
    tmp = pto.vdiv(src, reg_two, subnormal_mask)
    tmp = pto.vexp(tmp, subnormal_mask)
    tmp = pto.vmul(tmp, tmp, subnormal_mask)

    result = pto.vsel(tmp, dst, subnormal_mask)
    return result
