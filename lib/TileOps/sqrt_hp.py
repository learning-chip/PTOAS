# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tilelang_dsl as pto

@pto.inline_proc
def _tl_sqrt_precision_f16(src, mask):
    multiply_factor0 = pto.f16("0x6c00")
    multiply_factor1 = pto.f16("0x2400")
    subnormal_threshold = pto.f16("0x03ff")
    
    subnormal_mask = pto.vcmps(src, subnormal_threshold, mask, pto.CmpMode.LT)
    
    tmp = pto.vmuls(src, multiply_factor0, subnormal_mask)
    src_adjusted = pto.vsel(tmp, src, subnormal_mask)
    
    dst = pto.vsqrt(src_adjusted, mask)
    
    tmp = pto.vmuls(dst, multiply_factor1, subnormal_mask)
    result = pto.vsel(tmp, dst, subnormal_mask)
    
    return result


@pto.inline_proc
def _tl_sqrt_precision_f32(src, mask):
    multiply_factor0 = pto.f32(16777216.0)
    multiply_factor1 = pto.f32(0.000244140625)
    subnormal_bound = pto.f32(1.0)
    half_factor = pto.f32(0.5)
    neg_one = pto.f32(-1.0)
    
    subnormal_mask = pto.vcmps(src, subnormal_bound, mask, pto.CmpMode.LT)
    
    tmp = pto.vmuls(src, multiply_factor0, subnormal_mask)
    src_adjusted = pto.vsel(tmp, src, subnormal_mask)
    
    reg_one = pto.vbr(pto.f32(1.0))
    tmp_sqrt = pto.vsqrt(src_adjusted, mask)
    dst = pto.vdiv(reg_one, tmp_sqrt, mask)
    
    reg_neg_one = pto.vmuls(dst, neg_one, mask)
    err = pto.vmul(dst, src_adjusted, mask)
    reg_one_adj = pto.vmula(reg_one, err, reg_neg_one, mask)
    tmp_half = pto.vmuls(dst, half_factor, mask)
    dst = pto.vmula(dst, reg_one_adj, tmp_half, mask)
    
    res = pto.vmul(dst, src_adjusted, mask)
    tmp_neg = pto.vmuls(res, neg_one, mask)
    err = pto.vmula(src_adjusted, res, tmp_neg, mask)
    tmp_half = pto.vmuls(dst, half_factor, mask)
    tmp = pto.vmul(err, tmp_half, mask)
    tmp = pto.vadd(tmp, res, mask)
    
    tmp_scaled = pto.vmuls(tmp, multiply_factor1, mask)
    result = pto.vsel(tmp_scaled, tmp, subnormal_mask)
    
    pos_inf = pto.ui32(0x7f800000)
    neg_zero = pto.ui32(0x80000000)
    
    src_as_u32 = pto.vbitcast(src_adjusted, pto.ui32)
    is_inf_mask = pto.vcmps(src_as_u32, pos_inf, mask, pto.CmpMode.EQ)
    src_with_sign = pto.vor(src_as_u32, pto.vbr(neg_zero), mask)
    is_zero_mask = pto.vcmps(src_with_sign, neg_zero, mask, pto.CmpMode.EQ)
    special_mask = pto.por(is_zero_mask, is_inf_mask, mask)
    
    result = pto.vsel(src_adjusted, result, special_mask)
    
    return result


@pto.inline_proc
def _tl_sqrt_precision(src, mask, dtype):
    if pto.constexpr(dtype == pto.f16):
        result = _tl_sqrt_precision_f16(src, mask)
    else:
        result = _tl_sqrt_precision_f32(src, mask)
    return result