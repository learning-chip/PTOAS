# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tfmod

Aligned with pto-isa/include/pto/npu/a5/TFMod.hpp:
- float: vdiv -> vtrc(ROUND_Z) -> vmul -> vsub
- half:  vcvt(half->float, PART_EVEN/ODD) -> vdiv -> vtrc(ROUND_Z) -> vmul -> vsub 
         -> vcvt(float->half, ROUND_Z, RS_ENABLE, PART_EVEN/ODD) -> vor
- other (i16/ui16): vdiv -> vmul -> vsub (no vtrc, integer div is trunc by nature)
"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tfmod",
    dtypes=[
        (pto.f32, pto.f32, pto.f32),
        (pto.f16, pto.f16, pto.f16),
        (pto.i16, pto.i16, pto.i16),
        (pto.ui16, pto.ui16, pto.ui16),
    ],
)
def template_tfmod(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            
            if pto.constexpr(dtype == pto.f32):
                quotient = pto.vdiv(lhs, rhs, mask)
                quotient = pto.vtrc(quotient, mask, rnd=pto.VcvtRoundMode.Z)
                truncated_mul = pto.vmul(quotient, rhs, mask)
                result = pto.vsub(lhs, truncated_mul, mask)
            elif pto.constexpr(dtype == pto.f16):
                lhs_even = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
                rhs_even = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
                quotient_even = pto.vdiv(lhs_even, rhs_even, mask)
                quotient_even = pto.vtrc(quotient_even, mask, rnd=pto.VcvtRoundMode.Z)
                truncated_mul_even = pto.vmul(quotient_even, rhs_even, mask)
                result_even = pto.vsub(lhs_even, truncated_mul_even, mask)
                dst_even = pto.vcvt(result_even, pto.f16, mask, rnd=pto.VcvtRoundMode.Z, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.EVEN)
                
                lhs_odd = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
                rhs_odd = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
                quotient_odd = pto.vdiv(lhs_odd, rhs_odd, mask)
                quotient_odd = pto.vtrc(quotient_odd, mask, rnd=pto.VcvtRoundMode.Z)
                truncated_mul_odd = pto.vmul(quotient_odd, rhs_odd, mask)
                result_odd = pto.vsub(lhs_odd, truncated_mul_odd, mask)
                dst_odd = pto.vcvt(result_odd, pto.f16, mask, rnd=pto.VcvtRoundMode.Z, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.ODD)
                
                result = pto.vor(dst_even, dst_odd, mask)
            else:
                quotient = pto.vdiv(lhs, rhs, mask)
                truncated_mul = pto.vmul(quotient, rhs, mask)
                result = pto.vsub(lhs, truncated_mul, mask)
            
            pto.vsts(result, dst[row, col:], mask)
    return