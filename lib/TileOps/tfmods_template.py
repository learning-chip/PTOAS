# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tfmods

Note: A5 hardware implements tfmods as:
  - float: dst = src - trunc(src / scalar) * scalar (f32 precision)
  - half:  dst = src - trunc(src / scalar) * scalar (computed in f32 precision, then converted back to f16)
  - integer: dst = src - (src / scalar) * scalar (integer division already truncates)

f16 path: convert f16 to f32 (even/odd), compute in f32 with vtrc(ROUND_Z),
convert back to f16 with ROUND_Z, merge with vor.
"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tfmods",
    dtypes=[
        (pto.f32, pto.f32, pto.f32),
        (pto.f16, pto.f16, pto.f16),
        (pto.i32, pto.i32, pto.i32),
        (pto.i16, pto.i16, pto.i16),
    ],
    advanced=True,
)
def template_tfmods(src: pto.Tile, scalar: pto.AnyType, dst: pto.Tile):
    """dst = src - trunc(src / scalar) * scalar"""
    dtype = src.element_type
    valid_rows, valid_cols = src.valid_shape

    if pto.constexpr(dtype == pto.f32):
        # f32 path: direct f32 computation
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
                mask, remained = pto.make_mask(pto.f32, remained)
                vec = pto.vlds(src[row, col:])
                scalar_vec = pto.vbr(scalar)
                quotient = pto.vdiv(vec, scalar_vec, mask)
                truncated = pto.vtrc(quotient, mask, rnd="Z")
                product = pto.vmuls(truncated, scalar, mask)
                result = pto.vsub(vec, product, mask)
                pto.vsts(result, dst[row, col:], mask)
    elif pto.constexpr(dtype == pto.f16):
        # f16 path: compute in f32 precision, then convert back to f16
        full_mask_b16 = pto.make_mask(pto.f16, pto.PAT.ALL)
        full_mask_b32 = pto.make_mask(pto.f32, pto.PAT.ALL)
        scalar_vec_f16 = pto.vbr(scalar)
        scalar_f32_vec = pto.vcvt(scalar_vec_f16, pto.f32, full_mask_b16, part=pto.VcvtPartMode.EVEN)

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            remained_f32 = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(pto.f16)):
                mask_f16, remained = pto.make_mask(pto.f16, remained)
                mask_f32, remained_f32 = pto.make_mask(pto.f32, remained_f32)
                vec = pto.vlds(src[row, col:])

                # Convert f16 to f32 (even and odd parts)
                vec_even = pto.vcvt(vec, pto.f32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
                vec_odd = pto.vcvt(vec, pto.f32, full_mask_b16, part=pto.VcvtPartMode.ODD)

                # Even part: f32 computation
                scalar_f32 = pto.f32(scalar)
                quotient_even = pto.vdiv(vec_even, scalar_f32_vec, mask_f32)
                truncated_even = pto.vtrc(quotient_even, mask_f32, rnd="Z")
                product_even = pto.vmuls(truncated_even, scalar_f32, mask_f32)
                result_even = pto.vsub(vec_even, product_even, mask_f32)

                # Odd part: f32 computation
                quotient_odd = pto.vdiv(vec_odd, scalar_f32_vec, mask_f32)
                truncated_odd = pto.vtrc(quotient_odd, mask_f32, rnd="Z")
                product_odd = pto.vmuls(truncated_odd, scalar_f32, mask_f32)
                result_odd = pto.vsub(vec_odd, product_odd, mask_f32)

                # Convert f32 results back to f16 with ROUND_Z + saturation
                result_f16_even = pto.vcvt(result_even, pto.f16, full_mask_b32,
                                           rnd=pto.VcvtRoundMode.Z,
                                           sat=pto.VcvtSatMode.SAT,
                                           part=pto.VcvtPartMode.EVEN)
                result_f16_odd = pto.vcvt(result_odd, pto.f16, full_mask_b32,
                                          rnd=pto.VcvtRoundMode.Z,
                                          sat=pto.VcvtSatMode.SAT,
                                          part=pto.VcvtPartMode.ODD)

                # Merge even and odd parts
                result_f16 = pto.vor(result_f16_even, result_f16_odd, mask_f16)
                pto.vsts(result_f16, dst[row, col:], mask_f16)
    else:
        # Integer path: vdiv already truncates towards zero, no vtrc needed
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, pto.get_lanes(dtype)):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                scalar_vec = pto.vbr(scalar)
                quotient = pto.vdiv(vec, scalar_vec, mask)
                product = pto.vmuls(quotient, scalar, mask)
                result = pto.vsub(vec, product, mask)
                pto.vsts(result, dst[row, col:], mask)
    return
