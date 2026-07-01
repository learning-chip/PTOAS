# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tilelang_dsl as pto


@pto.inline_proc
def _tl_soft_vdiv_u8(vec, scalar_vec, mask):
    zero = pto.ui8(0)
    zero_q = pto.ui8(0xFF)
    full_mask_b8 = pto.pset_b8(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)
    active_low = pto.punpack(active_mask, pto.PredicatePart.LOWER)
    active_high = pto.punpack(active_mask, pto.PredicatePart.HIGHER)

    vec_low = pto.vzunpack(vec, 0)
    vec_high = pto.vzunpack(vec, 1)
    scalar_low = pto.vzunpack(scalar_vec, 0)
    scalar_high = pto.vzunpack(scalar_vec, 1)

    q_low = _tl_soft_vdiv_u16(vec_low, scalar_low, active_low)
    q_high = _tl_soft_vdiv_u16(vec_high, scalar_high, active_high)
    packed_low = pto.vpack(q_low, pto.PredicatePart.LOWER)
    packed_high = pto.vpack(q_high, pto.PredicatePart.HIGHER)
    q = pto.vor(packed_low, packed_high, full_mask_b8)
    return pto.vsel(pto.vbr(zero_q), q, zero_mask)


@pto.inline_proc
def _tl_soft_vdiv_i8(vec, scalar_vec, mask):
    zero = pto.i8(0)
    neg_one = pto.i8(-1)
    full_mask_b8 = pto.pset_b8(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)
    active_low = pto.punpack(active_mask, pto.PredicatePart.LOWER)
    active_high = pto.punpack(active_mask, pto.PredicatePart.HIGHER)

    vec_low = pto.vsunpack(vec, 0)
    vec_high = pto.vsunpack(vec, 1)
    scalar_low = pto.vsunpack(scalar_vec, 0)
    scalar_high = pto.vsunpack(scalar_vec, 1)

    q_low = _tl_soft_vdiv_i16(vec_low, scalar_low, active_low)
    q_high = _tl_soft_vdiv_i16(vec_high, scalar_high, active_high)
    packed_low = pto.vpack(q_low, pto.PredicatePart.LOWER)
    packed_high = pto.vpack(q_high, pto.PredicatePart.HIGHER)
    q = pto.vbitcast(pto.vor(packed_low, packed_high, full_mask_b8), pto.i8)
    return pto.vsel(pto.vbr(neg_one), q, zero_mask)


@pto.inline_proc
def _tl_soft_vdiv_u16(vec, scalar_vec, mask):
    zero = pto.ui16(0)
    one = pto.ui16(1)
    fp32_one = pto.f32(1.0)
    full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
    full_mask_b32 = pto.pset_b32(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    zero_u16 = pto.vbr(zero)
    vy_lower_u16, vy_higher_u16 = pto.vintlv(scalar_vec, zero_u16)
    vy_lower_u32 = pto.vcvt(vy_lower_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    vy_higher_u32 = pto.vcvt(vy_higher_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    active_low = pto.vcmps(vy_lower_u32, pto.ui32(0), full_mask_b32, pto.CmpMode.NE)
    active_high = pto.vcmps(vy_higher_u32, pto.ui32(0), full_mask_b32, pto.CmpMode.NE)
    vy_lower_f32 = pto.vcvt(pto.vbitcast(vy_lower_u32, pto.i32), pto.f32, active_low, rnd=pto.VcvtRoundMode.F)
    vy_higher_f32 = pto.vcvt(pto.vbitcast(vy_higher_u32, pto.i32), pto.f32, active_high, rnd=pto.VcvtRoundMode.F)

    vy_rec_lower = pto.vdiv(pto.vbr(fp32_one), vy_lower_f32, active_low)
    vy_rec_higher = pto.vdiv(pto.vbr(fp32_one), vy_higher_f32, active_high)
    vy_scale_lower = pto.vmul(vy_rec_lower, pto.vbr(pto.f32(65536.0)), active_low)
    vy_scale_higher = pto.vmul(vy_rec_higher, pto.vbr(pto.f32(65536.0)), active_high)

    v_lower_i32 = pto.vcvt(
        vy_scale_lower,
        pto.i32,
        active_low,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
    )
    v_higher_i32 = pto.vcvt(
        vy_scale_higher,
        pto.i32,
        active_high,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
    )
    v_lower_u32 = pto.vbitcast(v_lower_i32, pto.ui32)
    v_higher_u32 = pto.vbitcast(v_higher_i32, pto.ui32)

    vx_lower_u16, vx_higher_u16 = pto.vintlv(vec, zero_u16)
    vx_lower_u32 = pto.vcvt(vx_lower_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    vx_higher_u32 = pto.vcvt(vx_higher_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    q_tmp_lower = pto.vmul(v_lower_u32, vx_lower_u32, active_low)
    q_tmp_higher = pto.vmul(v_higher_u32, vx_higher_u32, active_high)
    _q_lower, q_tmp = pto.vdintlv(pto.vbitcast(q_tmp_lower, pto.ui16), pto.vbitcast(q_tmp_higher, pto.ui16))

    yq_tmp = pto.vmul(q_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsub(vec, yq_tmp, active_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    zero_q = pto.vbr(pto.ui16(0xFFFF))
    return pto.vsel(zero_q, q_tmp, zero_mask)


@pto.inline_proc
def _tl_soft_vdiv_i16(vec, scalar_vec, mask):
    zero = pto.i16(0)
    neg_one = pto.i16(-1)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui16)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui16)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    p_pos = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    q_abs = _tl_soft_vdiv_u16(abs_x, abs_y, active_mask)
    neg_q = pto.vneg(pto.vbitcast(q_abs, pto.i16), active_mask)
    q = pto.vsel(pto.vbitcast(q_abs, pto.i16), neg_q, p_pos)
    return pto.vsel(pto.vbr(neg_one), q, zero_mask)


@pto.inline_proc
def _tl_soft_vmod_u8(vec, scalar_vec, mask):
    zero = pto.ui8(0)
    zero_r = pto.ui8(0xFF)
    full_mask_b8 = pto.pset_b8(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)
    active_low = pto.punpack(active_mask, pto.PredicatePart.LOWER)
    active_high = pto.punpack(active_mask, pto.PredicatePart.HIGHER)

    vec_low = pto.vzunpack(vec, 0)
    vec_high = pto.vzunpack(vec, 1)
    scalar_low = pto.vzunpack(scalar_vec, 0)
    scalar_high = pto.vzunpack(scalar_vec, 1)

    r_low = _tl_soft_vmod_u16(vec_low, scalar_low, active_low)
    r_high = _tl_soft_vmod_u16(vec_high, scalar_high, active_high)
    packed_low = pto.vpack(r_low, pto.PredicatePart.LOWER)
    packed_high = pto.vpack(r_high, pto.PredicatePart.HIGHER)
    r = pto.vor(packed_low, packed_high, full_mask_b8)
    return pto.vsel(pto.vbr(zero_r), r, zero_mask)


@pto.inline_proc
def _tl_soft_vmod_i8(vec, scalar_vec, mask):
    zero = pto.i8(0)
    neg_one = pto.i8(-1)
    full_mask_b8 = pto.pset_b8(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)
    active_low = pto.punpack(active_mask, pto.PredicatePart.LOWER)
    active_high = pto.punpack(active_mask, pto.PredicatePart.HIGHER)

    vec_low = pto.vsunpack(vec, 0)
    vec_high = pto.vsunpack(vec, 1)
    scalar_low = pto.vsunpack(scalar_vec, 0)
    scalar_high = pto.vsunpack(scalar_vec, 1)

    r_low = _tl_soft_vmod_i16(vec_low, scalar_low, active_low)
    r_high = _tl_soft_vmod_i16(vec_high, scalar_high, active_high)
    packed_low = pto.vpack(r_low, pto.PredicatePart.LOWER)
    packed_high = pto.vpack(r_high, pto.PredicatePart.HIGHER)
    r = pto.vbitcast(pto.vor(packed_low, packed_high, full_mask_b8), pto.i8)
    return pto.vsel(pto.vbr(neg_one), r, zero_mask)


@pto.inline_proc
def _tl_soft_vmod_u16(vec, scalar_vec, mask):
    zero = pto.ui16(0)
    one = pto.ui16(1)
    zero_r = pto.ui16(0xFFFF)
    fp32_one = pto.f32(1.0)
    full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
    full_mask_b32 = pto.pset_b32(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    zero_u16 = pto.vbr(zero)
    vy_lower_u16, vy_higher_u16 = pto.vintlv(scalar_vec, zero_u16)
    vy_lower_u32 = pto.vcvt(vy_lower_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    vy_higher_u32 = pto.vcvt(vy_higher_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    active_low = pto.vcmps(vy_lower_u32, pto.ui32(0), full_mask_b32, pto.CmpMode.NE)
    active_high = pto.vcmps(vy_higher_u32, pto.ui32(0), full_mask_b32, pto.CmpMode.NE)
    vy_lower_f32 = pto.vcvt(pto.vbitcast(vy_lower_u32, pto.i32), pto.f32, active_low, rnd=pto.VcvtRoundMode.F)
    vy_higher_f32 = pto.vcvt(pto.vbitcast(vy_higher_u32, pto.i32), pto.f32, active_high, rnd=pto.VcvtRoundMode.F)

    vy_rec_lower = pto.vdiv(pto.vbr(fp32_one), vy_lower_f32, active_low)
    vy_rec_higher = pto.vdiv(pto.vbr(fp32_one), vy_higher_f32, active_high)
    vy_scale_lower = pto.vmul(vy_rec_lower, pto.vbr(pto.f32(65536.0)), active_low)
    vy_scale_higher = pto.vmul(vy_rec_higher, pto.vbr(pto.f32(65536.0)), active_high)

    v_lower_i32 = pto.vcvt(
        vy_scale_lower,
        pto.i32,
        active_low,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
    )
    v_higher_i32 = pto.vcvt(
        vy_scale_higher,
        pto.i32,
        active_high,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
    )
    v_lower_u32 = pto.vbitcast(v_lower_i32, pto.ui32)
    v_higher_u32 = pto.vbitcast(v_higher_i32, pto.ui32)

    vx_lower_u16, vx_higher_u16 = pto.vintlv(vec, zero_u16)
    vx_lower_u32 = pto.vcvt(vx_lower_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    vx_higher_u32 = pto.vcvt(vx_higher_u16, pto.ui32, full_mask_b16, part=pto.VcvtPartMode.EVEN)
    q_tmp_lower = pto.vmul(v_lower_u32, vx_lower_u32, active_low)
    q_tmp_higher = pto.vmul(v_higher_u32, vx_higher_u32, active_high)
    _q_lower, q_tmp = pto.vdintlv(pto.vbitcast(q_tmp_lower, pto.ui16), pto.vbitcast(q_tmp_higher, pto.ui16))

    yq_tmp = pto.vmul(q_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsub(vec, yq_tmp, active_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    return pto.vsel(pto.vbr(zero_r), r_tmp, zero_mask)


@pto.inline_proc
def _tl_soft_vdiv_u32(vec, scalar_vec, mask):
    zero = pto.ui32(0)
    one = pto.ui32(1)
    zero_q = pto.ui32(0xFFFFFFFF)
    fp32_one = pto.f32(1.0)
    full_mask = pto.pset_b32(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    zero_u32 = pto.vbr(zero)
    zero_f32 = pto.vbr(pto.f32(0.0))
    vy_lower_u32, vy_higher_u32 = pto.vintlv(scalar_vec, zero_u32)
    vy_lower_f32 = pto.vcvt(pto.vbitcast(vy_lower_u32, pto.i64), pto.f32, full_mask, rnd=pto.VcvtRoundMode.F, part=pto.VcvtPartMode.EVEN)
    vy_higher_f32 = pto.vcvt(pto.vbitcast(vy_higher_u32, pto.i64), pto.f32, full_mask, rnd=pto.VcvtRoundMode.F, part=pto.VcvtPartMode.EVEN)
    vy_float, _vy_waste = pto.vdintlv(vy_lower_f32, vy_higher_f32)

    vy_rec = pto.vdiv(pto.vbr(fp32_one), vy_float, full_mask)
    vy_scale = pto.vmul(vy_rec, pto.vbr(pto.f32(4294966784.0)), full_mask)

    vy_scale_lower_f32, vy_scale_higher_f32 = pto.vintlv(vy_scale, zero_f32)
    v_lower_i64 = pto.vcvt(
        vy_scale_lower_f32,
        pto.i64,
        full_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    v_higher_i64 = pto.vcvt(
        vy_scale_higher_f32,
        pto.i64,
        full_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    z, _z_waste = pto.vdintlv(pto.vbitcast(v_lower_i64, pto.ui32), pto.vbitcast(v_higher_i64, pto.ui32))

    tmp_0 = pto.vmul(z, scalar_vec, full_mask)
    tmp_0 = pto.vbitcast(pto.vneg(pto.vbitcast(tmp_0, pto.i32), full_mask), pto.ui32)
    _z_lower, z_high = pto.vmull(z, tmp_0, full_mask)
    z = pto.vadd(z, z_high, full_mask)

    _q_lower, q_tmp = pto.vmull(vec, z, full_mask)
    yq_tmp = pto.vmul(q_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsub(vec, yq_tmp, active_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    return pto.vsel(pto.vbr(zero_q), q_tmp, zero_mask)


@pto.inline_proc
def _tl_soft_vmod_i16(vec, scalar_vec, mask):
    zero = pto.i16(0)
    neg_one = pto.i16(-1)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui16)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui16)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    p_pos = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    q_abs = _tl_soft_vdiv_u16(abs_x, abs_y, active_mask)
    neg_q = pto.vneg(pto.vbitcast(q_abs, pto.i16), active_mask)
    q = pto.vsel(pto.vbitcast(q_abs, pto.i16), neg_q, p_pos)

    qy = pto.vmul(q, scalar_vec, active_mask)
    remainder = pto.vsub(vec, qy, active_mask)

    nonzero_remainder = pto.vcmps(remainder, zero, active_mask, pto.CmpMode.NE)
    sign_x = pto.vcmps(vec, zero, active_mask, pto.CmpMode.GE)
    sign_y = pto.vcmps(scalar_vec, zero, active_mask, pto.CmpMode.GE)
    sign_diff = pto.pxor(sign_x, sign_y, active_mask)
    need_floor_fix = pto.pand(sign_diff, nonzero_remainder, active_mask)
    amended_remainder = pto.vadd(scalar_vec, remainder, active_mask)
    remainder = pto.vsel(amended_remainder, remainder, need_floor_fix)
    return pto.vsel(pto.vbr(neg_one), remainder, zero_mask)


@pto.inline_proc
def _tl_soft_vdiv_i32(vec, scalar_vec, mask):
    zero = pto.i32(0)
    neg_one = pto.i32(-1)
    fp32_one = pto.f32(1.0)
    false_mask = pto.pset_b32(pto.PAT.ALLF)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui32)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui32)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    p_pos = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    y_float = pto.vcvt(pto.vbitcast(abs_y, pto.i32), pto.f32, active_mask, rnd=pto.VcvtRoundMode.R)
    y_rec = pto.vdiv(pto.vbr(fp32_one), y_float, active_mask)
    f_z_tmp_bits = pto.vadds(pto.vbitcast(y_rec, pto.ui32), pto.ui32(0x0FFFFFFE), active_mask)

    low_mask, high_mask = pto.pintlv_b32(active_mask, false_mask)
    lower_bits, higher_bits = pto.vintlv(f_z_tmp_bits, pto.vbr(pto.ui32(0)))
    lower_i64 = pto.vcvt(
        pto.vbitcast(lower_bits, pto.f32),
        pto.i64,
        low_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    higher_i64 = pto.vcvt(
        pto.vbitcast(higher_bits, pto.f32),
        pto.i64,
        high_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    z, _z_waste = pto.vdintlv(pto.vbitcast(lower_i64, pto.ui32), pto.vbitcast(higher_i64, pto.ui32))
    active_mask, _waste_mask = pto.pdintlv_b32(low_mask, high_mask)

    fz_negative = pto.vcmps(pto.vbitcast(f_z_tmp_bits, pto.f32), pto.f32(0.0), active_mask, pto.CmpMode.LT)
    z = pto.vsel(pto.vbr(pto.ui32(0)), z, fz_negative)

    tmp_0 = pto.vmul(z, abs_y, active_mask)
    tmp_0 = pto.vbitcast(pto.vneg(pto.vbitcast(tmp_0, pto.i32), active_mask), pto.ui32)
    _z_lower, z_high = pto.vmull(z, tmp_0, active_mask)
    z = pto.vadd(z, z_high, active_mask)

    _q_lower, q_tmp = pto.vmull(abs_x, z, active_mask)
    yq_tmp = pto.vmul(q_tmp, abs_y, active_mask)
    r_tmp = pto.vsub(abs_x, yq_tmp, active_mask)
    ge_mask = pto.vcmp(r_tmp, abs_y, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, abs_y, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, pto.ui32(1), active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    ge_mask = pto.vcmp(r_tmp, abs_y, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, abs_y, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, pto.ui32(1), active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)

    neg_q = pto.vneg(pto.vbitcast(q_tmp, pto.i32), active_mask)
    q = pto.vsel(pto.vbitcast(q_tmp, pto.i32), neg_q, p_pos)
    return pto.vsel(pto.vbr(neg_one), q, zero_mask)


@pto.inline_proc
def _tl_soft_vmod_u32(vec, scalar_vec, mask):
    zero = pto.ui32(0)
    one = pto.ui32(1)
    zero_r = pto.ui32(0xFFFFFFFF)
    fp32_one = pto.f32(1.0)
    full_mask = pto.pset_b32(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    zero_u32 = pto.vbr(zero)
    zero_f32 = pto.vbr(pto.f32(0.0))
    vy_lower_u32, vy_higher_u32 = pto.vintlv(scalar_vec, zero_u32)
    vy_lower_f32 = pto.vcvt(pto.vbitcast(vy_lower_u32, pto.i64), pto.f32, full_mask, rnd=pto.VcvtRoundMode.F, part=pto.VcvtPartMode.EVEN)
    vy_higher_f32 = pto.vcvt(pto.vbitcast(vy_higher_u32, pto.i64), pto.f32, full_mask, rnd=pto.VcvtRoundMode.F, part=pto.VcvtPartMode.EVEN)
    vy_float, _vy_waste = pto.vdintlv(vy_lower_f32, vy_higher_f32)

    vy_rec = pto.vdiv(pto.vbr(fp32_one), vy_float, full_mask)
    vy_scale = pto.vmul(vy_rec, pto.vbr(pto.f32(4294966784.0)), full_mask)

    vy_scale_lower_f32, vy_scale_higher_f32 = pto.vintlv(vy_scale, zero_f32)
    v_lower_i64 = pto.vcvt(
        vy_scale_lower_f32,
        pto.i64,
        full_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    v_higher_i64 = pto.vcvt(
        vy_scale_higher_f32,
        pto.i64,
        full_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    z, _z_waste = pto.vdintlv(pto.vbitcast(v_lower_i64, pto.ui32), pto.vbitcast(v_higher_i64, pto.ui32))

    tmp_0 = pto.vmul(z, scalar_vec, full_mask)
    tmp_0 = pto.vbitcast(pto.vneg(pto.vbitcast(tmp_0, pto.i32), full_mask), pto.ui32)
    _z_lower, z_high = pto.vmull(z, tmp_0, full_mask)
    z = pto.vadd(z, z_high, full_mask)

    _q_lower, q_tmp = pto.vmull(vec, z, full_mask)
    yq_tmp = pto.vmul(q_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsub(vec, yq_tmp, active_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    ge_mask = pto.vcmp(r_tmp, scalar_vec, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, scalar_vec, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, one, active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    return pto.vsel(pto.vbr(zero_r), r_tmp, zero_mask)


@pto.inline_proc
def _tl_soft_vmod_i32(vec, scalar_vec, mask):
    zero = pto.i32(0)
    neg_one = pto.i32(-1)
    fp32_one = pto.f32(1.0)
    false_mask = pto.pset_b32(pto.PAT.ALLF)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui32)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui32)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    p_pos = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    y_float = pto.vcvt(pto.vbitcast(abs_y, pto.i32), pto.f32, active_mask, rnd=pto.VcvtRoundMode.R)
    y_rec = pto.vdiv(pto.vbr(fp32_one), y_float, active_mask)
    f_z_tmp_bits = pto.vadds(pto.vbitcast(y_rec, pto.ui32), pto.ui32(0x0FFFFFFE), active_mask)

    low_mask, high_mask = pto.pintlv_b32(active_mask, false_mask)
    lower_bits, higher_bits = pto.vintlv(f_z_tmp_bits, pto.vbr(pto.ui32(0)))
    lower_i64 = pto.vcvt(
        pto.vbitcast(lower_bits, pto.f32),
        pto.i64,
        low_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    higher_i64 = pto.vcvt(
        pto.vbitcast(higher_bits, pto.f32),
        pto.i64,
        high_mask,
        rnd=pto.VcvtRoundMode.F,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    z, _z_waste = pto.vdintlv(pto.vbitcast(lower_i64, pto.ui32), pto.vbitcast(higher_i64, pto.ui32))
    active_mask, _waste_mask = pto.pdintlv_b32(low_mask, high_mask)

    fz_negative = pto.vcmps(pto.vbitcast(f_z_tmp_bits, pto.f32), pto.f32(0.0), active_mask, pto.CmpMode.LT)
    z = pto.vsel(pto.vbr(pto.ui32(0)), z, fz_negative)

    tmp_0 = pto.vmul(z, abs_y, active_mask)
    tmp_0 = pto.vbitcast(pto.vneg(pto.vbitcast(tmp_0, pto.i32), active_mask), pto.ui32)
    _z_lower, z_high = pto.vmull(z, tmp_0, active_mask)
    z = pto.vadd(z, z_high, active_mask)

    _q_lower, q_tmp = pto.vmull(abs_x, z, active_mask)
    yq_tmp = pto.vmul(q_tmp, abs_y, active_mask)
    r_tmp = pto.vsub(abs_x, yq_tmp, active_mask)
    ge_mask = pto.vcmp(r_tmp, abs_y, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, abs_y, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, pto.ui32(1), active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)
    ge_mask = pto.vcmp(r_tmp, abs_y, active_mask, pto.CmpMode.GE)
    refined_r = pto.vsub(r_tmp, abs_y, active_mask)
    r_tmp = pto.vsel(refined_r, r_tmp, ge_mask)
    q_inc = pto.vadds(q_tmp, pto.ui32(1), active_mask)
    q_tmp = pto.vsel(q_inc, q_tmp, ge_mask)

    neg_q = pto.vneg(pto.vbitcast(q_tmp, pto.i32), active_mask)
    q = pto.vsel(pto.vbitcast(q_tmp, pto.i32), neg_q, p_pos)

    qy = pto.vmul(q, scalar_vec, active_mask)
    remainder = pto.vsub(vec, qy, active_mask)
    nonzero_remainder = pto.vcmps(pto.vbitcast(r_tmp, pto.i32), zero, active_mask, pto.CmpMode.NE)
    sign_x = pto.vcmps(vec, zero, active_mask, pto.CmpMode.GE)
    sign_y = pto.vcmps(scalar_vec, zero, active_mask, pto.CmpMode.GE)
    sign_diff = pto.pxor(sign_x, sign_y, active_mask)
    need_floor_fix = pto.pand(sign_diff, nonzero_remainder, active_mask)
    amended_remainder = pto.vadd(scalar_vec, remainder, active_mask)
    remainder = pto.vsel(amended_remainder, remainder, need_floor_fix)
    return pto.vsel(pto.vbr(neg_one), remainder, zero_mask)


@pto.inline_proc
def _tl_soft_vmod(vec, scalar_vec, mask, dtype):
    if pto.constexpr(dtype == pto.ui8):
        result = _tl_soft_vmod_u8(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.i8):
        result = _tl_soft_vmod_i8(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.ui16):
        result = _tl_soft_vmod_u16(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.i16):
        result = _tl_soft_vmod_i16(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.ui32):
        result = _tl_soft_vmod_u32(vec, scalar_vec, mask)
    else:
        result = _tl_soft_vmod_i32(vec, scalar_vec, mask)
    return result


@pto.inline_proc
def _tl_soft_vdiv(vec, scalar_vec, mask, dtype):
    if pto.constexpr(dtype == pto.ui8):
        result = _tl_soft_vdiv_u8(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.i8):
        result = _tl_soft_vdiv_i8(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.ui16):
        result = _tl_soft_vdiv_u16(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.i16):
        result = _tl_soft_vdiv_i16(vec, scalar_vec, mask)
    elif pto.constexpr(dtype == pto.ui32):
        result = _tl_soft_vdiv_u32(vec, scalar_vec, mask)
    else:
        result = _tl_soft_vdiv_i32(vec, scalar_vec, mask)
    return result
