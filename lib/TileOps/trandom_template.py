# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trandom - unified template using constexpr"""

import tilelang_dsl as pto

TRANDOM_ONCE_REPEAT = 4
TRANDOM_CONST_0 = 0xD2511F53
TRANDOM_CONST_1 = 0xCD9E8D57
TRANDOM_CONST_KEY_ADD_0 = 0x9E3779B9
TRANDOM_CONST_KEY_ADD_1 = 0xBB67AE85


def _check_row_major(dst) -> bool:
    return dst.config.b_layout == pto.BLayout.ROW_MAJOR


@pto.vkernel(
    target="a5",
    op="pto.trandom",
    dtypes=[
        (pto.i32, pto.i32, pto.i32, pto.i32, pto.i32, pto.i32, pto.ui32),
    ],
    constraints=[_check_row_major],
    advanced=True,
)
def template_trandom(
    key0: pto.i32,
    key1: pto.i32,
    counter0: pto.i32,
    counter1: pto.i32,
    counter2: pto.i32,
    counter3: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.get_lanes(dtype)
    n_loop = (valid_cols + TRANDOM_ONCE_REPEAT * lanes - 1) // (TRANDOM_ONCE_REPEAT * lanes)
    rounds_str = pto.get_op_attr("rounds", "10")

    pg = pto.pset_b32(pto.PAT.ALL)

    ctr0_init = pto.vbitcast(pto.vbr(counter0), pto.ui32)
    ctr1_init = pto.vbitcast(pto.vbr(counter1), pto.ui32)
    ctr2_init = pto.vbitcast(pto.vbr(counter2), pto.ui32)
    ctr3_init = pto.vbitcast(pto.vbr(counter3), pto.ui32)
    key0_v = pto.vbitcast(pto.vbr(key0), pto.ui32)
    key1_v = pto.vbitcast(pto.vbr(key1), pto.ui32)
    zeros = pto.vbr(pto.ui32(0))
    const0 = pto.vbr(pto.ui32(TRANDOM_CONST_0))
    const1 = pto.vbr(pto.ui32(TRANDOM_CONST_1))
    inc_idx = pto.vbitcast(pto.vci(pto.i32(0)), pto.ui32)

    ctr0, pd = pto.vaddc(ctr0_init, inc_idx, pg)
    ctr1, pd = pto.vaddcs(ctr1_init, zeros, pd, pg)
    ctr2, pd = pto.vaddcs(ctr2_init, zeros, pd, pg)
    ctr3, pd = pto.vaddcs(ctr3_init, zeros, pd, pg)

    for i in range(0, valid_rows, 1):
        s_reg = valid_cols
        counter_add_val = lanes
        for j in range(0, n_loop, 1):
            tmp_ctr0 = ctr0
            tmp_ctr1 = ctr1
            tmp_ctr2 = ctr2
            tmp_ctr3 = ctr3
            tmp_key0 = key0_v
            tmp_key1 = key1_v

            if pto.constexpr(rounds_str == "10"):
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
            elif pto.constexpr(rounds_str == "7"):
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0
                tmpL0, tmpH0 = pto.vmull(tmp_ctr0, const0, pg)
                tmpL1, tmpH1 = pto.vmull(tmp_ctr2, const1, pg)
                tmpH1 = pto.vxor(tmpH1, tmp_ctr1, pg)
                tmp_ctr0 = pto.vxor(tmpH1, tmp_key0, pg)
                tmpH0 = pto.vxor(tmpH0, tmp_ctr3, pg)
                tmp_ctr2 = pto.vxor(tmpH0, tmp_key1, pg)
                tmp_key0 = pto.vadds(tmp_key0, pto.ui32(TRANDOM_CONST_KEY_ADD_0), pg)
                tmp_key1 = pto.vadds(tmp_key1, pto.ui32(TRANDOM_CONST_KEY_ADD_1), pg)
                tmp_ctr1 = tmpL1
                tmp_ctr3 = tmpL0

            tmpL0, tmpH0 = pto.vintlv(tmp_ctr0, tmp_ctr2)
            tmpL1, tmpH1 = pto.vintlv(tmp_ctr1, tmp_ctr3)
            tmp_ctr0, tmp_ctr1 = pto.vintlv(tmpL0, tmpL1)
            tmp_ctr2, tmp_ctr3 = pto.vintlv(tmpH0, tmpH1)

            remained = s_reg
            mask0, remained = pto.make_mask(dtype, remained)
            mask1, remained = pto.make_mask(dtype, remained)
            mask2, remained = pto.make_mask(dtype, remained)
            mask3, remained = pto.make_mask(dtype, remained)

            pto.vsts(tmp_ctr0, dst[i, TRANDOM_ONCE_REPEAT * j * lanes:], mask0)
            pto.vsts(tmp_ctr1, dst[i, (TRANDOM_ONCE_REPEAT * j + 1) * lanes:], mask1)
            pto.vsts(tmp_ctr2, dst[i, (TRANDOM_ONCE_REPEAT * j + 2) * lanes:], mask2)
            pto.vsts(tmp_ctr3, dst[i, (TRANDOM_ONCE_REPEAT * j + 3) * lanes:], mask3)

            if s_reg >= TRANDOM_ONCE_REPEAT * lanes:
                s_reg = s_reg - TRANDOM_ONCE_REPEAT * lanes
            else:
                s_reg = 0

            if j != n_loop - 1:
                counter_add_val = lanes
            else:
                counter_add_val = (valid_cols - 1) % lanes + 1
            v_ele_stride = pto.vbr(pto.ui32(counter_add_val))
            ctr0_next, pd_next = pto.vaddc(ctr0, v_ele_stride, pg)
            ctr1_next, pd_next2 = pto.vaddcs(ctr1, zeros, pd_next, pg)
            ctr2_next, pd_next3 = pto.vaddcs(ctr2, zeros, pd_next2, pg)
            ctr3_next, pd_next4 = pto.vaddcs(ctr3, zeros, pd_next3, pg)
            ctr0 = ctr0_next
            ctr1 = ctr1_next
            ctr2 = ctr2_next
            ctr3 = ctr3_next

    return
