# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmrgsort"""

import tilelang_dsl as pto

STRUCT_SIZE = 8  # bytes per structure (value + index)
STRUCT_SIZE_SHIFT = 3  # log2(8)
BLOCK_NUM = 4


@pto.inline_proc
def tmrgsort_single_list_instr(dst: pto.Tile, src: pto.Tile,
                         num_structures, repeat_times):
    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()

    count = pto.i64(num_structures)
    count = count | (pto.i64(num_structures) << pto.i64(16))
    count = count | (pto.i64(num_structures) << pto.i64(32))
    count = count | (pto.i64(num_structures) << pto.i64(48))

    offset = num_structures * STRUCT_SIZE // pto.bytewidth(dst.element_type)
    src0 = src_ptr
    src1 = pto.addptr(src_ptr, offset)
    src2 = pto.addptr(src_ptr, offset * 2)
    src3 = pto.addptr(src_ptr, offset * 3)

    config = pto.i64(repeat_times)
    config = config | (pto.i64(0b1111) << pto.i64(8))
    config = config | (pto.i64(0b0) << pto.i64(12))

    pto.vmrgsort4(dst_ptr, src0, src1, src2, src3, pto.i64(count), pto.i64(config))
    return


@pto.inline_proc
def tmrgsort_multi_list2_instr(tmp: pto.Tile, src0: pto.Tile, src1: pto.Tile,
                                    src0_structures: int, src1_structures: int):
    tmp_ptr = tmp.as_ptr()
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    
    count = pto.i64(src0_structures)
    count = count | (pto.i64(src1_structures) << pto.i64(16))
    
    repeat_time = 1
    list_mask = 0b0011
    exhausted_bit = 0
    
    exhausted_str = pto.get_op_attr("exhausted", "0")
    if pto.constexpr(exhausted_str == "1"):
        exhausted_bit = 1

    config = pto.i64(repeat_time)
    config = config | (pto.i64(list_mask) << pto.i64(8))
    config = config | (pto.i64(exhausted_bit) << pto.i64(12))
    
    pto.vmrgsort4(tmp_ptr, src0_ptr, src1_ptr, src0_ptr, src0_ptr,
                   count, config)
    
    return


@pto.inline_proc
def tmrgsort_multi_list3_instr(tmp: pto.Tile, src0: pto.Tile, src1: pto.Tile, src2: pto.Tile,
                                   src0_structures: int, src1_structures: int, src2_structures: int):
    tmp_ptr = tmp.as_ptr()
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    src2_ptr = src2.as_ptr()
    
    count = pto.i64(src0_structures)
    count = count | (pto.i64(src1_structures) << pto.i64(16))
    count = count | (pto.i64(src2_structures) << pto.i64(32))
    
    repeat_time = 1
    list_mask = 0b0111
    exhausted_bit = 0
    
    exhausted_str = pto.get_op_attr("exhausted", "0")
    if pto.constexpr(exhausted_str == "1"):
        exhausted_bit = 1

    config = pto.i64(repeat_time)
    config = config | (pto.i64(list_mask) << pto.i64(8))
    config = config | (pto.i64(exhausted_bit) << pto.i64(12))
    
    pto.vmrgsort4(tmp_ptr, src0_ptr, src1_ptr, src2_ptr, src0_ptr,
                   count, config)
    
    return


@pto.inline_proc
def tmrgsort_multi_list4_instr(tmp: pto.Tile, src0: pto.Tile, src1: pto.Tile, 
                                 src2: pto.Tile, src3: pto.Tile,
                                 src0_structures: int, src1_structures: int, 
                                 src2_structures: int, src3_structures: int):
    dtype = tmp.element_type
    
    tmp_ptr = tmp.as_ptr()
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    src2_ptr = src2.as_ptr()
    src3_ptr = src3.as_ptr()
    
    count = pto.i64(src0_structures)
    count = count | (pto.i64(src1_structures) << pto.i64(16))
    count = count | (pto.i64(src2_structures) << pto.i64(32))
    count = count | (pto.i64(src3_structures) << pto.i64(48))
    
    repeat_time = 1
    list_mask = 0b1111
    exhausted_bit = 0

    exhausted_str = pto.get_op_attr("exhausted", "0")
    if pto.constexpr(exhausted_str == "1"):
        exhausted_bit = 1

    config = pto.i64(repeat_time)
    config = config | (pto.i64(list_mask) << pto.i64(8))
    config = config | (pto.i64(exhausted_bit) << pto.i64(12))
    
    pto.vmrgsort4(tmp_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,
                   count, config)
    
    return


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort",
    advanced=True,
)
def template_tmrgsort_single_list(src: pto.Tile, block_len: pto.AnyInt, dst: pto.Tile):
    """Format1 template: single list internal block sorting.

    Standard Format1: single vmrgsort4 for block sorting.
    TopK variant is handled by ST kernel via iterative tmrgsort + tmov calls.
    """
    src_valid_col = src.valid_shape[1]

    # Block length in structures
    num_structures = block_len * pto.bytewidth(src.element_type) >> STRUCT_SIZE_SHIFT

    # Repeat times: how many groups of 4 blocks need merging
    repeat_times = src_valid_col // (block_len * BLOCK_NUM)

    # Standard Format1: single merge operation
    tmrgsort_single_list_instr(dst, src, num_structures, repeat_times)

    return None


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort",
    advanced=True,
)
def template_tmrgsort_multi_list2(src0: pto.Tile, src1: pto.Tile, 
                            tmp: pto.Tile, dst: pto.Tile, ex_vec: pto.AnyInt):
    dtype = dst.element_type
    bw = pto.bytewidth(dtype)
    
    src0_valid_col = src0.valid_shape[1]
    src1_valid_col = src1.valid_shape[1]
    dst_valid_col = dst.valid_shape[1]
    
    if pto.constexpr(bw == 4):
        src0_structures = src0_valid_col // 2
        src1_structures = src1_valid_col // 2
    else:
        src0_structures = src0_valid_col // 4
        src1_structures = src1_valid_col // 4
    
    dst_elements = dst_valid_col
    
    tmrgsort_multi_list2_instr(tmp, src0, src1, src0_structures, src1_structures)
    
    lanes = pto.get_lanes(dtype)
    for col in range(0, dst_elements, lanes):
        remained = dst_elements - col
        mask, remained = pto.make_mask(dtype, remained)
        data = pto.vlds(tmp[0, col:])
        pto.vsts(data, dst[0, col:], mask)
    
    return None


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort",
    advanced=True,
)
def template_tmrgsort_multi_list3(src0: pto.Tile, src1: pto.Tile, src2: pto.Tile,
                            tmp: pto.Tile, dst: pto.Tile, ex_vec: pto.AnyInt):
    dtype = dst.element_type
    bw = pto.bytewidth(dtype)
    
    src0_valid_col = src0.valid_shape[1]
    src1_valid_col = src1.valid_shape[1]
    src2_valid_col = src2.valid_shape[1]
    dst_valid_col = dst.valid_shape[1]
    
    if pto.constexpr(bw == 4):
        src0_structures = src0_valid_col // 2
        src1_structures = src1_valid_col // 2
        src2_structures = src2_valid_col // 2
    else:
        src0_structures = src0_valid_col // 4
        src1_structures = src1_valid_col // 4
        src2_structures = src2_valid_col // 4
    
    dst_elements = dst_valid_col
    
    tmrgsort_multi_list3_instr(tmp, src0, src1, src2, src0_structures, src1_structures, src2_structures)
    
    lanes = pto.get_lanes(dtype)
    for col in range(0, dst_elements, lanes):
        remained = dst_elements - col
        mask, remained = pto.make_mask(dtype, remained)
        data = pto.vlds(tmp[0, col:])
        pto.vsts(data, dst[0, col:], mask)
    
    return None


@pto.vkernel(
    target="a5",
    op="pto.tmrgsort",
    advanced=True,
)
def template_tmrgsort_multi_list4(src0: pto.Tile, src1: pto.Tile, src2: pto.Tile, src3: pto.Tile,
                             tmp: pto.Tile, dst: pto.Tile, ex_vec: pto.AnyInt):
    dtype = dst.element_type
    bw = pto.bytewidth(dtype)
    
    src0_valid_col = src0.valid_shape[1]
    src1_valid_col = src1.valid_shape[1]
    src2_valid_col = src2.valid_shape[1]
    src3_valid_col = src3.valid_shape[1]
    dst_valid_col = dst.valid_shape[1]
    
    if pto.constexpr(bw == 4):
        src0_structures = src0_valid_col // 2
        src1_structures = src1_valid_col // 2
        src2_structures = src2_valid_col // 2
        src3_structures = src3_valid_col // 2
    else:
        src0_structures = src0_valid_col // 4
        src1_structures = src1_valid_col // 4
        src2_structures = src2_valid_col // 4
        src3_structures = src3_valid_col // 4
    
    dst_elements = dst_valid_col
    
    tmrgsort_multi_list4_instr(tmp, src0, src1, src2, src3, src0_structures, src1_structures, src2_structures, src3_structures)
    
    lanes = pto.get_lanes(dtype)
    for col in range(0, dst_elements, lanes):
        remained = dst_elements - col
        mask, remained = pto.make_mask(dtype, remained)
        data = pto.vlds(tmp[0, col:])
        pto.vsts(data, dst[0, col:], mask)
    
    return None