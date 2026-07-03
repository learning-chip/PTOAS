# Block MX Quant on PTO: VMI vs MI vs CCE

This walkthrough explains the block MX quantization examples in this directory.
It is written for algorithm engineers who want the scale/quant math to stay
visible, while still understanding the register-layout work that MI and CCE must
spell explicitly.

| Layer | Example files | Use it for |
|-------|---------------|------------|
| **CCE** | `bmx_cce_kernels.h` | Intrinsics ground truth and expected UB behavior |
| **MI** (`pto.mi`) | `mx_block_quant_scale_ocp_bf16.mi.pto`, `mx_block_quant_y1_fp8_f16_e4m3.mi.pto` | Hardware-faithful PTO micro-ops |
| **VMI** (`pto.vmi`) | `mx_block_quant_scale_ocp_bf16.vmi.pto`, `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto` | Logical-vector authoring |

**How to use this note:** read it as a bridge between the MX scale/quant formulas
and the actual VMI/MI/CCE code. The goal is to make it clear what each VMI
instruction means, what MI/CCE pattern it usually lowers to, and which lines are
algorithmic math versus register-layout repair.

**Concrete VF tile in the simulator examples:** correctness tests use
`rowNum=32`, `colBlockSize=256`, `ubBlockSize=32`, and `vlForHalfNumber=128`.
Each input row is `256` bf16/f16 values (`512` bytes). There are `8` scale groups
per row (`256/32`), `scale1` is `32*32=1024` bytes, `scale2` is `256` bytes,
and reciprocal scale is `16` uint16 lanes (`32` bytes). Wall-time configs also
list `(rowNum,colBlockSize)=(4,64),(16,128),(32,256)`, but the local vector
pipeline is written around a 256-lane row.

VMI keeps layout bookkeeping out of the source you write. It does not remove
row/block loops, UB stride math, DMA, pipeline flags, or the need to inspect
lowered MI for debugging and tuning.

---

## 1. MX Quant Math

The examples cover two vector compute stages.

### Step 1: OCP Shared Scale

For each 32-column group, compute a shared exponent across the rows in the tile:

```text
max_exp = max(biased_exponent(x_i))      # across rows and lanes in the group
scale1  = encode_E8M0(max(max_exp, yMaxExp) - yMaxExp)
recip   = 0x7F00 - (max(max_exp, yMaxExp) - yMaxExp)
```

Special cases are explicit in all layers:

- Inf/NaN block: `scale1 = 0xFF`, reciprocal uses the custom BF16 NaN pattern
- all-zero block: `scale1 = 0`, reciprocal `= 0`
- exponent below target range: clamp by `yMaxExp`
- invalid reciprocal edge: substitute `SPECIAL_EXP_THRESHOLD`

The bf16 scale demo extracts exponent bits with `x & 0x7F80`, reduces across
rows and 32-column groups, then writes `scale1`, `scale2`, and reciprocal scale.

### Step 2: f16 to fp8 E4M3

For each 256-element row:

```text
y_i = fp8_e4m3( f32(x_i) * f32(reciprocal_scale_for_group(i)) )
```

The f16 demo widens to fp32, multiplies by a broadcast reciprocal scale, narrows
to FP8 E4M3 with round-nearest-even and saturation, then stores packed fp8
bytes. This matches the FP16 branch of `ComputeY1ToFP8` in CCE.

---

## 2. What VMI Changes

VMI lets the author write logical vectors such as `!pto.vmi.vreg<256xf16>` and
semantic ops such as `pto.vmi.vload`, `pto.vmi.vcvt`, `pto.vmi.vmul`,
`pto.vmi.vcmax`, `pto.vmi.vbrc`, and `pto.vmi.vstore`.

Lowering still emits MI instructions with concrete:

- split-load distributions such as `DINTLV_B16`
- part selections such as `PART_EVEN`, `PART_ODD`, and `PART_P0`
- layout repairs such as `vintlv`
- packing stores such as `PK4_B32`
- mask families such as `b16`, `b32`, and `b8`

The VMI conversion ops mirror MLIR `arith` naming:

- `vcvt`: widen, narrow, or reinterpret lane element types at the semantic level

These names describe lane-wise meaning. `pto.vmi.vcvt` over `vreg<256xf16>` is not
one hardware instruction; the MI lowering has to split a 512-byte row, widen
four physical streams, and repair layout.

---

## 3. How to Read the Examples

Tag instructions by role:

| Role | Question | MX examples |
|------|----------|-------------|
| **MATH** | Does it change the numeric value? | max/select, scale encode, `vmul` |
| **TYPE** | Does it change dtype while preserving lane `i`? | `vcvt`, `vpack` |
| **LAYOUT** | Does it only move lanes or bytes? | `DINTLV_B16`, `PART_EVEN/ODD`, `vintlv`, `PK4_B32`, `vselr` |
| **MEMORY** | Does it cross the UB/register boundary? | `vload`, `vstore`, `vlds`, `vsts` |

VMI keeps **MATH** and **TYPE** in the source. MI/CCE expose **LAYOUT** and
**MEMORY** because the author is programming physical registers.

### Instruction Map

| Math intent | CCE / MI shape | VMI shape |
|-------------|----------------|-----------|
| Load 256 b16 values | `pto.mi.vldsx2 DINTLV_B16` | `pto.vmi.vload` |
| Extract bf16 exponent bits | `pto.mi.vand` + b16 mask | `pto.vmi.vand` |
| Running max | `vmax` over two split accumulators | `vcmp` + `vsel` over one logical vector |
| Group max | `pto.mi.vcgmax` | `pto.vmi.vcmax` + `pto.vmi.vbrc` |
| Scale byte narrowing | `pto.mi.vpack LOWER` | `pto.vmi.vcvt` |
| Load reciprocal scale | `pto.mi.vlds E2B_B16` + bitcast + `pto.mi.vcvt EVEN` | `pto.vmi.vload` + `pto.vmi.vcvt` |
| Widen f16 to fp32 | 4x `pto.mi.vcvt EVEN/ODD` | `pto.vmi.vcvt` |
| Repair widened layout | 4x `vintlv` | compiler lowering |
| fp32 multiply | 4x `pto.mi.vmul` + b32 mask | `pto.vmi.vmul` |
| fp32 to fp8 | 4x `pto.mi.vcvt {P0,rnd=R,sat=SAT}` | `pto.vmi.vcvt` |
| Store fp8 row | 4x `pto.mi.vsts PK4_B32` | `pto.vmi.vstore` |

### Expected Lowering Shapes

**256-lane f16 load + widen:**

```mlir
%x_f16 = pto.vmi.vload %x_ub[%row_off] : ... -> !pto.vmi.vreg<256xf16>
%x_f32 = pto.vmi.vcvt %x_f16 : ... -> !pto.vmi.vreg<256xf32>
```

Expected MI/CCE shape:

```mlir
%x0, %x1 = pto.mi.vldsx2 %xHalf[%row_off], "DINTLV_B16" : ...
%x0_even = pto.mi.vcvt %x0, %mask16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
%x0_odd  = pto.mi.vcvt %x0, %mask16 {part = "ODD"}  : ... -> !pto.mi.vreg<64xf32>
// same for x1, followed by vintlv repair before fp8 conversion
```

`vcvt` means “produce f32 lane `i` for every logical `x[i]`.” MI has to expose
the physical split and repair.

**fp32 to fp8 narrow + store:**

```mlir
%y_fp8 = pto.vmi.vcvt %scaled : ... -> !pto.vmi.vreg<256xf8E4M3FN>
pto.vmi.vstore %y_fp8, %y_ub[%row_off], %mask : ...
```

Expected MI/CCE shape:

```mlir
%chunk_fp8 = pto.mi.vcvt %chunk_f32, %mask32 {part = "P0", rnd = "R", sat = "SAT"} : ...
%chunk_u8 = pto.mi.vbitcast %chunk_fp8 : ...
pto.mi.vsts %chunk_u8, %y_ub[%off], %mask8 {dist = "PK4_B32"} : ...
// repeated at byte offsets 0, 64, 128, 192
```

`vcvt` preserves logical lane order and FP8 semantics. `PART_P0` and
`PK4_B32` are the physical byte-placement protocol.

**Scale-byte narrowing:**

```mlir
%scale_u8 = pto.vmi.vcvt %scale_u16 : ... -> !pto.vmi.vreg<256xui8>
```

Expected MI/CCE shape:

```mlir
%scale_u8 = pto.mi.vpack %scale_u16, "LOWER" : !pto.mi.vreg<128xi16> -> !pto.mi.vreg<256xui8>
```

Only the first eight scale bytes are meaningful in the default 256-column row;
the rest are padding/layout material.

---

## 4. Running Example: f16 to fp8 E4M3

This is the clearest VMI win. The algorithm is one multiply and one cast; MI and
CCE spend most of the loop repairing physical layout.

VMI row body:

```mlir
%x_f16 = pto.vmi.vload %x_ub[%row_off] : ... -> !pto.vmi.vreg<256xf16>
%x_f32 = pto.vmi.vcvt %x_f16 : ... -> !pto.vmi.vreg<256xf32>
%scaled = pto.vmi.vmul %x_f32, %scale_f32 : ...
%y_fp8 = pto.vmi.vcvt %scaled : ... -> !pto.vmi.vreg<256xf8E4M3FN>
pto.vmi.vstore %y_fp8, %y_ub[%row_off], %mask : ...
```

The same row in MI/CCE needs:

1. `DINTLV_B16` load: 256 f16 values become two 128-lane registers.
2. Four `vcvt {part=EVEN/ODD}`: each 128-lane f16 half becomes two 64-lane f32
   streams.
3. Four `vmul` operations: scale each f32 stream.
4. Four `vintlv` operations: rebuild contiguous logical order.
5. Four `vcvt {part=P0,rnd=R,sat=SAT}` operations: narrow f32 chunks to FP8.
6. Four `vbitcast` + `vsts {dist=PK4_B32}` pairs: pack P0 bytes into UB.

Physical index flow:

```text
UB f16 [0..255]
  -> DINTLV_B16:
       x0F16 = [0,2,4,...,254]
       x1F16 = [1,3,5,...,255]
  -> vcvt EVEN/ODD:
       [0,4,8,...], [2,6,10,...], [1,5,9,...], [3,7,11,...]
  -> vmul:
       same index ownership, scaled
  -> vintlv repair:
       contiguous f32 chunks [0..127] and [128..255]
  -> vcvt P0 + PK4_B32:
       dense fp8 bytes [0..255]
```

VMI hides the ownership bookkeeping. It does not make the layout work disappear;
it moves the work into lowering.

---

## 5. Running Example: OCP Scale

The scale path is less visually dramatic than the quant path, but VMI still
removes important split-register ceremony.

VMI shape:

```mlir
%x_bits = pto.vmi.vload %xAddr[%load_off] : ... -> !pto.vmi.vreg<256xui16>
%x_exp = pto.vmi.vand %x_bits, %expMaskBF16 : ...
%is_larger = pto.vmi.vcmp "slt", %acc, %x_exp : ...
%acc_next = pto.vmi.vsel %is_larger, %x_exp, %acc : ...
```

MI/CCE shape:

```mlir
%x0, %x1 = pto.mi.vldsx2 %xBf[%load_off], "DINTLV_B16" : ...
%x0_exp = pto.mi.vand %x0_bits, %expMaskBF16, %pregAllB16 : ...
%x1_exp = pto.mi.vand %x1_bits, %expMaskBF16, %pregAllB16 : ...
%acc0 = pto.mi.vmax %acc0, %x0_exp, %pregAllB16 : ...
%acc1 = pto.mi.vmax %acc1, %x1_exp, %pregAllB16 : ...
```

The math is “extract exponent and take a running max.” MI/CCE must track two
accumulators because `DINTLV_B16` split the row into even and odd logical
indices.

After row accumulation:

- MI/CCE use `pto.mi.vcgmax` to reduce within grouped lanes and broadcast results.
- VMI names this as `vcmax` plus `vbrc`.
- MI/CCE use `pto.mi.vpack LOWER` for compact E8M0 bytes.
- VMI names that semantic byte narrowing as `trunci`.

The scale path does not need the four-stream f32 `vintlv` repair from the quant
path, because it stays in b16 integer/exponent fields.

---

## 6. Debug Appendix: Physical Rules

Use this section when inspecting lowered MI or comparing against the CCE comments
in `bmx_cce_kernels.h`.

### 256-Byte Register Model

| Quantity | Bytes | Physical consequence |
|----------|-------|----------------------|
| 128 x f16/bf16/ui16 | 256 B | fits one b16 register |
| 256 x f16/bf16/ui16 | 512 B | needs split load or two registers |
| 64 x f32 | 256 B | fits one b32 register |
| 256 x f32 | 1024 B | four f32 registers at peak |
| 256 x fp8 bytes | 256 B | fits after P0 placement and PK4 packing |

### DINTLV_B16

`DINTLV_B16` loads one 256-element b16 row into two registers:

```text
x0: [x0, x2, x4, ..., x254]
x1: [x1, x3, x5, ..., x255]
```

Everything downstream must remember which register owns which logical indices.
VMI records that ownership in layout metadata instead of source variable names.

### PART_EVEN / PART_ODD

Widening 128 f16 values to f32 doubles byte size, so each source register splits
again:

```text
x0F16 logical indices: [0,2,4,6,...]
vcvt EVEN -> [0,4,8,...]
vcvt ODD  -> [2,6,10,...]
```

The same happens to `x1F16`, yielding four f32 streams at stride 4.

### vintlv Repair

`vintlv` merges lane streams to restore contiguous logical chunks. The f16→fp8
path uses two intra-half interleaves and two cross-half interleaves before FP8
conversion. These operations do not change values; they only repair lane order.

### PART_P0 and PK4_B32

FP8 conversion writes the 8-bit result into the P0 byte of each 32-bit lane:

```text
f32 lane -> [fp8_byte, 0, 0, 0]
```

`PK4_B32` stores extract those P0 bytes densely. The quant row uses four
64-byte stores at offsets `0`, `64`, `128`, and `192`.

### E2B_B16 Scale Load

The reciprocal scale is stored as bf16 exponent fields. MI loads it with
`E2B_B16`, bitcasts to bf16, and widens with `PART_EVEN`. VMI expresses this as
logical `vload` plus `vcvt`; lowering chooses the broadcast distribution.

### Common Bugs VMI Helps Avoid

- forgetting one `vintlv` repair and producing a permuted but byte-valid row
- using the wrong `part` for FP8 conversion
- using the wrong `PK4_B32` store offset
- mixing `b16`, `b32`, and `b8` masks in the quant loop
- treating `vcgmax` as an ordinary max instead of a grouped reduction/broadcast

---

## 7. Practical Guidance

| Task | Prefer |
|------|--------|
| Author or review scale and quant math | **VMI** |
| Verify bit-exact behavior against intrinsics | **CCE** or lowered **MI** |
| Debug lane permutations or packed stores | Lowered **MI** plus CCE comments |
| Learn DINTLV/part/PK rules | MI examples and Pack/Unpack reference |

Review checklist for VMI MX quant:

1. Logical vector width matches the row shape (`256` lanes).
2. Scale math still spells Inf/NaN, zero, clamp, and reciprocal edge cases.
3. `vcvt` appears before fp32 multiply and again for FP8/FP4 output and compact
   scale bytes.
4. Lowered MI for f16→fp8 has the expected `DINTLV_B16`, `PART_EVEN/ODD`,
   `vintlv`, `PART_P0`, and four `PK4_B32` stores.
5. For CCE comparison, start from the `Expected UB effect` block and then read
   the register allocation and phase comments in `ComputeOcp`, `ComputeDdr`,
   `ComputeY1ToFP4`, or `ComputeY1ToFP8`.

Suggested reading order:

1. Section 1 for math.
2. `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto`.
3. Section 4 here for the quant lowering shape.
4. `mx_block_quant_y1_fp8_f16_e4m3.mi.pto`.
5. `mx_block_quant_scale_ocp_bf16.vmi.pto` and `.mi.pto`.
6. `bmx_cce_kernels.h` `ComputeOcp` and `ComputeY1ToFP8`.
7. Section 6 here when debugging physical lanes.
8. `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`.

---

## Example File Index

| File | Layer | Stage | Notes |
|------|-------|-------|-------|
| `mx_block_quant_scale_ocp_bf16.vmi.pto` | VMI | Scale | Logical 256-lane exponent/reduction flow |
| `mx_block_quant_scale_ocp_bf16.mi.pto` | MI | Scale | DINTLV split, dual accumulators, `vcgmax`, `vpack` |
| `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto` | VMI | Quant | Best MX VMI demo |
| `mx_block_quant_y1_fp8_f16_e4m3.mi.pto` | MI | Quant | CCE-faithful split/part/vintlv/PK path |
| `bmx_cce_kernels.h` | CCE | Both | Intrinsics ground truth |

Related docs:

- `ROPE_VMI_vs_MI.md`
- `PTO-Gym-vmi/docs/PTO-vmi-design.en.md`
- `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
- `PTO-Gym-vmi/docs/PTO-micro-Instruction-SPEC.md`
