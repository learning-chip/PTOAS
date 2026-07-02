# Block MX Quant on PTO: Why VMI Helps Algorithm Engineers

This note compares the block MX quantization example kernels in this directory
across three expression layers. The examples cover the two main compute stages
of OCP Microscaling (MX) block quantization:

| Layer | Example files | Who it is for |
|-------|---------------|---------------|
| **CCE** | `bmx_cce_kernels.h` | Ground-truth intrinsics execution |
| **MI** (`pto.mi`) | `mx_block_quant_scale_ocp_bf16.mi.pto`, `mx_block_quant_y1_fp8_f16_e4m3.mi.pto` | Hardware-faithful PTO micro-ops |
| **VMI** (`pto.vmi`) | `mx_block_quant_scale_ocp_bf16.vmi.pto`, `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto` | Logical-vector authoring |

**Audience:** algorithm engineers implementing or reviewing MX block quant who
want scale math and quant math visible in code, without hand-managing DINTLV
splits, part selection, mask families, or store packing modes.

**Scope:** two paired demos on a fixed tile shape — **256 half-precision
elements per row**, **32-element column blocks**, OCP scale path on **bf16**
input, and **f16 → fp8 E4M3** quant execution using a precomputed reciprocal
scale. DMA shells differ between the full-kernel VMI quant example and the
compute-only MI/CCE excerpts; this doc focuses on the **vector compute bodies**.

This document is intentionally honest: VMI is a large win on the quant path and
a solid win on scale computation, but it does not remove tiling, stride math, or
pipeline synchronization.

---

## 1. Block MX quant math (what both stages implement)

OCP MX block quantization processes input `X` in **32×32 element super-blocks**
(32 columns × 32 rows). Each super-block shares one **scale1** (E8M0, `uint8`)
and produces a **reciprocal scale** used to normalize values before casting to
FP8/FP4.

### 1.1 Step 1 — shared scale (`ComputeOcp`)

For each 32-element column group inside the super-block:

```
max_exp = max( biased_exponent(x_i) )     // over all rows and lanes in the block
scale1  = encode_E8M0( clamp(max_exp - emax_target) )
recip   = bias - max_exp                  // bf16 exponent field for dequant in step 2
```

Special cases handled in all three layers:

- **Inf/NaN block** → scale1 = `0xFF`, reciprocal → custom NaN pattern
- **All-zero block** → scale1 = 0, reciprocal = 0
- **Exponent below representable range** → clamp to `yMaxExp` before encoding
- **Subnormal reciprocal edge** → substitute a special exponent threshold

The bf16 demo (`ComputeOcp_bf16`) extracts exponents with `x & 0x7F80`, reduces
across rows, then writes **scale1**, **scale2** (interleaved layout), and
**reciprocal** to UB.

### 1.2 Step 2 — quantize to FP8 (`ComputeY1ToFP8`, f16 path)

For each row of 256 f16 elements:

```
y_i = quant_fp8( widen_f32(x_i) * widen_f32(reciprocal_scale) )
```

The f16 demo uses **fp32 intermediate arithmetic** before narrowing to **fp8
E4M3** with round-nearest-even and saturation — matching the CCE reference
(`ComputeY1ToFP8`, FP16 branch).

---

## 2. What VMI actually is

VMI (`pto.vmi`) sits between your algorithm and MI (`pto.mi`). You write
**logical contiguous vectors** (`!pto.vmi.vreg<N×T>`) and **semantic ops**
(`vmi.load`, `vmi.extf`, `vmi.mulf`, `vmi.truncf`, `group_reduce_maxi`, …).
The compiler lowers to MI with the correct `dist`, `part`, `PK/UNPK`, and mask
granularity.

See `PTO-Gym-vmi/docs/PTO-vmi-design.en.md` and
`PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
for the underlying layout rules VMI hides.

**VMI removes:** DINTLV even/odd register splitting, dual accumulators caused
by that split, per-op mask plumbing, explicit `part`/`rnd`/`sat` on every
convert, and multi-store PK packing choreography.

**VMI does not remove:** row/block loop nesting, UB stride math, gather-index
setup at the API boundary, DMA, or pipeline flags.

---

## 3. Instruction map: math → CCE → MI → VMI

| Math intent | CCE intrinsic | MI (`pto.mi`) | VMI (`pto.vmi`) |
|-------------|---------------|---------------|-----------------|
| Load 256 half-precision elements | `vlds` (`DINTLV_B16`) | `pto.vldsx2` + `"DINTLV_B16"` | `pto.vmi.load` → `vreg<256×T>` |
| Broadcast scalar constants | `vbr` | `pto.vbr` → `vreg<128×T>` | `pto.vmi.broadcast` → `vreg<256×T>` |
| Extract bf16 exponent field | `vand` (`MODE_ZEROING` + mask) | `pto.vand` + `!pto.mask<b16>` | `pto.vmi.andi` (no mask) |
| Running max over exponents | `vmax` (`MODE_ZEROING` + mask) | `pto.vmax` + mask; **two accumulators** for even/odd halves | `pto.vmi.cmpi` + `pto.vmi.select` (single accumulator) |
| Cross-lane block max | `vcgmax` (`MODE_ZEROING` + mask) | `pto.vcgmax` (reduce + broadcast fused) | `pto.vmi.group_reduce_maxi` + `pto.vmi.group_broadcast` |
| Compare + conditional replace | `vcmp` + `vsel` | `pto.vcmp` + `pto.vsel` + mask | `pto.vmi.cmpi` + `pto.vmi.select` |
| Shift exponent diff → E8M0 | `vshrs` (`MODE_ZEROING` + mask) | `pto.vshrs` + mask | `pto.vmi.shrui` |
| Pack scale bytes | `vpack` (`LOWER`) | `pto.vpack` `"LOWER"` | `pto.vmi.trunci` |
| Gather scale2 layout | `vlds` (`NORM`) + `vselr` | same | direct `pto.vmi.store` when lanes are uniform after broadcast |
| Store scale / reciprocal | `vsts` (`NORM_B8` / `NORM_B16` + mask) | `pto.vsts` + `{dist}` + mask | `pto.vmi.store` / `pto.vmi.masked_store` |
| Load reciprocal scale | `vlds` (`E2B_B16`) | `pto.vlds` `{dist = "E2B_B16"}` + `vbitcast` + `vcvt {part=EVEN}` | `pto.vmi.load` + `pto.vmi.extf` |
| Widen f16/bf16 → f32 | `vcvt` (`PART_EVEN` / `PART_ODD`, `MODE_ZEROING`) | 4× `pto.vcvt` after DINTLV split | `pto.vmi.extf` |
| Scale multiply in fp32 | `vmul` (`MODE_ZEROING` + mask) | 4× `pto.vmul` + `!pto.mask<b32>` | `pto.vmi.mulf` |
| Re-order after widen | `vintlv` | 4× `pto.vintlv` to rebuild P0–P3 layout | *(compiler)* |
| Narrow f32 → fp8 E4M3 | `vcvt` (`ROUND_R`, `RS_ENABLE`, `PART_P0`, `MODE_ZEROING`) | 4× `pto.vcvt` + `vbitcast` + 4× `pto.vsts` `{dist=PK4_B32}` | `pto.vmi.truncf` + `pto.vmi.masked_store` |
| Tail / active lanes | `pset` / `plt_b16` / `plt_b32` / `plt_b8` | `pto.pset_b16` / `pset_b32` / `pset_b8` / `pge_*` | `pto.vmi.create_mask` |

### 3.1 MI details algorithm engineers should not have to memorize

These are the hardware subtleties VMI is designed to hide in MX quant:

**DINTLV_B16 load:** 256 half-precision elements in UB become **two 128-lane
physical registers** — even indices in one register, odd indices in the other.
Every subsequent op must reason about which half owns which logical index.

**Nested part selection:** after DINTLV, widening f16→f32 splits **each** half
again into `PART_EVEN` and `PART_ODD`, yielding **four** fp32 streams. The MI
quant loop then uses multiple `vintlv` steps to rebuild an order compatible
with `PART_P0` FP8 packing.

**E2B_B16 scale load:** reciprocal scales are loaded with a broadcast
distribution; valid bf16 lanes sit in a specific layout. MI must `vbitcast` and
`vcvt {part=EVEN}` before use.

**PK4_B32 store:** FP8 output requires four packed stores with 64-byte offsets.
Each store needs `vbitcast` to `ui8` and a `b32`/`b8` mask pairing.

**Mask family mismatch:** scale computation mixes `!pto.mask<b16>` (compute) with
`!pto.mask<b8>` (scale1 store) and `!pto.mask<b16>` (reciprocal store). Quant
mixes `b16`, `b32`, and `b8` in one loop body.

VMI replaces these with “load 256 f16”, “extend to fp32”, “multiply”, “truncate
to fp8”, and “store”.

---

## 4. Advantage by stage (ranked)

Honest ranking of **VMI syntax / intuitiveness / math closeness** vs MI:

| Rank | Stage | VMI advantage | Why |
|------|-------|---------------|-----|
| 1 | **f16 → fp8 E4M3 quant** | **Very large** | MI loop body is ~17 ops; most are layout repair (`DINTLV`, `EVEN/ODD`, `vintlv`, `PK4_B32`). VMI loop body is 5 semantic ops |
| 2 | **OCP scale (bf16)** | **Large** | Removes dual accumulators, mask-family switching, `vcgmax` opacity, and `vpack`/`vselr` ceremony |
| 3 | **Outer DMA / tiling shell** | **Small / tie** | Same GM→UB setup, flags, and stride math in all layers |

**Takeaway:** VMI pays off most where **logical width exceeds one physical
register** or **narrow types require nested part/store choreography**. Block MX
quant hits both pain points in step 2.

---

## 5. Side-by-side: OCP scale (`ComputeOcp_bf16`)

### 5.1 Per-row exponent accumulation

**VMI** — one logical vector, one accumulator:

```mlir
%xBitsU16 = pto.vmi.load %xAddr[%loadOffset] : ... -> !pto.vmi.vreg<256xui16>
%xExp = pto.vmi.andi (pto.vmi.bitcast %xBitsU16), %expMaskBF16 : ...
%nextAcc = pto.vmi.select (pto.vmi.cmpi "slt", %acc, %xExp), %xExp, %acc : ...
```

**MI** — DINTLV split + dual `vmax`:

```mlir
%x0Bf16, %x1Bf16 = pto.vldsx2 %xBf[%loadOffset], "DINTLV_B16" : ...
%x0ExpBF16 = pto.vand %x0Bits, %expMaskBF16, %pregAllB16 : ...
%x1ExpBF16 = pto.vand %x1Bits, %expMaskBF16, %pregAllB16 : ...
%nextAcc0 = pto.vmax %acc0, %x0ExpBF16, %pregAllB16 : ...
%nextAcc1 = pto.vmax %acc1, %x1ExpBF16, %pregAllB16 : ...
```

**CCE** (`bmx_cce_kernels.h`) — same split as MI:

```c
vlds(x0Bf16, x1Bf16, xBf, loadOffsetOcp, DINTLV_B16);
vand(x0ExpBF16, (vector_u16 &)x0Bf16, expMaskBF16, pregAll, MODE_ZEROING);
vand(x1ExpBF16, (vector_u16 &)x1Bf16, expMaskBF16, pregAll, MODE_ZEROING);
vmax(expMax1Dim2, expMax1Dim2, x0ExpBF16, pregAll, MODE_ZEROING);
vmax(expMax2Dim2, expMax2Dim2, x1ExpBF16, pregAll, MODE_ZEROING);
```

VMI advantage: the algorithm reads as “mask exponent bits, running max over 256
lanes”. MI/CCE force you to track **even/odd half ownership** from the first
load.

### 5.2 Cross-row reduction and scale encoding

**VMI:**

```mlir
%expMaxScalar32 = pto.vmi.group_reduce_maxi %expMaxRows32, %mask256 {num_groups = 1} : ...
%expMaxDim32 = pto.vmi.group_broadcast %expMaxScalar32 {num_groups = 1} : ...
%mxScale1Full = pto.vmi.trunci (pto.vmi.shrui ...) : !pto.vmi.vreg<256xui8>
```

**MI:**

```mlir
%expMaxDim0 = pto.vcgmax %expMaxDimPre, %pregAllB16 : ...
%mxScale1B8 = pto.vpack %mxScaleB16_2, "LOWER" : !pto.vreg<128xi16> -> !pto.vreg<256xui8>
```

**CCE:**

```c
vmax(expMaxDim, expMax1Dim2, expMax2Dim2, pregAll, MODE_ZEROING);
vcgmax(expMaxDim, expMaxDim, pregAll, MODE_ZEROING);
vshrs(mxScaleB16, expMaxDim, SHR_NUM_FOR_BF16, pregAll, MODE_ZEROING);
vpack(mxScale1B8, mxScaleB16, LOWER);
```

VMI makes reduction and narrowing **two explicit semantic steps**
(`group_reduce_maxi`, `group_broadcast`, `trunci`). MI/CCE fuse reduction with
broadcast (`vcgmax`) and use hardware pack (`vpack LOWER`) instead of typed
truncate.

### 5.3 scale2 and reciprocal stores

MI/CCE write scale2 via **gather**:

```c
vlds(gatherIndex, gatherIndexAddr, 0, NORM);
vselr(mxScale2B8, mxScale1B8, gatherIndex);
vsts(mxScale2B8, mxScale2Addr, 0, NORM_B8, pregB8);
```

VMI stores the fully broadcast scale vector directly when every lane already
holds the same post-reduction value — the gather becomes a compiler/layout
detail, not algorithm code.

---

## 6. Side-by-side: f16 → fp8 E4M3 quant

This is the clearest VMI win in the MX quant suite.

### 6.1 Loop body size

| Layer | Semantic steps per 256-element row | Layout/repair ops |
|-------|-----------------------------------|-------------------|
| **VMI** | 5 (`load`, `extf`, `mulf`, `truncf`, `masked_store`) | 0 explicit |
| **MI** | 17 total | ~12 (splits, converts, interleaves, bitcasts, PK stores) |
| **CCE** | same structure as MI | same |

### 6.2 VMI quant path (reads like the math)

```mlir
%x_f16 = pto.vmi.load %x_ub[%row_off] : ... -> !pto.vmi.vreg<256xf16>
%x_fp32 = pto.vmi.extf %x_f16 : ... -> !pto.vmi.vreg<256xf32>
%res_fp32 = pto.vmi.mulf %x_fp32, %scale_fp32 : ...
%res_fp8 = pto.vmi.truncf %res_fp32 : ... -> !pto.vmi.vreg<256xf8E4M3FN>
pto.vmi.masked_store %res_fp8, %y_ub[%row_off], %full_mask : ...
```

### 6.3 MI quant path (same math, hardware vocabulary)

```mlir
%x0F16, %x1F16 = pto.vldsx2 %xHalf[%blockBase], "DINTLV_B16" : ...
%x0Zero0 = pto.vcvt %x0F16, %pregAllB16 {part = "EVEN"} : ... -> !pto.vreg<64xf32>
%x0One0  = pto.vcvt %x0F16, %pregAllB16 {part = "ODD"}  : ... -> !pto.vreg<64xf32>
%x0Zero1 = pto.vmul %x0Zero0, %scaleForMulFP32, %pregAllB32 : ...
// ... two more vmul, four vintlv, four vcvt {rnd=R, sat=SAT, part=P0},
//     four vbitcast, four vsts {dist = "PK4_B32"} ...
```

### 6.4 CCE reference (FP16 branch excerpt)

```c
vlds(x0F16, x1F16, xHalf, loadStrideY8, DINTLV_B16, POST_UPDATE);
vcvt(x0ZeroFP32, x0F16, pregAll, PART_EVEN, MODE_ZEROING);
vcvt(x0OneFP32,  x0F16, pregAll, PART_ODD,  MODE_ZEROING);
vmul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, pregB32, MODE_ZEROING);
vintlv(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
// ... mirror for x1, cross-vintlv, four vcvt(..., ROUND_R, RS_ENABLE, PART_P0, ...),
//     four vsts(..., PK4_B32, pregB8)
```

Same story as RoPE interleave, but longer: **the quant algorithm is a multiply
and a cast**; MI/CCE spend most lines rebuilding **logical index order** after
every hardware split.

---

## 7. What VMI does *not* simplify

Be explicit about remaining low-level work:

| Concern | Still visible in VMI examples? |
|---------|-------------------------------|
| Row/block loop structure | **Yes** — `scf.for` over rows or blocks |
| `vlForHalfNumber`, `ubBlockSize`, stride products | **Yes** |
| Special-case scale math (Inf/NaN/zero/clamp) | **Yes** — still authored, just without per-op masks |
| DMA GM↔UB, pipeline `set_flag` / `wait_flag` | **Yes** in the full VMI kernel wrapper |
| FP16 scale path inside `ComputeOcp` | **Not in bf16 demo** — production CCE still has FP16-only Inf/NaN handling before widen |
| Verifying PK4 lane placement | **Harder from VMI source** — use lowered MI or CCE sim when debugging store artifacts |

---

## 8. Practical guidance

| Task | Prefer |
|------|--------|
| Author or review MX scale + quant math | **VMI** |
| Bit-exact compare against CCE sim / intrinsics golden | **MI** or **`bmx_cce_kernels.h`** |
| Debug wrong FP8 lanes / PK offsets / padding | Lowered **MI** (VMI hides the failure point) |
| Learn DINTLV + part + PK rules hands-on | **MI** + `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md` |

**Suggested reading order:**

1. Section 1 of this doc (math)
2. `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto` — best “before/after” showcase
3. `mx_block_quant_y1_fp8_f16_e4m3.mi.pto` — see what VMI removed
4. `mx_block_quant_scale_ocp_bf16.vmi.pto` vs `.mi.pto` — scale path
5. `bmx_cce_kernels.h` `ComputeOcp` / `ComputeY1ToFP8` — annotated ground truth

---

## 9. One-page summary

| Question | Answer |
|----------|--------|
| Same MX semantics? | **Yes**, for the covered bf16 scale + f16→E4M3 quant paths |
| Cleaner syntax? | **Yes** — largest on quant loop bodies, large on scale compute |
| Closer to math? | **Yes** — VMI keeps “max exponent → scale → multiply → cast” visible |
| Easier than CCE? | **Yes for compute bodies**; **tie** for DMA/tiling |
| Biggest win? | **f16→fp8 quant**, where MI spends most lines on layout repair |

VMI is not “MX quant in five lines.” It is **MX quant with the split/part/PK
contract moved into the compiler** — most valuable exactly where block quant
touches **256-wide logical tiles**, **mixed precisions**, and **sub-byte output
packing**.

---

## Example file index

| File | Layer | Stage | Notes |
|------|-------|-------|-------|
| `mx_block_quant_scale_ocp_bf16.vmi.pto` | VMI | Scale (OCP) | 256-wide vectors, explicit group reduce/broadcast |
| `mx_block_quant_scale_ocp_bf16.mi.pto` | MI | Scale (OCP) | DINTLV + dual accumulators + `vcgmax` + `vpack` |
| `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto` | VMI | Quant | Full kernel with DMA; **best VMI demo** |
| `mx_block_quant_y1_fp8_f16_e4m3.mi.pto` | MI | Quant | 17-op loop body, CCE-faithful layout |
| `bmx_cce_kernels.h` | CCE | Both | `ComputeOcp`, `ComputeY1ToFP8` ground truth |

Related docs:

- `ROPE_VMI_vs_MI.md` — same three-layer comparison pattern on RoPE
- `PTO-Gym-vmi/docs/PTO-vmi-design.en.md`
- `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
