# RoPE on PTO: Why VMI Helps Algorithm Engineers

This note compares the RoPE example kernels in this directory across three
expression layers:

| Layer | Example files | Who it is for |
|-------|---------------|---------------|
| **CCE** | `rope_cce_compute.h` | Ground-truth intrinsics execution |
| **MI** (`pto.mi`) | `rope_{f16,bf16,f32}.mi.pto`, `rope_f16_v2.mi.pto` | Hardware-faithful PTO micro-ops |
| **VMI** (`pto.vmi`) | `rope_{f16,bf16,f32}.vmi.pto` | Logical-vector authoring |

**Audience:** algorithm engineers who want the math to be visible in code and
prefer not to manage SIMD register layout, mask families, or load/store
distribution modes by hand.

**Scope:** the six RoPE variants in the examples — `{f16, bf16, f32} × {HALF,
INTERLEAVE}` — on the fixed demo tile `D=64`. All examples share the same outer
kernel shell (GM→UB DMA, pipeline flags, UB→GM writeback).

This document is intentionally honest: VMI is a real abstraction win in several
places, but it is not magic, and it does not remove all low-level work.

---

## 1. RoPE math (what every variant implements)

RoPE rotates pairs of dimensions by a position-dependent angle θ using
precomputed `(cos θ, sin θ)`.

### 1.1 HALF mode (NeoX / LLaMA-style layout)

The head dimension `D` is split into two contiguous halves. For each index `d`
in the low half:

```
y[d]       = x[d]       * cos[d]       - x[d + D/2] * sin[d]
y[d+D/2]   = x[d+D/2]   * cos[d+D/2] + x[d]       * sin[d+D/2]
```

This is a **2×2 real rotation** applied independently to `(x_low, x_high)`
pairs. No complex-number bookkeeping is required in code — just two fused
half-vector updates.

### 1.2 INTERLEAVE mode (GPT-J-style layout)

Adjacent elements form pairs: `(x[2k], x[2k+1])`. Two algebraically equivalent
spellings appear in the examples:

**Complex-multiply form** (CCE, `rope_f16_v2.mi.pto`, f16/bf16 VMI interleave):

```
y = x * cos + (i*x) * sin
```

where `(i*x)` in interleaved real layout is `[-x1, x0, -x3, x2, ...]`.

**Cartesian expansion form** (`rope_f16.mi.pto` v1, f32 VMI interleave):

```
y_even = x_even * cos_even - x_odd * sin_even
y_odd  = x_odd  * cos_odd  + x_even * sin_odd
```

after deinterleaving even/odd streams from `[x0, x1, x2, x3, ...]`.

Both are standard; they differ in **how** you schedule the SIMD ops, not in the
RoPE result.

### 1.3 bf16 numerics

For bf16 paths, **x/y are bf16** but **cos/sin are fp16**, and the inner
compute is done in **fp32** before narrowing back to bf16. This matches the CCE
reference (`ComputeBf16`), not native bf16 arithmetic throughout.

---

## 2. What VMI actually is

VMI (`pto.vmi`) sits between your algorithm and MI (`pto.mi`). You write
**logical contiguous vectors** (`!pto.vmi.vreg<N×T>`) and **semantic ops**
(`vmi.load`, `vmi.mulf`, `vmi.extf`, `channel_split`, …). The compiler
(`pto.as`) lowers to MI instructions with the correct `dist`, `part`, `PK/UNPK`,
and mask granularity.

See `PTO-Gym-vmi/docs/PTO-vmi-design.en.md` for
the full design.

**VMI removes:** per-op mask plumbing, register layout choices, and
type-conversion protocol details.

**VMI does not remove:** loop nesting (`s → rep/blk → n`), stride/alignment
math, DMA setup, or pipeline synchronization. Those remain visible in the
examples.

---

## 3. Instruction map: math → MI → VMI

The table below connects the math step to what you write at each layer.

| Math intent | CCE intrinsic | MI (`pto.mi`) | VMI (`pto.vmi`) |
|-------------|---------------|---------------|-----------------|
| Load a logical vector from UB | `vlds` (`NORM` / `UNPK_B16`) | `pto.vlds` (+ optional `{dist = "UNPK_B16"}`) | `pto.vmi.load` |
| Widen bf16/fp16 → fp32 | `vcvt` (`PART_EVEN`, `MODE_ZEROING`) | `pto.vcvt {part = "EVEN"}` | `pto.vmi.extf` |
| Narrow fp32 → bf16/fp16 | `vcvt` (`ROUND_R`, `RS_DISABLE`, `PART_EVEN`, `MODE_ZEROING`) | `pto.vcvt {part = "EVEN", rnd, sat}` | `pto.vmi.truncf` |
| Element-wise multiply / add / sub | `vmul` / `vadd` / `vsub` (`MODE_ZEROING` + mask) | same + **mask on every op** | `vmi.mulf` / `addf` / `subf` (no mask) |
| Tail / partial vector mask | `plt_b16` / `plt_b32` (`POST_UPDATE`) | `pto.plt_b16` / `pge_b32 "PAT_VL32"` | `pto.vmi.create_mask %active` |
| Split interleaved even/odd | `vdintlv` | `pto.vdintlv` | `pto.vmi.channel_split` |
| Negate odd lane stream | `vbr` + `vmul` (`MODE_ZEROING`) | same | `pto.vmi.negf` |
| Merge even/odd back | `vintlv` | `pto.vintlv` | `pto.vmi.channel_merge` |
| Store with tail mask | `vsts` (`NORM_B16` / `NORM_B32` / `PK_B32` + mask) | `pto.vsts` (+ optional `{dist = "PK_B32"}`) + mask | `pto.vmi.masked_store` |

### 3.1 MI details algorithm engineers should not have to memorize

These are the hardware subtleties VMI is designed to hide:

**UNPK_B16 load:** 64 dense b16 elements in memory expand into a 128-lane
physical register with valid values only at **even** halfword positions; odd
lanes are padding. If you later widen with the wrong `part`, you read zeros or
garbage.

**PART_EVEN vcvt:** selects the even-position lanes after an UNPK load. Required
for bf16/fp16 widen paths in MI.

**PK_B32 store:** packs bf16 values scattered at even register positions back
into dense memory. Required after narrowing in MI bf16 paths.

**Mask family mismatch:** f16 arithmetic uses `!pto.mask<b16>`; fp32 uses
`!pto.mask<b32>`. A bf16 kernel typically needs **both** in the same loop body.

**vdintlv / vintlv:** hardware deinterleave/interleave with a specific lane
mapping. Correct for performance, opaque for reading.

VMI replaces these with “load 64 bf16”, “extend to fp32”, “store bf16”, and
“split/merge even/odd channels”.

---

## 4. Advantage by variant (ranked)

Honest ranking of **VMI syntax / intuitiveness / math closeness** vs MI,
from largest win to smallest:

| Rank | Variant | VMI advantage | Why |
|------|---------|---------------|-----|
| 1 | **bf16 HALF** | **Very large** | MI exposes the full UNPK→EVEN→compute→EVEN→PK chain + dual mask families on every op |
| 2 | **bf16 INTERLEAVE** | **Very large** | Same conversion chain, plus `channel_split/merge/negf` vs `vdintlv/vintlv/vbr` |
| 3 | **f16 INTERLEAVE** | **Large** | `channel_split → negf → channel_merge` reads as “build i·x”; optional fp32 middle via `extf/truncf` |
| 4 | **f32 INTERLEAVE** | **Moderate** | Cartesian expansion is already readable; VMI mainly removes per-op masks and names shuffles semantically |
| 5 | **f16 HALF** | **Moderate** | Core 6-op math is identical; VMI drops mask args and uses logical `create_mask` |
| 6 | **f32 HALF** | **Small** | MI f32 HALF is already close to the math; VMI mostly removes redundant mask parameters |

**Takeaway:** the further you are from “native dense fp32 load/compute/store”, the
more VMI pays off. **bf16 is the headline case.**

If MI is written like `rope_f16_v2.mi.pto` (CCE-faithful loops and masks), VMI
still wins on **bf16** and **mask-free arithmetic**, but the **algorithm
structure** gap shrinks — you are mostly comparing syntax sugar vs hardware
protocol.

---

## 5. Walkthroughs

### 5.1 bf16 HALF — largest VMI win

**Math (fp32 inner loop):**

```
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

**MI** (`rope_bf16.mi.pto`) — one cos load/widen sequence:

```mlir
%cos_lo_16 = pto.vlds %cos_ub[%cs_off] {dist = "UNPK_B16"} : ... -> !pto.vreg<128xf16>
%cos_lo = pto.vcvt %cos_lo_16, %mask16_all {part = "EVEN"} : ... -> !pto.vreg<64xf32>
// ... repeat for sin, x halves ...
%t0 = pto.vmul %cos_lo, %x_lo, %mask32_half : ...
%t1 = pto.vmul %sin_lo, %x_hi, %mask32_half : ...
%y_lo_f32 = pto.vsub %t0, %t1, %mask32_half : ...
%y_lo_16 = pto.vcvt %y_lo_f32, %mask32_half {part = "EVEN", rnd = "R", sat = "SAT"} : ...
pto.vsts %y_lo_16, %y_ub[%y_off], %mask32_half {dist = "PK_B32"} : ...
```

Reading this as an algorithm engineer:

- Four instruction **families** before the first multiply: UNPK load, EVEN
  widen, masked fp32 op, EVEN narrow + PK store.
- `%mask16_all` for loads/converts, `%mask32_half` for compute — easy to mix up.
- `{part = "EVEN"}` is a correctness requirement, not an optimization knob.

**VMI** (`rope_bf16.vmi.pto`) — same cos load/widen:

```mlir
%cos1_16 = pto.vmi.load %cos_ub[%cos1_off] : ... -> !pto.vmi.vreg<64xf16>
%cos1 = pto.vmi.extf %cos1_16 : ... -> !pto.vmi.vreg<64xf32>
// ...
%x1_cos = pto.vmi.mulf %x1, %cos1 : ...
%x2_sin = pto.vmi.mulf %x2, %sin1 : ...
%out1_f32 = pto.vmi.subf %x1_cos, %x2_sin : ...
%out1 = pto.vmi.truncf %out1_f32 : ... -> !pto.vmi.vreg<64xbf16>
pto.vmi.masked_store %out1, %y_ub[%y1_off], %mask : ...
```

This reads as the math plus dtype semantics: **load → widen → compute →
truncate → store**. The compiler chooses UNPK/PART_EVEN/PK_B32 and mask
families.

**CCE** (`ComputeBf16` HALF) documents the same protocol explicitly in comments
— excellent for verification, heavy for authoring.

---

### 5.2 bf16 INTERLEAVE — complex-multiply form in VMI

**Math:**

```
y = x * cos + (i*x) * sin
```

**VMI** (`rope_bf16.vmi.pto`):

```mlir
%x = pto.vmi.extf %x16 : ... -> !pto.vmi.vreg<128xf32>
%x_even, %x_odd = "pto.vmi.channel_split"(%x) : ... -> (!pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>)
%neg_x_odd = pto.vmi.negf %x_odd : ...
%rot = "pto.vmi.channel_merge"(%neg_x_odd, %x_even) : ... -> !pto.vmi.vreg<128xf32>
%x_cos = pto.vmi.mulf %x, %cos : ...
%rot_sin = pto.vmi.mulf %rot, %sin : ...
%y_f32 = pto.vmi.addf %x_cos, %rot_sin : ...
%y16 = pto.vmi.truncf %y_f32 : ... -> !pto.vmi.vreg<128xbf16>
```

Instruction meaning for algorithm engineers:

- **`channel_split`:** `[e0,o0,e1,o1,…] → (even=[e0,e1,…], odd=[o0,o1,…])`.
  Semantic name for MI `vdintlv`.
- **`negf`:** negate the odd stream → `-odd`.
- **`channel_merge`:** interleave `(-odd, even)` → `[-o0,e0,-o1,e1,…]`, which
  is `(i*x)` in interleaved layout.
- **`mulf` / `addf`:** the two-term complex multiply, mask-free.

**MI v1** (`rope_bf16.mi.pto`) uses **Cartesian expansion** instead: deinterleave
cos, sin, **and** x, then eight masked fp32 ops, then `vintlv`, then narrow +
PK store. Correct, but the code looks like a shuffle kernel, not `y = x·cos +
(i·x)·sin`.

**MI v2 f16** (`rope_f16_v2.mi.pto`) matches the CCE complex-multiply path and
is closer to VMI structurally — but still uses `vbr`, `plt_b16`, and masked
`vmul/vadd`.

---

### 5.3 f16 HALF — moderate win

The six-op HALF core is the same everywhere:

```
out1 = x1*cos - x2*sin
out2 = x2*cos + x1*sin
```

**MI** (every op carries a mask):

```mlir
%t0 = pto.vmul %cos_lo, %x_lo, %mask16 : ...
%t1 = pto.vmul %sin_lo, %x_hi, %mask16 : ...
%y_lo = pto.vsub %t0, %t1, %mask16 : ...
```

**VMI:**

```mlir
%x1_cos = pto.vmi.mulf %x1_16, %cos1_16 : ...
%x2_sin = pto.vmi.mulf %x2_16, %sin1_16 : ...
%out1 = pto.vmi.subf %x1_cos, %x2_sin : ...
```

VMI advantage here is **clarity, not algorithm**: masks and load/store dist
modes disappear from the math block. Loop tiling (`rep`, `half_d_aligned`,
dynamic tail) remains equally verbose in both.

---

### 5.4 f16 INTERLEAVE — VMI matches CCE math spelling

VMI f16 interleave uses the **complex-multiply form** with an fp32 middle
(`extf` before shuffle, `truncf` before store). That is a **semantic** match to
CCE/v2 MI, with a small **numeric** difference vs native f16 throughout (VMI
chooses fp32 for the interleave body because `extf/truncf` are cheap at the
logical level).

Compare the fused body:

**VMI:**

```mlir
%rot = "pto.vmi.channel_merge"(%neg_x_odd, %x_even) : ...
%out_f32 = pto.vmi.addf (pto.vmi.mulf %x, %cos), (pto.vmi.mulf %rot, %sin) : ...
```

**CCE / MI v2:**

```c
vdintlv(heven, hodd, xr, xr);
vmul(hnegodd, hodd, negOne, maskPair, MODE_ZEROING);
vintlv(hxnew, hxnew_hi, hnegodd, heven);
vmul(hta, xr, cosr, mask, MODE_ZEROING);
vmul(htb, hxnew, sinr, mask, MODE_ZEROING);
vadd(htb, hta, htb, mask, MODE_ZEROING);
```

Same story, different vocabulary: **merge(−odd, even)** vs **vintlv after
negate**.

---

### 5.5 f32 HALF — smallest win

f32 is the “easy” case: dense loads, native fp32 compute, dense stores. MI is
already readable if you ignore the mask on every line.

**MI:**

```mlir
%t0 = pto.vmul %cos_lo, %x_lo, %mask32_half : ...
%y_lo = pto.vsub %t0, %t1, %mask32_half : ...
pto.vsts %y_lo, %y_ub[%y_off], %mask32_half : ...
```

**VMI:**

```mlir
%x1_cos = pto.vmi.mulf %x1, %cos1 : ...
%out1 = pto.vmi.subf %x1_cos, %x2_sin : ...
pto.vmi.masked_store %out1, %y_ub[%y1_off], %mask : ...
```

VMI still helps (uniform `vmi.load`, one mask type, no `pge_b32`), but this is
not where you save days of debugging UNPK/PK issues.

---

### 5.6 f32 INTERLEAVE — Cartesian form; moderate win

The f32 VMI example intentionally uses **Cartesian expansion** (like MI v1),
not the complex-multiply fused path used for f16/bf16 VMI interleave:

```mlir
%y_even = pto.vmi.subf (pto.vmi.mulf %x_even, %cos_even),
                       (pto.vmi.mulf %x_odd, %sin_even) : ...
%y_odd  = pto.vmi.addf (pto.vmi.mulf %x_odd, %cos_odd),
                       (pto.vmi.mulf %x_even, %sin_odd) : ...
%out = "pto.vmi.channel_merge"(%y_even, %y_odd) : ...
```

This is honest: the math **is** the Cartesian expansion, and VMI makes it
readable via `channel_split/merge` instead of `vdintlv/vintlv`. The win over MI
is real but incremental — you are mostly shedding eight mask parameters and
getting semantic shuffle names.

Note: CCE f32 INTERLEAVE uses the **complex-multiply** path (like f16 v2), so
VMI f32 interleave is **not** a line-for-line spelling match to CCE — it is an
equivalent Cartesian organization chosen for clarity at the logical vector
width.

---

## 6. What VMI does not fix (honest limits)

1. **Tiling and strides** — `half_d_aligned`, `rep`/`blk` loops, `x_s_step`, and
   DMA byte rounding are identical in burden across MI and VMI examples.

2. **Not all MI spellings match** — MI v1 interleave (Cartesian) vs MI v2/CCE
   (complex-multiply) vs VMI f32 interleave (Cartesian) vs VMI f16 interleave
   (complex-multiply). VMI simplifies each spelling, but authors still pick the
   spelling.

3. **Bit-exact parity** — VMI f16 interleave uses fp32 middle math; CCE f16
   interleave is native f16. Expect small numeric differences, not algorithmic
   ones.

4. **Performance visibility** — hiding `dist/part/PK` means the compiler must
   choose correctly. For peak tuning, engineers still inspect lowered MI (same
   as inspecting assembly after CCE).

5. **HALF vs INTERLEAVE layout** — VMI does not choose NeoX vs GPT-J for you;
   `mode` still branches the kernel.

---

## 7. Practical guidance

| If you are… | Prefer |
|-------------|--------|
| Writing or reviewing RoPE math, especially bf16 | **VMI** |
| Verifying against CCE sim / intrinsics golden | **MI v2** or **CCE** |
| Debugging wrong lanes / padding / PK artifacts | Lowered **MI** (VMI source may hide the bug location) |
| f32 HALF only, team already knows MI masks | Either; VMI is nicer, not essential |

**Suggested reading order for new algorithm engineers:**

1. Section 1 of this doc (math)
2. `rope_bf16.vmi.pto` HALF loop body — best VMI showcase
3. `rope_bf16.mi.pto` same section — see what VMI removed
4. `rope_cce_compute.h` `ComputeBf16` — ground truth with educational comments
5. `rope_f16_v2.mi.pto` — CCE-faithful MI reference for f16

---

## 8. One-page summary

| Question | Answer |
|----------|--------|
| Same RoPE semantics? | **Yes**, modulo bf16/fp32 inner precision choices and f32 interleave spelling variant |
| Cleaner syntax? | **Yes**, strongest for bf16, moderate for f16 interleave, modest for f32 HALF |
| Closer to math? | **Yes** where VMI hides UNPK/PK/part and uses `channel_split/merge` for GPT-J |
| Easier than CCE? | **Yes for compute bodies**; **tie** for DMA/tiling shell |
| Still worth it if MI is v2-quality? | **Yes for bf16**; **marginal for f32 HALF** |

VMI is not “RoPE in three lines.” It is **RoPE with the hardware contract moved
into the compiler** — most valuable exactly where that contract is longest:
**mixed-precision loads, widened compute, and interleaved layouts.**

---

## Example file index

| File | Layer | Dtype | Notes |
|------|-------|-------|-------|
| `rope_f16.vmi.pto` | VMI | f16 | HALF native f16; INTERLEAVE complex-multiply + fp32 middle |
| `rope_bf16.vmi.pto` | VMI | bf16 | **Best VMI demo** — extf/truncf throughout |
| `rope_f32.vmi.pto` | VMI | f32 | HALF simple; INTERLEAVE Cartesian expansion |
| `rope_f16.mi.pto` | MI v1 | f16 | INTERLEAVE Cartesian expansion; simplified tiling |
| `rope_f16_v2.mi.pto` | MI v2 | f16 | CCE-faithful; INTERLEAVE complex-multiply |
| `rope_bf16.mi.pto` | MI | bf16 | Full UNPK/EVEN/PK exposure |
| `rope_f32.mi.pto` | MI | f32 | Mask on every op |
| `rope_cce_compute.h` | CCE | all | Intrinsics ground truth |

Related docs:

- `PTO-Gym-vmi/docs/PTO-vmi-design.en.md`
- `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
- `PTO-Gym-vmi/docs/PTO-micro-Instruction-SPEC.md`
