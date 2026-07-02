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

**Reader contract:** after reading this note, you should be able to connect the
RoPE equations to the VMI source, predict the main MI/CCE lowering shape
(`dist`, `part`, masks, packing), and recognize the common lane-layout bugs VMI
is meant to prevent.

**Concrete VF tile used by the simulator examples:** the correctness tests use
`sCount=15`, `nCount=32`, `dLen=dAlign=64`; the per-head stride is
`xNStep=yNStep=csSStep=64`, and the per-sequence stride is
`xSStep=ySStep=nCount*64=2048` elements. Wall-time configs also exercise
`(s,n)=(1,2),(15,4),(15,8),(15,16),(15,32)`.

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

The VMI cast names intentionally follow MLIR `arith` vocabulary. `pto.vmi.extf`
means “extend each logical lane to a wider floating type”, and
`pto.vmi.truncf` means “narrow each logical lane to the destination floating
type”. They are not claims about the number of hardware instructions. On bf16
RoPE, one `extf` may lower through `vlds UNPK_B16` and `vcvt PART_EVEN`; one
`truncf` plus `masked_store` may lower through `vcvt PART_EVEN` and `vsts
PK_B32`. The abstraction is about preserving the algorithmic lane contract, not
making the hardware layout disappear.

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

How to read the instruction map:

- `vmi.load` is a logical UB load. In f32 and dense f16 cases it lowers close to
  `vlds NORM`. In bf16/f16 widen paths it may need `UNPK_B16` so that the next
  MI `vcvt PART_EVEN` sees valid halfword lanes.
- `vmi.mulf`, `vmi.addf`, and `vmi.subf` are element-wise arithmetic on logical
  vectors. MI still needs `mask<b16>` or `mask<b32>` on every arithmetic op;
  VMI carries the active-lane predicate at load/store or higher-level mask
  creation points.
- `channel_split` and `channel_merge` describe parity layout intent. They are
  the VMI names for the same even/odd axis that MI exposes with `vdintlv` and
  `vintlv`.
- `masked_store` says “write active logical lanes”. MI still decides whether the
  store is a plain `NORM_*` store or a packing store such as `PK_B32`.

### 3.1 How to use the side-by-side snippets

When reading the walkthroughs below, separate each instruction into one of four
roles:

| Role | Question to ask | RoPE examples |
|------|-----------------|---------------|
| **MATH** | Does this change the value? | `mulf`, `addf`, `subf`, `negf` |
| **TYPE** | Does this change dtype while preserving logical lane `i`? | `extf`, `truncf`, `vcvt` |
| **LAYOUT** | Does this only move lanes around? | `UNPK_B16`, `PK_B32`, `PART_EVEN`, `vdintlv`, `vintlv` |
| **MEMORY** | Does this cross the UB/register boundary? | `vmi.load`, `masked_store`, `vlds`, `vsts` |

Most VMI lines are **MATH** or **TYPE**. Many MI/CCE lines are **LAYOUT** or
**MEMORY** required by the 256-byte register contract.

### 3.2 Expected lowering shapes for recurring VMI ops

These are not formal lowering rules, but they are useful review expectations for
the concrete examples in this directory.

**bf16 load + widen in HALF mode:**

```mlir
%x16 = pto.vmi.load %x_ub[%off] : ... -> !pto.vmi.vreg<64xbf16>
%x32 = pto.vmi.extf %x16 : ... -> !pto.vmi.vreg<64xf32>
```

Expected MI/CCE shape:

```mlir
%x16_phys = pto.vlds %x_ub[%off], %mask16 {dist = "UNPK_B16"} : ...
%x32 = pto.vcvt %x16_phys, %mask16 {part = "EVEN"} : ... -> !pto.vreg<64xf32>
```

The `EVEN` part is not an optimization; it is required because `UNPK_B16` places
dense bf16 memory values into even physical lanes.

**bf16 narrow + store:**

```mlir
%y16 = pto.vmi.truncf %y32 : ... -> !pto.vmi.vreg<64xbf16>
pto.vmi.masked_store %y16, %y_ub[%off], %mask : ...
```

Expected MI/CCE shape:

```mlir
%y16_phys = pto.vcvt %y32, %mask32 {part = "EVEN", rnd = "R", sat = "SAT"} : ...
pto.vsts %y16_phys, %y_ub[%off], %mask32 {dist = "PK_B32"} : ...
```

`truncf` preserves logical lane order; `PK_B32` repairs the physical even-lane
layout before values reach dense UB memory.

**INTERLEAVE rotation helper:**

```mlir
%even, %odd = "pto.vmi.channel_split"(%x) : ...
%rot = "pto.vmi.channel_merge"(%neg_odd, %even) : ...
```

Expected MI/CCE shape:

```mlir
%even, %odd = pto.vdintlv %x, %x : ...
%rot, %rot_hi = pto.vintlv %neg_odd, %even : ...
```

The argument order matters: `channel_merge(neg_odd, even)` builds
`[-x1, x0, -x3, x2, ...]`, i.e. `(i*x)` in GPT-J interleaved layout.

### 3.3 MI details algorithm engineers should not have to memorize

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

### 3.4 Common bugs VMI helps avoid

- **Wrong `part` after `UNPK_B16`:** using `PART_ODD` instead of `PART_EVEN`
  reads padding lanes, not bf16 values.
- **Wrong store distribution after bf16 narrow:** a dense `NORM_B16` store after
  even-lane narrow leaks padded lane layout into UB; bf16 paths need `PK_B32`.
- **Swapped interleave merge order:** `merge(even, neg_odd)` is not `(i*x)`;
  the CCE/VMI complex-multiply spelling needs `merge(neg_odd, even)`.
- **Mask-family drift:** bf16 paths mix `b16` load/convert masks with `b32`
  arithmetic masks. VMI keeps one logical active-lane mask and lets lowering pick
  the concrete predicate family.

### 3.5 Physical register model (256-byte contract, D=64 demo)

Every A5 vector register is **256 bytes**. The RoPE examples process one head row
of **D=64** elements per inner tile. How those 64 elements map to physical
registers depends on **dtype** and **layout mode**:

| Case | Memory bytes (64 elems) | Typical physical reg | Load `dist` |
|------|-------------------------|----------------------|-------------|
| f16 / fp16 dense | 128 B | `vreg<128×f16>` (64 active lanes) | `NORM` |
| bf16 dense | 128 B | `vreg<128×bf16>` after UNPK | `UNPK_B16` |
| f32 dense | 256 B | `vreg<64×f32>` | `NORM` |
| GPT-J interleaved (all dtypes) | same as above | one dense reg, then parity split | as above + `vdintlv` |

**Logical view (VMI):** `!pto.vmi.vreg<64×T>` or `!pto.vmi.vreg<128×T>` means
elements `0..N−1` in algorithm order. HALF mode treats indices `[0..D/2−1]` and
`[D/2..D−1]` as **partner halves**. INTERLEAVE mode treats `(2k, 2k+1)` as
**rotation pairs**.

**Physical view (MI/CCE):** each register has **lane slots** with placement
rules. UNPK, `part=EVEN/ODD`, `vdintlv`/`vintlv`, and `PK_B32` are not math —
they maintain the mapping between lane slot and logical index.

RoPE at D=64 is **smaller** than block MX quant (256-wide rows), so you rarely
need `DINTLV_B16` split loads in these examples. The dominant layout axes here
are **UNPK/PK** (bf16), **parity shuffle** (INTERLEAVE), and **mask family**
(b16 vs b32).

See also `BLOCK_MX_QUANT_VMI_vs_MI.md` §3.5–§3.9 for the same 256B contract
applied to 256-element quant rows (where `DINTLV` + four `vintlv` steps appear).

### 3.6 UNPK_B16 load and PK_B32 store (bf16 / fp16 dense paths)

bf16 RoPE loads **64 dense halfwords** from UB, but a b16 register holds
**128 lanes**. `UNPK_B16` expands each memory element into an **even physical
lane**, leaving odd lanes as zero/padding:

```
Memory (32 dense bf16 in one HALF tile, D/2=32):
  [b0, b1, b2, ..., b31]   ← 64 bytes

After UNPK_B16 → vreg<128×bf16> (256 B):
  phys lane:  0    1    2    3    4  ...  62   63   64  ... 127
  holds:      b0   __   b1   __   b2  ...  b31  __   __  ...  __
              ↑ even only; odd lanes are padding
```

**Widen:** `vcvt {part=EVEN}` reads lanes `0,2,4,…,126` → **64×f32** register
with the 32 meaningful values in the active mask lanes. Using `part=ODD` would
read the padding lanes and produce garbage — CCE documents this explicitly in
`ComputeBf16`.

**Narrow + store:** `vcvt {part=EVEN}` writes fp32 results back into even b16
lanes, then `vsts {dist=PK_B32}` extracts even lanes and packs them into **dense
memory**:

```
After PART_EVEN narrow → vreg<128×bf16>:
  [y0, __, y1, __, y2, __, ...]

PK_B32 store → memory:
  [y0, y1, y2, ...]   ← dense again
```

**VMI:** `vmi.load` → `vreg<64×bf16>`, `vmi.extf` / `vmi.truncf`, `vmi.masked_store`.
The UNPK → EVEN → PK chain is compiler lowering, not author code.

### 3.7 HALF mode — contiguous halves (NeoX layout)

HALF mode splits **D=64** into two partner blocks of **32 elements** each. No
parity deinterleave is required — partners are already contiguous in memory:

```
Logical head row (D=64):
  x[0..31]  = low half (x1)     partner with  x[32..63] = high half (x2)

Memory layout (HALF):
  ┌─────────────────┬─────────────────┐
  │  x[0] … x[31]   │ x[32] … x[63]   │
  └─────────────────┴─────────────────┘
     load x_lo          load x_hi  (+halfD or +halfD_aligned offset)
```

**f16 / f32 HALF (dense):** each half fits in one register. MI loads
`vreg<128×f16>` with a **32-lane tail mask** (`pge_b16 "PAT_VL32"` or dynamic
`plt_b16` in v2). Only the low 32 physical lanes hold data; the register is
128-wide because that is the native b16 vector width.

**bf16 HALF:** each half uses UNPK (§3.6) + fp32 compute + PK store. MI needs
**two mask families** in one loop body: `mask<b16>` for UNPK load and PART_EVEN
widen of cos/sin, `mask<b32>` for fp32 `vmul`/`vsub`/`vadd`.

**Physical register picture (bf16 HALF, one partner half):**

```
cos_lo_16  ──UNPK──► [c0,__,c1,__,…] ──PART_EVEN──► cos_lo (64×f32, 32 active)
x_lo_16    ──UNPK──► [x0,__,x1,__,…] ──PART_EVEN──► x_lo  (64×f32, 32 active)
                              │
                              ├── vmul / vsub / vadd  (mask<b32>)
                              ▼
y_lo_f32 (64×f32) ──PART_EVEN narrow──► [y0,__,y1,__,…] ──PK_B32──► dense UB
```

VMI names the same math on logical `vreg<64×bf16>` / `vreg<64×f32>` without
 exposing the even-lane landing zone.

### 3.8 INTERLEAVE mode — parity axis (GPT-J layout)

GPT-J layout stores rotation pairs as **adjacent elements**:

```
Logical (D=64 interleaved):
  [x0, x1,  x2, x3,  x4, x5, ... x62, x63]
   └── pair 0 ──┘  └── pair 1 ──┘
```

Hardware `vdintlv` / `vintlv` implement the **parity axis** — the same axis
used by `PART_EVEN`/`PART_ODD` on widen, but here applied to **already-widened
or dense b16/f16 vectors** to separate even- and odd-indexed streams:

```
Dense reg x (128 lanes, 64 active interleaved):
  [x0, x1, x2, x3, x4, x5, x6, x7, ...]

vdintlv(x, x) →
  x_even: [x0, x2, x4, x6, ...]     ← even logical indices
  x_odd:  [x1, x3, x5, x7, ...]     ← odd logical indices
```

**Complex-multiply form** (CCE, `rope_f16_v2.mi.pto`, f16/bf16 VMI): build
`(i·x)` in interleaved layout without splitting cos/sin:

```
x_even, x_odd = vdintlv(x)
neg_x_odd     = x_odd * (-1)
x_rot, _      = vintlv(neg_x_odd, x_even)   → [-x1, x0, -x3, x2, ...]

y = x*cos + x_rot*sin                     ← 3 arithmetic ops on dense regs
```

Physical effect of `vintlv(neg_x_odd, x_even)` (8-element sketch):

```
neg_x_odd:  [-x1, -x3, -x5, -x7]
x_even:     [ x0,  x2,  x4,  x6]

x_rot:      [-x1, x0, -x3, x2, -x5, x4, -x7, x6]   ← interleaved (i·x)
```

**Cartesian expansion form** (MI v1, f32 VMI interleave): deinterleave **all**
streams, compute even/odd updates separately, merge back:

```
cos_even, cos_odd = vdintlv(cos)
sin_even, sin_odd = vdintlv(sin)
x_even,   x_odd   = vdintlv(x)

y_even = x_even*cos_even - x_odd*sin_even
y_odd  = x_odd*cos_odd  + x_even*sin_odd

y, _ = vintlv(y_even, y_odd)    ← re-pack to [y0,y1,y2,y3,...]
```

Same RoPE result; **more layout ops** in the Cartesian path (three `vdintlv` on
tables + one `vintlv` repack vs one `vdintlv` + one `vintlv` on `x` only).

**VMI mapping:**

| MI/CCE | VMI | Physical meaning |
|--------|-----|------------------|
| `vdintlv` | `channel_split` | dense → even/odd parity streams |
| `vintlv` | `channel_merge` | even/odd → dense interleaved |
| `vmul` + `vbr(-1)` | `negf` | negate odd stream |

### 3.9 Mask families and predicate width

MI RoPE kernels often carry **two predicate granularities** in one loop:

| Mask type | Used for | Example |
|-----------|----------|---------|
| `!pto.mask<b16>` | UNPK loads, b16 `vcvt`, dense f16 `vmul`, `plt_b16` tails | bf16 load/widen; f16 interleave |
| `!pto.mask<b32>` | fp32 `vmul`/`vsub`/`vadd`, PART_EVEN narrow, `PK_B32` store | bf16 HALF compute body |

A bf16 HALF loop typically creates **both** `%mask16_all` and `%mask32_half`.
Using a b16 mask on a b32 op (or vice versa) is a silent wrong-lane bug.

VMI uses one logical `!pto.vmi.mask<N×pred>` per tile; the compiler lowers to
the correct `pset`/`plt`/`punpack` family.

### 3.10 Variant → layout complexity (honest map)

| Variant | Physical layout burden | Dominant MI ops beyond arithmetic |
|---------|------------------------|-----------------------------------|
| **f32 HALF** | Low | Tail mask only |
| **f16 HALF** | Low | Tail mask; 128-wide reg, 32 active lanes |
| **bf16 HALF** | **High** | UNPK, PART_EVEN ×2, PK_B32, dual masks |
| **f16 INTERLEAVE v2** | Medium | `vdintlv`, `vbr`, `vintlv` on `x`; dense cos/sin |
| **f16 INTERLEAVE v1** | **High** | `vdintlv` ×3, 8 arith, `vintlv` repack |
| **bf16 INTERLEAVE** | **High** | UNPK/PK chain + parity shuffle in fp32 |
| **f32 INTERLEAVE VMI** | Medium | Cartesian `channel_split/merge`; no UNPK |

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

Instruction-by-instruction, that means:

- `%cos1_16 = pto.vmi.load ... -> vreg<64xf16>` reads the table as 64 logical
  f16 lanes. The MI equivalent has to remember that a b16 physical register is
  128 lanes and that `UNPK_B16` places useful values on even lanes.
- `%cos1 = pto.vmi.extf %cos1_16` is the same semantic operation as
  `arith.extf` lifted to a VMI vector register. The MI equivalent is `vcvt
  PART_EVEN` because of the previous unpacked physical layout.
- `%x1_cos = pto.vmi.mulf %x1, %cos1` is exactly the first term in
  `x1*cos`. The MI equivalent is `vmul` plus `mask<b32>`.
- `%out1_f32 = pto.vmi.subf %x1_cos, %x2_sin` is the low-half RoPE subtraction.
  No hidden numerical trick is implied; it is the same fp32 subtract as MI/CCE.
- `%out1 = pto.vmi.truncf %out1_f32` narrows each logical lane back to bf16.
  MI must spell the rounding/saturation and destination part explicitly.
- `pto.vmi.masked_store` writes the active logical lanes. MI/CCE use `vsts
  PK_B32` here because the narrowed bf16 values live in even physical lanes.

**CCE** (`ComputeBf16` HALF) documents the same protocol explicitly in comments
— excellent for verification, heavy for authoring.

**Physical layout (bf16 HALF, one 32-element partner half):**

| Step | MI/CCE op | Register state (logical indices) |
|------|-----------|----------------------------------|
| 0 | UB memory | 32 dense bf16: `x[0..31]` or `x[32..63]` |
| 1 | `vlds UNPK_B16` | 128×bf16 even lanes: phys `2k → x[k]` |
| 2 | `vcvt PART_EVEN` | 64×f32: active lanes hold same 32 values |
| 3 | `vmul` / `vsub` / `vadd` | fp32 partners combined (`mask<b32>`) |
| 4 | `vcvt PART_EVEN` narrow | bf16 back into even lanes of 128×bf16 |
| 5 | `vsts PK_B32` | dense 32 bf16 in UB |

VMI collapses steps 1–2 into `vmi.load` + `vmi.extf`, steps 4–5 into
`vmi.truncf` + `vmi.masked_store`.

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

**Physical layout (bf16 INTERLEAVE, complex-multiply, one 64-element row):**

```
UB dense (64 bf16 interleaved):  x0 x1 x2 x3 ... x63
        │ UNPK + PART_EVEN widen
        ▼
x (64×f32 logical, interleaved in one dense fp32 reg)
        │ channel_split  ≡  vdintlv(x, x)
        ├──────────────┬──────────────
        ▼              ▼
   x_even         x_odd        [x0,x2,...] [x1,x3,...]
        │ negf         │
        └──────┬───────┘
               │ channel_merge(neg_odd, even)  ≡  vintlv
               ▼
   x_rot = [-x1,x0,-x3,x2,...]     cos, sin stay dense (no vdintlv)
               │
               └── mulf/addf:  y = x*cos + x_rot*sin
                       │ truncf + PK
                       ▼
               dense bf16 interleaved out
```

Cartesian MI v1 instead runs `vdintlv` on cos/sin **and** x, performs eight
masked ops on parity streams, then `vintlv(y_even, y_odd)` to rebuild interleaved
output — see §3.8.

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

**Physical layout (f16 HALF):** partners are **two separate dense loads**, not a
parity split. For D=64 with `halfD=32`:

```
x_lo reg (128×f16):  lanes 0..31 ← x[0..31]     (mask covers 32 lanes)
x_hi reg (128×f16):  lanes 0..31 ← x[32..63]    (loaded from x + halfD_aligned)

Six fp32-equivalent ops (native f16 here):
  y_lo = cos_lo * x_lo - sin_lo * x_hi
  y_hi = cos_hi * x_hi + sin_lo * x_lo
```

No `vdintlv`/`vintlv` — the NeoX layout already matches how HALF loads data.

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

**Index map for complex-multiply (8-element sketch, matches v2/CCE/VMI f16):**

| Stage | Register | Logical contents |
|-------|----------|------------------|
| load | `x` | `[x0,x1,x2,x3,x4,x5,x6,x7]` |
| `vdintlv` | `x_even` | `[x0,x2,x4,x6]` |
| `vdintlv` | `x_odd` | `[x1,x3,x5,x7]` |
| negate odd | `neg_x_odd` | `[-x1,-x3,-x5,-x7]` |
| `vintlv` | `x_rot` | `[-x1,x0,-x3,x2,-x5,x4,-x7,x6]` |
| `vmul` + `vadd` | `y` | `[y0,y1,…,y7]` interleaved |

`cos` and `sin` remain **dense interleaved** throughout — unlike MI v1, which
splits them with separate `vdintlv` calls.

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

**Physical layout (f32 HALF):** the easy baseline — **64×f32 fits one register**
(256 B). Load, compute, store are all `NORM` with no UNPK, part, or PK:

```
x1: vreg<64×f32> ← x[0..31]     x2: vreg<64×f32> ← x[32..63]
        └── native fp32 vmul/vsub/vadd (mask<b32> for tail only)
```

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

**Physical layout (f32 INTERLEAVE Cartesian):**

```
x (64×f32 dense interleaved)
  │ channel_split  ≡  vdintlv
  ├─────────┬─────────
  ▼         ▼
x_even   x_odd     (+ same for cos, sin in MI v1; VMI splits x/cos/sin logically)
  │ 4 muls + 2 subs/adds on streams
  ▼
y_even, y_odd
  │ channel_merge  ≡  vintlv
  ▼
y dense interleaved
```

CCE f32 would instead keep cos/sin dense and use one `vdintlv`/`vintlv` pair on
`x` only (complex-multiply form, same as §5.4 index table).

---

### 5.7 End-to-end layout comparison (all six variants)

| Variant | Loads | Compute reg layout | Shuffle ops | Stores |
|---------|-------|-------------------|-------------|--------|
| f32 HALF | 2× dense 64×f32 | native dense | none | 2× dense |
| f16 HALF | 2× dense 128×f16 (32 active) | native dense | none | 2× dense |
| bf16 HALF | 4× UNPK + PART_EVEN widen | 4× 64×f32 | none | 2× PART_EVEN + PK_B32 |
| f16 INT v2 | 3× dense 128×f16 | dense + parity on `x` | `vdintlv`+`vintlv` on `x` | 1× dense |
| f16 INT v1 | 3× dense + 3× `vdintlv` | 6 parity streams | 3× `vdintlv`, 1× `vintlv` | 1× dense |
| bf16 INT | UNPK + fp32 body | fp32 dense + parity | same as f16 v2 + PK | PK_B32 |

**VMI value correlates with shuffle + UNPK/PK column width**, not with RoPE
formula complexity (which is identical everywhere).

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

**Review checklist for VMI RoPE code:**

1. Confirm the logical vector width matches the RoPE slice (`32` for HALF halves,
   `64` for the D=64 interleaved row in these examples).
2. Confirm dtype transitions match the intended numerics: bf16 paths widen to
   fp32 and narrow back; f16/f32 paths differ by variant.
3. Confirm `channel_merge` argument order builds the expected rotation
   (`neg_odd, even` for complex-multiply interleave).
4. Inspect lowered MI for the expected `UNPK_B16`/`PART_EVEN`/`PK_B32` chain on
   bf16 paths and `vdintlv`/`vintlv` on interleave paths.
5. Compare against `rope_cce_compute.h` when debugging: start at the
   `Expected UB effect` block, then the register-role comments inside the
   matching `ComputeF16`, `ComputeBf16`, or `ComputeF32` body.

**Suggested reading order for new algorithm engineers:**

1. Section 1 of this doc (math)
2. **§3.5–§3.10** (physical register model — read before MI/CCE examples)
3. `rope_bf16.vmi.pto` HALF loop body — best VMI showcase
4. **§5.7** (variant layout comparison table)
5. `rope_bf16.mi.pto` same section — see what VMI removed
6. `rope_cce_compute.h` `ComputeBf16` — ground truth with educational comments
7. `rope_f16_v2.mi.pto` — CCE-faithful MI reference for f16
8. `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md` — full layout taxonomy
9. `BLOCK_MX_QUANT_VMI_vs_MI.md` — same layout axes at 256-wide quant scale

---

## 8. One-page summary

| Question | Answer |
|----------|--------|
| Same RoPE semantics? | **Yes**, modulo bf16/fp32 inner precision choices and f32 interleave spelling variant |
| Cleaner syntax? | **Yes**, strongest for bf16, moderate for f16 interleave, modest for f32 HALF |
| Closer to math? | **Yes** where VMI hides UNPK/PK/part and uses `channel_split/merge` for GPT-J |
| Easier than CCE? | **Yes for compute bodies**; **tie** for DMA/tiling shell |
| Still worth it if MI is v2-quality? | **Yes for bf16**; **marginal for f32 HALF** |
| Why UNPK / vdintlv / vintlv? | Physical 256B regs + dense vs interleaved memory; see §3.6–§3.8 |

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

- `BLOCK_MX_QUANT_VMI_vs_MI.md` — same three-layer pattern; deeper DINTLV/`vintlv` at 256-wide
- `PTO-Gym-vmi/docs/PTO-vmi-design.en.md`
- `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
- `PTO-Gym-vmi/docs/PTO-micro-Instruction-SPEC.md`
