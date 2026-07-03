# RoPE on PTO: VMI vs MI vs CCE

This walkthrough explains the RoPE examples in this directory from the point of
view of an algorithm engineer: keep the rotation math visible, but understand
where the hardware/register details still matter.

| Layer | Example files | Use it for |
|-------|---------------|------------|
| **CCE** | `rope_cce_compute.h` | Intrinsics ground truth and expected UB behavior |
| **MI** (`pto.mi`) | `rope_{f16,bf16,f32}.mi.pto`, `rope_f16_v2.mi.pto` | Hardware-faithful PTO micro-ops |
| **VMI** (`pto.vmi`) | `rope_{f16,bf16,f32}.vmi.pto` | Logical-vector authoring |

**How to use this note:** read it as a bridge between the RoPE equations and the
actual VMI/MI/CCE code. The goal is to make it clear what each VMI instruction
means, what MI/CCE pattern it usually lowers to, and which low-level details are
worth checking when lanes or stores look wrong.

**Concrete tile in the VF simulator:** correctness tests use `sCount=15`,
`nCount=32`, `dLen=dAlign=64`; `xNStep=yNStep=csSStep=64`, and
`xSStep=ySStep=nCount*64=2048` elements. Wall-time sweeps also use
`(s,n)=(1,2),(15,4),(15,8),(15,16),(15,32)`.

VMI is not a shortcut around the hardware. It keeps layout bookkeeping out of
the source you write; it does not remove loop structure, UB strides, DMA,
pipeline flags, or the need to inspect lowered MI for peak tuning.

---

## 1. RoPE Math

RoPE rotates pairs of dimensions by a position-dependent angle using precomputed
`cos` and `sin` tables.

### HALF Mode

The head dimension `D` is split into two contiguous halves. For `d < D/2`:

```text
y[d]       = x[d]       * cos[d]       - x[d + D/2] * sin[d]
y[d+D/2]   = x[d+D/2]   * cos[d+D/2] + x[d]       * sin[d+D/2]
```

In the demo tile, `D=64`, so each partner half has `32` elements.

### INTERLEAVE Mode

Adjacent elements form rotation pairs: `(x[2k], x[2k+1])`.

The CCE path, `rope_f16_v2.mi.pto`, and f16/bf16 VMI examples use the
complex-multiply spelling:

```text
y = x * cos + (i*x) * sin
```

where `(i*x)` in real interleaved layout is:

```text
[-x1, x0, -x3, x2, -x5, x4, ...]
```

Some examples use the equivalent Cartesian spelling:

```text
y_even = x_even * cos_even - x_odd * sin_even
y_odd  = x_odd  * cos_odd  + x_even * sin_odd
```

Both spellings produce the same RoPE result. They differ in which layout
operations are needed around the math.

### bf16 Numerics

For bf16 RoPE, `x` and `y` are bf16, `cos` and `sin` are fp16, and the inner
arithmetic is fp32 before narrowing back to bf16. This matches
`ComputeBf16` in the CCE reference.

---

## 2. What VMI Changes

VMI (`pto.vmi`) lets the author write logical vectors such as
`!pto.vmi.vreg<64xbf16>` and semantic operations such as `pto.vmi.vload`,
`pto.vmi.vcvt`, `pto.vmi.vmul`, `pto.vmi.vdintlv`, and `pto.vmi.vstore`.

The compiler lowers that logical view to MI instructions with concrete:

- load/store distributions such as `UNPK_B16`, `PK_B32`, and `NORM`
- conversion parts such as `PART_EVEN`
- mask families such as `b16` and `b32`
- shuffle operations such as `vdintlv` and `vintlv`

The conversion names intentionally mirror MLIR `arith` vocabulary:

- `pto.vmi.vcvt`: convert each logical lane to the destination type

These names describe lane-wise meaning; they are not one-instruction promises. For example, bf16
`vcvt` often lowers to an unpacked load plus `vcvt PART_EVEN`; bf16 `vcvt`
plus `vstore` lowers to a narrow plus `PK_B32` store.

---

## 3. How to Read the Examples

Tag each instruction by role:

| Role | Question | RoPE examples |
|------|----------|---------------|
| **MATH** | Does it change the numeric value? | `vmul`, `vadd`, `vsub`, `vneg` |
| **TYPE** | Does it change dtype while preserving lane `i`? | `vcvt` |
| **LAYOUT** | Does it only move lanes or repair physical placement? | `UNPK_B16`, `PK_B32`, `PART_EVEN`, `vdintlv`, `vintlv` |
| **MEMORY** | Does it cross the UB/register boundary? | `vload`, `vstore`, `vlds`, `vsts` |

VMI source mostly shows **MATH** and **TYPE**. MI/CCE show the **LAYOUT** and
**MEMORY** work because the author is addressing physical vector registers.

### Instruction Map

| Math intent | CCE / MI shape | VMI shape |
|-------------|----------------|-----------|
| Load a logical vector | `pto.mi.vlds NORM` or `pto.mi.vlds UNPK_B16` | `pto.vmi.vload` |
| Widen bf16/f16 to fp32 | `pto.mi.vcvt {part=EVEN/ODD}` | `pto.vmi.vcvt` |
| Narrow fp32 to bf16/f16 | `pto.mi.vcvt {rnd, sat, part}` | `pto.vmi.vcvt` |
| Multiply / add / subtract | `pto.mi.vmul` / `pto.mi.vadd` / `pto.mi.vsub` + concrete mask | `pto.vmi.vmul` / `vadd` / `vsub` |
| Split adjacent pairs | `pto.mi.vdintlv` | `pto.vmi.vdintlv` |
| Merge even/odd streams | `pto.mi.vintlv` | `pto.vmi.vintlv` |
| Store active lanes | `pto.mi.vsts NORM_*` or `PK_B32` + mask | `pto.vmi.vstore` |

### Expected Lowering Shapes

These are review expectations for the examples, not formal lowering rules.

**bf16 load + widen in HALF mode:**

```mlir
%x16 = pto.vmi.vload %x_ub[%off] : ... -> !pto.vmi.vreg<64xbf16>
%x32 = pto.vmi.vcvt %x16 : ... -> !pto.vmi.vreg<64xf32>
```

Expected MI/CCE shape:

```mlir
%x16_phys = pto.mi.vlds %x_ub[%off], %mask16 {dist = "UNPK_B16"} : ...
%x32 = pto.mi.vcvt %x16_phys, %mask16 {part = "EVEN"} : ... -> !pto.mi.vreg<64xf32>
```

`PART_EVEN` is a correctness requirement because `UNPK_B16` places dense bf16
memory values into even physical lanes.

**bf16 narrow + store:**

```mlir
%y16 = pto.vmi.vcvt %y32 : ... -> !pto.vmi.vreg<64xbf16>
pto.vmi.vstore %y16, %y_ub[%off], %mask : ...
```

Expected MI/CCE shape:

```mlir
%y16_phys = pto.mi.vcvt %y32, %mask32 {part = "EVEN", rnd = "R", sat = "SAT"} : ...
pto.mi.vsts %y16_phys, %y_ub[%off], %mask32 {dist = "PK_B32"} : ...
```

`vcvt` preserves logical lane order; `PK_B32` repairs the physical even-lane
layout before the values are written densely to UB.

**INTERLEAVE rotation helper:**

```mlir
%even, %odd = "pto.vmi.vdintlv"(%x) : ...
%neg_odd = pto.vmi.vneg %odd : ...
%rot = "pto.vmi.vintlv"(%neg_odd, %even) : ...
```

Expected MI/CCE shape:

```mlir
%even, %odd = pto.mi.vdintlv %x, %x : ...
%neg_odd = pto.mi.vmul %odd, %minus_one, %mask : ...
%rot, %rot_hi = pto.mi.vintlv %neg_odd, %even : ...
```

Argument order matters: `merge(neg_odd, even)` builds
`[-x1, x0, -x3, x2, ...]`.

---

## 4. Running Example: bf16 HALF

This is the best RoPE showcase because the algorithm is small while the MI
layout protocol is long.

Math:

```text
y1 = x1 * cos1 - x2 * sin1
y2 = x2 * cos2 + x1 * sin2
```

VMI source shape:

```mlir
%cos1_16 = pto.vmi.vload %cos_ub[%cos1_off] : ... -> !pto.vmi.vreg<64xf16>
%cos1 = pto.vmi.vcvt %cos1_16 : ... -> !pto.vmi.vreg<64xf32>
%x1 = pto.vmi.vcvt %x1_16 : ... -> !pto.vmi.vreg<64xf32>
%x1_cos = pto.vmi.vmul %x1, %cos1 : ...
%x2_sin = pto.vmi.vmul %x2, %sin1 : ...
%out1_f32 = pto.vmi.vsub %x1_cos, %x2_sin : ...
%out1 = pto.vmi.vcvt %out1_f32 : ... -> !pto.vmi.vreg<64xbf16>
pto.vmi.vstore %out1, %y_ub[%y1_off], %mask : ...
```

MI/CCE must expose:

- `UNPK_B16` load for dense halfword data
- `PART_EVEN` widen to fp32
- `mask<b16>` for load/convert and `mask<b32>` for fp32 arithmetic
- `PART_EVEN` narrow
- `PK_B32` store back to dense bf16 UB

Physical pipeline for one 32-element partner half:

```text
UB dense bf16
  -> UNPK_B16:       [x0, __, x1, __, ...]
  -> vcvt EVEN:      64xf32 logical values
  -> vmul/sub/add:   fp32 RoPE math
  -> vcvt EVEN:      [y0, __, y1, __, ...]
  -> PK_B32 store:   UB dense bf16
```

VMI collapses the layout protocol into `vload`, `vcvt`, and `vstore`, so the
source reads as dtype-aware RoPE math.

---

## 5. Running Example: INTERLEAVE

For GPT-J layout, the useful mental model is:

```text
x          = [x0, x1, x2, x3, ...]
vdintlv(x) -> even=[x0,x2,...], odd=[x1,x3,...]
vintlv(-odd, even) -> [-x1,x0,-x3,x2,...] = i*x
y = x*cos + (i*x)*sin
```

VMI complex-multiply shape:

```mlir
%x_even, %x_odd = "pto.vmi.vdintlv"(%x) : ...
%neg_x_odd = pto.vmi.vneg %x_odd : ...
%rot = "pto.vmi.vintlv"(%neg_x_odd, %x_even) : ...
%x_cos = pto.vmi.vmul %x, %cos : ...
%rot_sin = pto.vmi.vmul %rot, %sin : ...
%y = pto.vmi.vadd %x_cos, %rot_sin : ...
```

The CCE and `rope_f16_v2.mi.pto` spelling is structurally the same but uses
`vdintlv`, `vbr(-1)`, `vmul`, and `vintlv`.

The Cartesian spelling splits `x`, `cos`, and `sin`, computes `y_even` and
`y_odd`, and merges them back. It is equally valid, but it has more explicit
shuffle and arithmetic lines. The f32 VMI interleave example uses this spelling,
so it is not line-for-line identical to CCE even though the math is equivalent.

---

## 6. Variant Map

| Variant | VMI advantage | Main hardware detail hidden |
|---------|---------------|-----------------------------|
| **bf16 HALF** | Very large | UNPK → EVEN → fp32 compute → EVEN → PK, plus dual masks |
| **bf16 INTERLEAVE** | Very large | bf16 conversion chain plus parity split/merge |
| **f16 INTERLEAVE** | Large | `vdintlv`/`vintlv` and explicit negate stream |
| **f32 INTERLEAVE** | Moderate | semantic `vdintlv/vintlv`, fewer mask arguments |
| **f16 HALF** | Moderate | mask plumbing and 128-lane physical register width |
| **f32 HALF** | Small | MI is already close to math; VMI mostly removes masks |

Takeaway: VMI helps most when the value is far from “dense fp32 load, compute,
store.” Mixed precision and interleaved layouts are the headline cases.

---

## 7. Debug Appendix: Physical Rules

Use this section when reviewing lowered MI or comparing against CCE comments.

### 256-Byte Register Model

Every A5 vector register is 256 bytes:

| Element type | Physical lanes per register |
|--------------|-----------------------------|
| f16 / bf16 / ui16 | 128 lanes |
| f32 | 64 lanes |
| ui8 / fp8 | 256 lanes |

RoPE uses `D=64`, so f32 fits exactly in one register. f16/bf16 use only part of
a 128-lane b16 register, and bf16 widening/narrowing introduces even-lane
placement.

### UNPK_B16 and PK_B32

`UNPK_B16` expands dense halfword memory into even physical lanes:

```text
memory:       [b0, b1, b2, ...]
physical reg: [b0, __, b1, __, b2, __, ...]
```

`vcvt PART_EVEN` reads those valid lanes into fp32. After fp32 compute,
`vcvt PART_EVEN` writes bf16 results back into even lanes, and `PK_B32` stores
them densely:

```text
physical reg: [y0, __, y1, __, y2, __, ...]
memory:       [y0, y1, y2, ...]
```

### Parity Split and Merge

`vdintlv` and `vintlv` implement the parity axis:

```text
[x0,x1,x2,x3,...] --vdintlv--> even=[x0,x2,...], odd=[x1,x3,...]
even/odd --vintlv--> [even0,odd0,even1,odd1,...]
```

VMI names these as `vdintlv` and `vintlv`.

### Mask Families

MI RoPE often carries both:

- `!pto.mi.mask<b16>` for b16 load/convert/f16 arithmetic
- `!pto.mi.mask<b32>` for fp32 arithmetic and bf16 narrow/store paths

Using the wrong mask family is a wrong-lane bug. VMI uses a logical
`!pto.vmi.mask<N×pred>` and lets lowering choose the concrete predicate family.

### Common Bugs VMI Helps Avoid

- using `PART_ODD` after `UNPK_B16`
- using `NORM_B16` instead of `PK_B32` after bf16 narrow
- swapping `vintlv(neg_odd, even)` to `vintlv(even, neg_odd)`
- mixing b16 and b32 predicates in bf16 paths

---

## 8. Practical Guidance

| Task | Prefer |
|------|--------|
| Author or review RoPE math, especially bf16 | **VMI** |
| Verify against intrinsics behavior | **CCE** or lowered **MI** |
| Debug wrong lanes, padding, or stores | Lowered **MI** plus CCE comments |
| Tune final schedules | Lowered **MI** / hardware traces |

Review checklist for VMI RoPE:

1. Logical vector width matches the slice (`32` for HALF halves, `64` for D=64
   interleaved rows).
2. bf16 paths widen to fp32 and narrow back to bf16.
3. INTERLEAVE complex multiply uses `vintlv(neg_odd, even)`.
4. Lowered MI contains the expected `UNPK_B16`/`PART_EVEN`/`PK_B32` sequence for
   bf16 and `vdintlv`/`vintlv` for interleave.
5. For CCE comparison, start with the `Expected UB effect` block and register
   role comments in the matching `ComputeF16`, `ComputeBf16`, or `ComputeF32`
   body.

Suggested reading order:

1. Section 1 of this doc for math.
2. `rope_bf16.vmi.pto` HALF loop body.
3. Section 4 here for the VMI/MI lowering shape.
4. `rope_bf16.mi.pto` same loop body.
5. `rope_cce_compute.h` `ComputeBf16`.
6. Section 7 here when debugging physical lanes.
7. `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`.

---

## Example File Index

| File | Layer | Notes |
|------|-------|-------|
| `rope_bf16.vmi.pto` | VMI | Best RoPE VMI demo: bf16 with fp32 inner math |
| `rope_bf16.mi.pto` | MI | Full UNPK/EVEN/PK exposure |
| `rope_f16.vmi.pto` | VMI | f16 HALF plus INTERLEAVE complex multiply |
| `rope_f16.mi.pto` | MI | f16 Cartesian interleave variant |
| `rope_f16_v2.mi.pto` | MI | CCE-faithful f16 complex-multiply variant |
| `rope_f32.vmi.pto` | VMI | f32 HALF and Cartesian interleave |
| `rope_f32.mi.pto` | MI | f32 masks and dense loads/stores |
| `rope_cce_compute.h` | CCE | Intrinsics ground truth |

Related docs:

- `BLOCK_MX_QUANT_VMI_vs_MI.md`
- `PTO-Gym-vmi/docs/PTO-vmi-design.en.md`
- `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
- `PTO-Gym-vmi/docs/PTO-micro-Instruction-SPEC.md`
