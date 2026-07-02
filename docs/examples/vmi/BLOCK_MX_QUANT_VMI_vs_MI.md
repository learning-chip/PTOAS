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

**Reader contract:** after reading this note, you should be able to connect the
scale/quant equations to the VMI source, predict the main MI/CCE lowering shape
(`DINTLV`, `part`, `vintlv`, `PK4_B32`, masks), and identify which lines are
algorithmic math versus register-layout repair.

**Concrete VF tile used by the simulator examples:** the correctness tests use
`rowNum=32`, `colBlockSize=256`, `ubBlockSize=32`, and
`vlForHalfNumber=128`. That means each input row is `256` bf16/f16 values
(`512` bytes), there are `8` scale groups per row (`256/32`), `scale1` occupies
`32*32=1024` bytes, `scale2` occupies `256` bytes, and reciprocal scale occupies
`16` uint16 lanes (`32` bytes). Wall-time configs also list
`(rowNum,colBlockSize)=(4,64),(16,128),(32,256)`, but the local vector pipeline
is written around the 256-lane row contract.

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

The VMI conversion ops intentionally mirror MLIR `arith` names, but operate on
abstract vector registers:

- `pto.vmi.extf` / `extsi` / `extui` mean per-lane extension while preserving
  logical index `i`.
- `pto.vmi.truncf` / `trunci` mean per-lane narrowing to the destination element
  type while preserving logical index `i`.

This is a semantic contract, not a one-instruction promise. In the f16→fp8
quant path, one `vmi.extf` over `vreg<256×f16>` lowers to split loads, four
`vcvt {part=EVEN/ODD}` conversions, and layout repair. One `vmi.truncf` plus
`masked_store` lowers to fp8 `vcvt` with rounding/saturation/part placement and
packed stores. VMI removes layout authorship; the lowered MI still performs the
necessary hardware work.

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

Instruction behavior and caveats:

- `vmi.load` names a logical row. For `vreg<256×f16>`, the lowered MI cannot fit
  the row into one 256-byte physical register, so it uses `DINTLV_B16` or an
  equivalent split layout.
- `vmi.broadcast` creates a logical vector constant. MI `vbr` broadcasts into one
  physical register width, which is why MI scale code often carries separate
  constants for each DINTLV half.
- `vmi.cmpi` + `vmi.select` are written as ordinary predicate/select data flow.
  MI uses `vcmp`/`vsel` with a concrete b16 mask, and CCE exposes the same
  predicate register handling.
- `group_reduce_maxi` and `group_broadcast` deliberately split the two meanings
  hidden inside `vcgmax`: reduction across lanes and replication of the result.
- `trunci` replaces `vpack LOWER` only at the VMI surface. Lowering still has to
  honor the source lane layout and destination byte layout.
- `masked_store` hides whether the MI store is `NORM_B8`, `NORM_B16`, or
  `PK4_B32`; those are layout decisions derived from the value type and assigned
  physical layout.

### 3.1 How to use the side-by-side snippets

When reading the scale and quant walkthroughs, tag each instruction by role:

| Role | Question to ask | MX examples |
|------|-----------------|-------------|
| **MATH** | Does this change the numeric value? | `max`, `select`, `mulf`, scale encode |
| **TYPE** | Does this change dtype while preserving logical lane `i`? | `extf`, `truncf`, `trunci`, `vcvt`, `vpack` |
| **LAYOUT** | Does this only move bits/lanes? | `DINTLV_B16`, `PART_EVEN/ODD`, `vintlv`, `PK4_B32`, `vselr` |
| **MEMORY** | Does this cross the UB/register boundary? | `load`, `store`, `masked_store`, `vlds`, `vsts` |

The VMI source tries to keep **MATH** and **TYPE** visible. MI/CCE show every
**LAYOUT** and **MEMORY** step because the author is working directly against
the 256-byte physical register model.

### 3.2 Expected lowering shapes for recurring VMI ops

These are review expectations for the concrete examples, not a promise that each
VMI op lowers to one MI instruction.

**256-lane f16 row load + widen in the quant path:**

```mlir
%x_f16 = pto.vmi.load %x_ub[%row_off] : ... -> !pto.vmi.vreg<256xf16>
%x_f32 = pto.vmi.extf %x_f16 : ... -> !pto.vmi.vreg<256xf32>
```

Expected MI/CCE shape:

```mlir
%x0, %x1 = pto.vldsx2 %xHalf[%row_off], "DINTLV_B16" : ...
%x0_even = pto.vcvt %x0, %mask16 {part = "EVEN"} : ... -> !pto.vreg<64xf32>
%x0_odd  = pto.vcvt %x0, %mask16 {part = "ODD"}  : ... -> !pto.vreg<64xf32>
// ... same for x1, then vintlv repair before contiguous fp8 conversion ...
```

`extf` means “produce f32 lane `i` for every logical `x[i]`.” The lowered MI
must split the 512-byte row, widen four physical streams, and repair layout.

**fp32 → fp8 narrow + store:**

```mlir
%y_fp8 = pto.vmi.truncf %scaled : ... -> !pto.vmi.vreg<256xf8E4M3FN>
pto.vmi.masked_store %y_fp8, %y_ub[%row_off], %mask : ...
```

Expected MI/CCE shape:

```mlir
%p0 = pto.vcvt %chunk0, %mask32 {part = "P0", rnd = "R", sat = "SAT"} : ...
%p0_u8 = pto.vbitcast %p0 : ...
pto.vsts %p0_u8, %y_ub[%row_off], %mask8 {dist = "PK4_B32"} : ...
// repeated for byte offsets 64, 128, 192
```

`truncf` preserves logical lane order and target FP8 semantics; `PK4_B32` is the
physical byte extraction needed to materialize dense UB bytes.

**Scale-byte narrowing:**

```mlir
%scale_u8 = pto.vmi.trunci %scale_u16 : ... -> !pto.vmi.vreg<256xui8>
```

Expected MI/CCE shape:

```mlir
%scale_u8 = pto.vpack %scale_u16, "LOWER" : !pto.vreg<128xi16> -> !pto.vreg<256xui8>
```

The VMI surface says “keep the low 8 bits per logical lane”; MI spells the
physical pack operation and later masks the first 8 meaningful scale bytes.

### 3.3 MI details algorithm engineers should not have to memorize

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

### 3.4 Common bugs VMI helps avoid

- **Dropping one layout repair step:** after `DINTLV_B16` and `PART_EVEN/ODD`,
  four f32 streams hold stride-4 logical indices. Skipping a `vintlv` produces a
  byte-exact but lane-permuted FP8 row.
- **Using the wrong `part` on FP8 cast:** the CCE/MI FP8 path writes into
  `PART_P0` before `PK4_B32`; another part changes which byte the store packs.
- **Wrong `PK4_B32` byte offset:** the four stores cover offsets
  `0,64,128,192`; a wrong offset aliases or gaps a quarter row.
- **Mask-family drift:** scale code mixes `b16` and `b8`; quant code mixes
  `b16`, `b32`, and `b8`. VMI keeps a logical active-lane mask and lowering picks
  the concrete predicate family.
- **Confusing `vcgmax` with ordinary max:** `vcgmax` is a grouped lane reduction
  with layout side effects. VMI names this as `group_reduce_maxi` plus
  `group_broadcast`, making the reduction boundary easier to review.

### 3.5 Physical register model (the 256-byte contract)

Every A5 vector register is **256 bytes (2048 bits)** wide. That single fact
drives almost all layout ceremony in block MX quant.

| Quantity | Bytes needed | Fits in one 256B reg? |
|----------|-------------|------------------------|
| 128 × f16 / bf16 | 256 B | **Yes** (128 b16 lanes) |
| 256 × f16 / bf16 | 512 B | **No** → need 2 regs or a split load |
| 64 × f32 | 256 B | **Yes** (64 b32 lanes) |
| 128 × f32 | 512 B | **No** → need 2 regs after widen |
| 256 × f32 | 1024 B | **No** → need 4 regs at peak |
| 256 × fp8 (packed in UB) | 256 B | **Yes**, but only after `PK4_B32` store packing |

**Logical view (VMI):** you hold `N` elements of type `T` in index order
`0, 1, 2, …, N−1`. The type system (`!pto.vmi.vreg<256×f16>`) promises that
contract regardless of how many physical registers exist underneath.

**Physical view (MI/CCE):** each instruction names **concrete registers** with
**lane-granular placement rules**. A lane is not “logical index `i`” until you
have explicitly loaded, split, widened, interleaved, and packed it into the
right slot. Layout ops (`DINTLV`, `part=`, `vintlv`, `PK4_B32`) exist only to
maintain or restore that index mapping.

See `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md`
§1–§2 for the full taxonomy. The MX quant examples exercise three layout axes:

| Axis | Mechanism | What it splits or merges |
|------|-----------|--------------------------|
| **Width** | `vcvt` + `part=EVEN/ODD` | One 128×f16 reg → two 64×f32 regs (2× widen) |
| **Parity** | `DINTLV_B16` load, `vintlv` | Even/odd **logical** indices into separate regs |
| **Sub-byte pack** | `vcvt {part=P0}` + `PK4_B32` | One f32 lane → one fp8 byte in memory |

VMI `extf` / `truncf` follow MLIR `arith` naming: they mean **“change element
type, preserve logical index `i`.”** They do **not** mean the hardware performs
one instruction. The compiler lowers them to the `vcvt` / `vintlv` / `vsts`
sequence appropriate for the assigned physical layout.

### 3.6 DINTLV load — first split (256 f16 → 2×128 f16)

UB memory holds one quant row as **256 contiguous f16**:

```
Memory (logical):  x[0] x[1] x[2] x[3] x[4] x[5] ... x[254] x[255]
                   └── 512 bytes ──┘
```

`DINTLV_B16` load (`vldsx2` / `vlds … DINTLV_B16`) cannot fit 512 B into one
256 B register, so it **deinterleaves by logical parity** at load time:

```
x0F16 (128 lanes, 256 B):  x[0]  x[2]  x[4]  x[6]  ... x[252] x[254]
                            phys lane k  ←→  logical index 2k

x1F16 (128 lanes, 256 B):  x[1]  x[3]  x[5]  x[7]  ... x[253] x[255]
                            phys lane k  ←→  logical index 2k+1
```

**Why load splits instead of loading 128 elements twice?** A normal `NORM` load
only fills one register. To process 256 elements without scalar loops, the
hardware exposes **split-load distribution modes** that populate two registers in
one memory transaction while respecting the 256 B cap.

From this point on, MI/CCE authors must track **two ownership domains**:
everything in `x0F16` is an even logical index; everything in `x1F16` is odd.

VMI `vmi.load` returns `vreg<256×f16>` — the split still happens in lowered
MI, but the VMI type + layout metadata record the inverse map
(`even_reg[k]→2k`, `odd_reg[k]→2k+1`).

### 3.7 PART_EVEN / PART_ODD widen — second split (each 128×f16 → 2×64×f32)

Widen f16→f32 doubles element width: 128 f16 (256 B) → 128 f32 (512 B). The
hardware writes widened results into **two 64×f32 registers** using part
selection on **physical lane parity inside each source reg**:

```
x0F16 phys lanes:     0    1    2    3    4  ...  126  127
logical index held:     0    2    4    6    8  ...  252  254

vcvt {part=EVEN}  →  x0Zero0 (64×f32):  f32(0),  f32(4),  f32(8),  ... f32(252)
vcvt {part=ODD}   →  x0One0  (64×f32):  f32(2),  f32(6),  f32(10), ... f32(254)
```

The same pattern on `x1F16`:

```
vcvt {part=EVEN}  →  x1Zero0:  f32(1),  f32(5),  f32(9),  ... f32(253)
vcvt {part=ODD}   →  x1One0:   f32(3),  f32(7),  f32(11), ... f32(255)
```

After four `vcvt`s you have **four 64×f32 streams**, each holding values at
**logical stride 4**. This is the **parity-interleaved** layout: arithmetic
(`vmul`) can run independently on each stream, but the results are **not** in
memory order `[0,1,2,3,…]`.

**What `vmi.extf` hides:** one semantic “256 f16 → 256 f32” op. Lowering emits
four `vcvt {part=EVEN/ODD}` plus (depending on the next op) relayout. The
author never names `x0Zero0` or chooses which part applies to which DINTLV half.

### 3.8 `vintlv` — rebuilding logical order (four steps on 256 elements)

`vintlv(src0, src1)` interleaves **lane-by-lane** from two source registers into
two destination registers. Within a 128-element local window it turns
step-2 parity streams back into contiguous order (see Pack-Unpack reference
§2.3.1).

The MI/CCE quant loop uses **four** `vintlv` calls in two stages:

**Stage A — repair widen split inside each DINTLV half (2× `vintlv`):**

```
vintlv(x0Zero1, x0One1)   // after vmul on x0 streams
  IN:  f32(0),f32(4),f32(8),...     +  f32(2),f32(6),f32(10),...
  OUT: x0Zero2 = f32(0),f32(2),f32(4),...f32(126)   [64 lanes, stride-2 evens]
       x0One2  = f32(128),f32(130),...f32(254)      [64 lanes, stride-2 evens, high half]

vintlv(x1Zero1, x1One1)   // same for odd-index load half
  OUT: x1Zero2 = f32(1),f32(3),f32(5),...f32(125)
       x1One2  = f32(129),f32(131),...f32(255)
```

**Stage B — repair DINTLV load split across halves (2× `vintlv`):**

```
vintlv(x0Zero2, x1Zero2)
  OUT: x0Zero = f32(0), f32(1), f32(2), ... f32(127)    ← logical [0..127] contiguous
       x1Zero = (second 64-lane chunk of interleave; paired with x0Zero for FP8)

vintlv(x0One2, x1One2)
  OUT: x0One  = f32(128), f32(129), ... f32(255)        ← logical [128..255] contiguous
       x1One  = (companion chunk)
```

CCE documents the post-cross-interleave contract explicitly:

```c
// x0ZeroFP32 + x1ZeroFP32 merge -> first 128 FP32 values (positions 0-127)
// x0OneFP32  + x1OneFP32  merge -> last 128 FP32 values (positions 128-255)
vintlv(x0ZeroFP32, x1ZeroFP32, x0ZeroFP32, x1ZeroFP32);
vintlv(x0OneFP32,  x1OneFP32,  x0OneFP32,  x1OneFP32);
```

**None of the four `vintlv` ops change numeric values** — they only permute which
physical lane holds which logical index so the subsequent FP8 path sees
contiguous 64×f32 chunks.

**Alternative (not used in this example):** keep parity-interleaved f32 through
multiply and repair at store time with `vstsx2 … INTLV_B32` instead of four
`vintlv` (Pack-Unpack reference §4.2). The MI demo chooses **repair-then-`PART_P0`**
because it matches the CCE `ComputeY1ToFP8` FP16 branch structure.

### 3.9 PART_P0 + PK4_B32 — FP8 narrow and store (third layout axis)

After the `vintlv` chain, each 64×f32 register holds **64 contiguous logical
values**. FP8 output still needs two more physical transforms:

**Narrow (`vcvt {part=P0, rnd=R, sat=SAT}`):** each 32-bit f32 lane group
receives one 8-bit fp8 value in the **P0 byte** of the lane; P1–P3 bytes zeroed
(Part_T axis, 4-way sub-part placement):

```
f32 lane k (32 bits):  [ f32 val ]  →  after P0:  [ fp8(k) | 00 | 00 | 00 ]
```

**Store (`vsts {dist=PK4_B32}`):** extract the low 8 bits of each 32-bit lane
and pack four fp8 bytes per 32-bit memory word → **64 fp8 bytes per store**.

The quant loop issues **four** `vcvt` + **four** `vsts PK4_B32` at offsets
`+0, +64, +128, +192` bytes — one 64-byte chunk per 64 fp8 values, 256 fp8
total per row. CCE wiring (note the cross-register sources after interleave):

```c
vcvt(x0ZeroFP8, x0ZeroFP32, pregB32, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
vcvt(x0OneFP8,  x1ZeroFP32, pregB32, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
vcvt(x1ZeroFP8, x0OneFP32,  pregB32, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
vcvt(x1OneFP8,  x1OneFP32,  pregB32, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
vsts(..., PK4_B32, pregB8);  // ×4 with POST_UPDATE / offset bumps
```

**What `vmi.truncf` + `vmi.masked_store` hide:** rounding mode, saturation,
part=P0 placement, `ui8` bitcast, PK4 lane extraction, four store addresses, and
`b8`/`b32` predicate splitting — all derived from the logical
`vreg<256×f8E4M3>` type and target memory layout.

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

Instruction-by-instruction:

- `vmi.load` reads the row as `vreg<256×ui16>` so the exponent extraction can be
  written once over logical lanes. MI starts with `vldsx2 DINTLV_B16`, producing
  even and odd ownership domains.
- `vmi.bitcast` changes interpretation from unsigned bits to signed integer
  lanes without moving data, matching MI `vbitcast`.
- `vmi.andi` masks with `0x7F80`, keeping bf16 exponent bits `[14:7]`. MI/CCE
  spell this as `vand` plus an all-lanes b16 predicate.
- `vmi.cmpi "slt"` produces the predicate `acc < xExp`, and `vmi.select` chooses
  the new maximum. MI/CCE use `vmax`, which is shorter but less explicit about
  the comparison/select relation.

**Physical layout note:** the scale path never widens to f32 per element, so it
does **not** need the four-quadrant `vintlv` chain from §3.8. It still pays the
**DINTLV load tax**: `acc0` owns exponent maxima for logical indices
`0,2,4,…,254` and `acc1` for `1,3,5,…,255`. `vcgmax` fuses reduction with
broadcast only **within each mask family** (`b16`), so the dual-accumulator
pattern persists until VMI's `group_reduce_maxi` collapses both halves under one
logical `vreg<256×…>` contract.

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

Each line corresponds directly to the row formula `y_i = fp8(x_i * scale_i)`:

- `vmi.load` establishes the logical index contract `[0..255]`.
- `extf` widens each f16 lane to f32; in MI this is four `vcvt` ops because
  `DINTLV_B16` and width expansion create four physical streams.
- `mulf` is the only numerical arithmetic in this stage.
- `truncf` narrows to FP8 E4M3 with target rounding/saturation rules. MI must
  spell `rnd=R`, `sat=SAT`, and part placement explicitly.
- `masked_store` writes the packed fp8 row. MI needs four `PK4_B32` stores at
  byte offsets `0, 64, 128, 192`.

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

### 6.5 End-to-end physical register pipeline (256 f16 row)

The table below tracks **which logical indices** each named register holds after
each MI/CCE step. This is the ground truth for debugging lane swaps.

| Step | Instruction(s) | Register | Logical indices held (64 lanes shown as range) |
|------|----------------|----------|--------------------------------------------------|
| 0 | *(UB memory)* | — | `[0..255]` contiguous f16 |
| 1 | `vldsx2 DINTLV_B16` | `x0F16` | `[0,2,4,…,254]` — 128 lanes |
| 1 | | `x1F16` | `[1,3,5,…,255]` — 128 lanes |
| 2a | `vcvt EVEN` | `x0Zero0` | `[0,4,8,…,252]` stride 4 |
| 2b | `vcvt ODD` | `x0One0` | `[2,6,10,…,254]` stride 4 |
| 2c | `vcvt EVEN` | `x1Zero0` | `[1,5,9,…,253]` stride 4 |
| 2d | `vcvt ODD` | `x1One0` | `[3,7,11,…,255]` stride 4 |
| 3 | `vmul` ×4 | same 4 regs | same indices, scaled f32 values |
| 4a | `vintlv` (intra x0) | `x0Zero2` | `[0,2,4,…,126]` stride 2 |
| 4a | | `x0One2` | `[128,130,…,254]` stride 2 |
| 4b | `vintlv` (intra x1) | `x1Zero2` | `[1,3,5,…,125]` stride 2 |
| 4b | | `x1One2` | `[129,131,…,255]` stride 2 |
| 5a | `vintlv` (cross, low) | `x0Zero` | **`[0..127]` contiguous** |
| 5a | | `x1Zero` | companion 64-lane chunk for FP8 path |
| 5b | `vintlv` (cross, high) | `x0One` | **`[128..255]` contiguous** |
| 5b | | `x1One` | companion chunk |
| 6 | `vcvt P0` ×4 | 4× fp8 regs | 64 fp8 each, P0-packed in 256B reg |
| 7 | `vsts PK4_B32` ×4 | UB | `[0..255]` contiguous fp8 bytes |

Steps 1–5 are **pure layout**; step 3 is the only **algorithm** (`× scale`);
steps 6–7 are **typed narrow + memory pack**.

### 6.6 Why VMI needs only `extf` and `truncf`

VMI and MI implement the **same row math**. The difference is **where** layout
is expressed:

| Concern | VMI surface | MI/CCE surface |
|---------|-------------|----------------|
| Index contract | `vreg<256×T>` — lane `i` is element `i` | Per-register index map (table above) |
| Widen f16→f32 | `vmi.extf` | 4× `vcvt {part=EVEN/ODD}` |
| Repair parity + DINTLV | *(compiler)* | 4× `vintlv` |
| Narrow f32→fp8 | `vmi.truncf` | 4× `vcvt {part=P0,rnd,sat}` + bitcast |
| Write fp8 row | `vmi.masked_store` | 4× `vsts {dist=PK4_B32}` |

`extf` is **not** a single hardware widen. It is a **semantic extension** (like
MLIR `arith.extf`): “for all `i ∈ [0,255]`, produce `f32(x[i])`.” The VMI
compiler already knows the value arriving from `vmi.load` was DINTLV-split; it
inserts the four `vcvt`s and, if the next op requires contiguous f32 (`vmi.mulf`
on a dense 256×f32 scale vector), schedules the four `vintlv`s **between**
widen and multiply in lowered MI.

Similarly, `truncf` is **not** one `vcvt`. It means “for all `i`, produce
`fp8_e4m3(f32[i])` with target-default rounding and saturation.” Lowering must
place each result into P0 slots and emit PK4 stores — work that MI authors write
by hand.

**Honest boundary:** VMI removes layout **authorship**, not layout **work**.
Cycle count after lowering is in the same ballpark as a well-written MI kernel;
the win is correctness and reviewability when logical width exceeds one physical
register.

### 6.7 Register diagram — MI path vs VMI logical view

**VMI (one logical vector throughout compute):**

```
UB f16 [0..255]
        │ vmi.load
        ▼
   vreg<256×f16>  logical [0..255]
        │ vmi.extf                    ┐ compiler lowers to:
        ▼                             │  4× vcvt EVEN/ODD + 4× vintlv
   vreg<256×f32>  logical [0..255]    ┘
        │ vmi.mulf (× scale)
        ▼
   vreg<256×f32>  logical [0..255]
        │ vmi.truncf                  ┐ compiler lowers to:
        ▼                             │  4× vcvt P0 + 4× vsts PK4_B32
   vreg<256×fp8>  logical [0..255]    ┘
        │ vmi.masked_store
        ▼
UB fp8 [0..255]
```

**MI/CCE (four physical registers at peak, explicit repair):**

```
UB f16 [0..255]
        │ DINTLV_B16
        ├──────────────────┬──────────────────
        ▼                  ▼
   x0F16 [0,2,4..]    x1F16 [1,3,5..]     ← 2 regs (parity axis)
        │ EVEN/ODD           │ EVEN/ODD
        ├────┬────           ├────┬────
        ▼    ▼               ▼    ▼
     x0Z0 x0O0            x1Z0 x1O0         ← 4 regs (stride-4)
        │ vmul×4
        │ vintlv×2 (intra)   │ vintlv×2
        ├────┬────           ├────┬────
        ▼    ▼               ▼    ▼
     x0Z2 x0O2            x1Z2 x1O2         ← stride-2 within each half
        │ vintlv×2 (cross low)  │ vintlv×2 (cross high)
        ▼                     ▼
     x0Zero [0..127]      x0One [128..255]  ← contiguous f32 (algorithm done)
        │ vcvt P0 + PK4         │ vcvt P0 + PK4 (×4 total)
        ▼                     ▼
UB fp8 [0..255]
```

### 6.8 Scale vector layout (reciprocal load, same axes)

The quant loop also loads a **bf16 reciprocal scale** vector. MI uses `E2B_B16`
broadcast distribution + `vbitcast` + `vcvt {part=EVEN}` to obtain a 64×f32
register usable by all four `vmul`s:

```
E2B_B16 load  →  128×ui16 in broadcast layout (valid bf16 in EVEN physical lanes)
vbitcast      →  128×bf16
vcvt EVEN     →  64×f32 scale (one value per 32-column block, broadcast within lanes)
```

VMI loads `vreg<256×bf16>` and `vmi.extf` to `vreg<256×f32>`. The compiler
chooses E2B vs dense load based on how the scale sits in UB; the author only
writes “widen scale to f32 for multiply.”

When scale lanes are uniform after block reduction (as in the full VMI kernel),
`vmi.mulf` is a true 256-lane semantic multiply. MI reuses one 64×f32 scale reg
for four stride-4 streams because **the scale is constant across the four
parity streams** — but the author must still bind `mask<b32>` on each `vmul`.

### 6.9 MI loop body inventory (17 core ops + store pack)

The MI example comments cite **17 instructions** in the loop body, counting
main vector ops (`vldsx2`, `vcvt`, `vmul`, `vintlv`) and the four FP8 `vcvt`s,
but **not** the four `vbitcast` + four `vsts PK4_B32` pairs (those add 8 more
lines for store typing and pack mode).

| # | MI op | Layout or math? | Physical effect |
|---|-------|-------------------|-----------------|
| 1 | `vldsx2 DINTLV_B16` | Layout | Split 256 f16 → 2×128 parity regs |
| 2–5 | `vcvt EVEN/ODD` ×2 + `vmul` ×2 on x0 | Widen + **math** | x0 → 2×64 f32 stride-4, scaled |
| 6 | `vintlv` (intra x0) | Layout | x0 stride-4 → stride-2 within even half |
| 7–10 | `vcvt EVEN/ODD` ×2 + `vmul` ×2 on x1 | Widen + **math** | x1 → 2×64 f32 stride-4, scaled |
| 11 | `vintlv` (intra x1) | Layout | x1 stride-4 → stride-2 within odd half |
| 12–13 | `vintlv` ×2 (cross) | Layout | Merge halves → `[0..127]`, `[128..255]` f32 |
| 14–17 | `vcvt {P0,rnd,sat}` ×4 | Narrow (**math** + Part_T) | 64 f32 → P0-packed fp8 reg each |
| +8 | `vbitcast` + `vsts PK4_B32` ×4 | Layout | Extract P0 bytes → contiguous UB fp8 |

VMI collapses rows 1–6 and 7–13 into `load` + `extf`, rows 14–17 and the +8
store pack into `truncf` + `masked_store`, leaving one `mulf` as the only
visible arithmetic.

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

**Review checklist for VMI MX quant code:**

1. Confirm the logical vector width matches the row contract (`256` lanes for
   the covered scale and quant bodies).
2. Confirm scale math still spells special cases explicitly: Inf/NaN, zero,
   clamp/range handling, and reciprocal edge cases.
3. Confirm dtype transitions match the intended stage: `extf` before fp32
   multiply, `truncf` for FP8/FP4 output, and `trunci` for compact scale bytes.
4. Inspect lowered MI for the expected `DINTLV_B16` split, `PART_EVEN/ODD`
   widen, `vintlv` repair, `PART_P0` FP8 cast, and four `PK4_B32` stores on the
   f16→fp8 path.
5. Compare against `bmx_cce_kernels.h` when debugging: start at the
   `Expected UB effect` block, then the register allocation and phase comments
   for `ComputeOcp`, `ComputeDdr`, `ComputeY1ToFP4`, or `ComputeY1ToFP8`.

**Suggested reading order:**

1. Section 1 of this doc (math)
2. **§3.5–§3.9** (physical register model — read before MI/CCE examples)
3. `mx_block_quant_y1_fp8_f16_e4m3.vmi.pto` — best “before/after” showcase
4. **§6.5–§6.9** (quant pipeline index map + VMI vs MI diagrams)
5. `mx_block_quant_y1_fp8_f16_e4m3.mi.pto` — see what VMI removed
6. `mx_block_quant_scale_ocp_bf16.vmi.pto` vs `.mi.pto` — scale path
7. `bmx_cce_kernels.h` `ComputeOcp` / `ComputeY1ToFP8` — annotated ground truth
8. `PTO-Gym-vmi/docs/PTO-micro-ISA-Pack-Unpack-Interleave-Part-Reference.md` — full layout taxonomy

---

## 9. One-page summary

| Question | Answer |
|----------|--------|
| Same MX semantics? | **Yes**, for the covered bf16 scale + f16→E4M3 quant paths |
| Cleaner syntax? | **Yes** — largest on quant loop bodies, large on scale compute |
| Closer to math? | **Yes** — VMI keeps “max exponent → scale → multiply → cast” visible |
| Easier than CCE? | **Yes for compute bodies**; **tie** for DMA/tiling |
| Biggest win? | **f16→fp8 quant**, where MI spends most lines on layout repair |
| Why `extf`/`truncf` vs `vintlv`? | VMI ops preserve **logical index `i`**; MI `vintlv` repairs **physical lane placement** after DINTLV + PART widen (§3.6–§3.9, §6.5–§6.7) |

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
