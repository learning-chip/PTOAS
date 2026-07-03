"""
* Ops that *materialize* a new shape/dtype (``load``, ``create_mask``) take it
  as a parametric subscript — ``vmi.load[vmi.Vreg[64, bf16]](p)`` — so the width
  is a real argument the builder receives, not a PEP 526 LHS annotation (which
  is not passed to the RHS call at runtime). This mirrors the Mojo ``load[64]``.
* Ops whose result shape/dtype is derivable from operands keep a PEP 526
  annotation as documentation (``out1_f32: vmi.Vreg[64, f32] = x1 * cos1 - ...``),
  written *once* rather than twice as in MLIR's ``: A -> B``.
* Operator overloading maps ``mulf``/``addf``/``subf``/``negf`` onto
  ``*``/``+``/``-`` and unary ``-``; ``extf``/``truncf``/``channel_split`` are
  methods on the vector-register value; ``ptr + off`` is address arithmetic.
* Host-side index math (``arith.muli``/``divui``/``addi``/``index_cast``) is
  just native Python integer arithmetic.
"""

from pto import pto, vmi


def align_up(value: int, granule: int) -> int:
    """(value + granule - 1) // granule * granule — matches the (x+31)/32*32 idiom."""
    return (value + granule - 1) // granule * granule


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


# ADVANTAGE (module/func attributes): the whole
#   module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
#     func.func @rope_vmi_bf16(...) attributes {pto.kernel} { ... }
# header collapses into a single decorator on a normal Python function.
@pto.kernel(target_arch="a5", kind="vector")
def rope_vmi_bf16(
    # ADVANTAGE (types visible, written once): `Ptr[ui16, GM]` is exactly
    # `!pto.ptr<ui16, gm>`, so the shape/space/dtype stay on the page — but the
    # type appears once at the parameter, not repeated in a `: type` suffix.
    x_gm16:   pto.Ptr[pto.ui16, pto.GM],
    cos_gm16: pto.Ptr[pto.ui16, pto.GM],
    sin_gm16: pto.Ptr[pto.ui16, pto.GM],
    y_gm16:   pto.Ptr[pto.ui16, pto.GM],
    sCount: int,   # i32
    nCount: int,   # i32
    mode:   int,   # i32
) -> None:
    # --- reinterpret the ui16 GM buffers as their real element types ---------
    # ADVANTAGE (method call vs. spelled-out cast): `.cast(bf16)` replaces
    #   %x_gm = pto.castptr %x_gm16 : !pto.ptr<ui16, gm> -> !pto.ptr<bf16, gm>
    # where MLIR must restate the full source and result pointer types.
    x_gm:   pto.Ptr[pto.bf16, pto.GM] = x_gm16.cast(pto.bf16)
    cos_gm: pto.Ptr[pto.f16, pto.GM]  = cos_gm16.cast(pto.f16)
    sin_gm: pto.Ptr[pto.f16, pto.GM]  = sin_gm16.cast(pto.f16)
    y_gm:   pto.Ptr[pto.bf16, pto.GM] = y_gm16.cast(pto.bf16)

    s_count: int = sCount
    n_count: int = nCount

    # --- DMA byte counts, 32-byte aligned (2 bytes/elem) ---------------------
    # ADVANTAGE (host math is native Python): each line here folds a chain of
    # MLIR index ops into ordinary arithmetic. `align_up(cos_elems * 2, 32)`
    # stands in for the SSA sequence
    #   %cos_bytes   = arith.muli %cos_elems_i64, %c2_i64
    #   %cos_lines   = arith.addi %cos_bytes, %c31_i64
    #   %cos_lines_q = arith.divui %cos_lines, %c32_i64
    #   %cos_dma_bytes = arith.muli %cos_lines_q, %c32_i64
    # with no %cNN constants or intermediate %SSA temporaries to track.
    cos_elems: int = s_count * 64
    xy_elems:  int = s_count * n_count * 64
    cos_dma_bytes: int = align_up(cos_elems * 2, 32)
    xy_dma_bytes:  int = align_up(xy_elems * 2, 32)

    # --- UB tiles at fixed byte addresses (y aliases x in place) -------------
    # ADVANTAGE (intent is visible): `ub_ptr(bf16, 3840)` is castptr from a
    # constant address (%x_ub = pto.castptr %c3840_i64 : i64 -> !pto.ptr<bf16, ub>),
    # and y_ub reusing address 3840 makes the in-place x/y aliasing obvious.
    cos_ub: pto.Ptr[pto.f16, pto.UB]  = pto.ub_ptr(pto.f16, 0)
    sin_ub: pto.Ptr[pto.f16, pto.UB]  = pto.ub_ptr(pto.f16, 1920)
    x_ub:   pto.Ptr[pto.bf16, pto.UB] = pto.ub_ptr(pto.bf16, 3840)
    y_ub:   pto.Ptr[pto.bf16, pto.UB] = pto.ub_ptr(pto.bf16, 3840)

    # --- stage cos/sin tables and x rows into UB -----------------------------
    # ADVANTAGE (named operands): keyword args offset=/length=/nburst= label
    # each field, replacing MLIR's positional type wall
    #   pto.mte_gm_ub %cos_gm, %cos_ub, %c0_i64, %cos_dma_bytes
    #     nburst(%c1_i64, %cos_dma_bytes, %cos_dma_bytes)
    #     : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64
    # where the trailing i64,i64,... carry no names.
    pto.copy_gm_to_ub(cos_gm, cos_ub, offset=0, length=cos_dma_bytes,
                      nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.copy_gm_to_ub(sin_gm, sin_ub, offset=0, length=cos_dma_bytes,
                      nburst=(1, cos_dma_bytes, cos_dma_bytes))
    pto.copy_gm_to_ub(x_gm, x_ub, offset=0, length=xy_dma_bytes,
                      nburst=(1, xy_dma_bytes, xy_dma_bytes))

    # NOTE (no abstraction where none is warranted): pipe sync is intrinsically
    # low-level, so it maps 1:1 to pto.set_flag/wait_flag and is left verbatim —
    # the eDSL only hides detail that has a faithful higher-level meaning.
    pto.set_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")
    pto.wait_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")

    # --- derived tiling constants --------------------------------------------
    half_d:         int = 64 // 2          # 32
    half_d_aligned: int = align_up(half_d, 16)
    half_repeats:   int = ceil_div(half_d, 64)

    x_s_step: int = n_count * 64
    x_n_step: int = 64
    cs_s_step: int = 64
    y_s_step: int = n_count * 64
    y_n_step: int = 64

    is_half_mode: bool = (mode == 0)

    # ADVANTAGE (structure via native control flow): `with vmi.vecscope()` is the
    # pto.vecscope { ... } region, and the `if`/`for` below are scf.if / scf.for —
    # but they nest with ordinary Python indentation and use plain loop variables
    # instead of explicit %c0/%c1/%s_count index SSA values and region syntax.
    with vmi.vecscope():
        if is_half_mode:
            # HALF / NeoX: y1 = x1*cos - x2*sin ; y2 = x2*cos + x1*sin
            for s in range(s_count):
                x_s_off:  int = s * x_s_step
                cs_off:   int = s * cs_s_step
                y_s_off:  int = s * y_s_step

                for rep in range(half_repeats):
                    elem_off: int = rep * 64
                    active:   int = min(half_d - elem_off, 64)
                    # ADVANTAGE (parametric result type is a real argument): the
                    # width lives in the [64] subscript, so the builder actually
                    # receives it — unlike a Python LHS annotation, which is
                    # discarded before the call runs. Replaces
                    #   %mask = pto.vmi.create_mask %active : index -> !pto.vmi.mask<64xpred>
                    mask = vmi.create_mask[64](active)

                    cos1_off: int = cs_off + elem_off
                    cos2_off: int = cs_off + half_d_aligned + elem_off
                    sin1_off: int = cs_off + elem_off
                    sin2_off: int = cs_off + half_d_aligned + elem_off

                    # ADVANTAGE (width/dtype in subscript + ptr+off addressing):
                    # `load[Vreg[64, f16]](cos_ub + cos1_off)` carries the result
                    # shape/dtype as a genuine parametric argument and does the
                    # offsetting with pointer arithmetic. Replaces
                    #   %cos1_16 = pto.vmi.load %cos_ub[%cos1_off]
                    #       : !pto.ptr<f16, ub> -> !pto.vmi.vreg<64xf16>
                    cos1_16 = vmi.load[vmi.Vreg[64, vmi.f16]](cos_ub + cos1_off)
                    cos2_16 = vmi.load[vmi.Vreg[64, vmi.f16]](cos_ub + cos2_off)
                    sin1_16 = vmi.load[vmi.Vreg[64, vmi.f16]](sin_ub + sin1_off)
                    sin2_16 = vmi.load[vmi.Vreg[64, vmi.f16]](sin_ub + sin2_off)

                    # ADVANTAGE (conversion is a method, type stated once): `.extf(f32)`
                    # names only the target dtype; the result annotation appears a
                    # single time. Replaces MLIR's two-sided form
                    #   %cos1 = pto.vmi.extf %cos1_16
                    #       : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xf32>
                    cos1: vmi.Vreg[64, vmi.f32] = cos1_16.extf(vmi.f32)
                    cos2: vmi.Vreg[64, vmi.f32] = cos2_16.extf(vmi.f32)
                    sin1: vmi.Vreg[64, vmi.f32] = sin1_16.extf(vmi.f32)
                    sin2: vmi.Vreg[64, vmi.f32] = sin2_16.extf(vmi.f32)

                    for n in range(n_count):
                        x_off: int = x_s_off + n * x_n_step
                        y_off: int = y_s_off + n * y_n_step
                        x1_off: int = x_off + elem_off
                        x2_off: int = x_off + half_d_aligned + elem_off
                        y1_off: int = y_off + elem_off
                        y2_off: int = y_off + half_d_aligned + elem_off

                        x1_16 = vmi.load[vmi.Vreg[64, vmi.bf16]](x_ub + x1_off)
                        x2_16 = vmi.load[vmi.Vreg[64, vmi.bf16]](x_ub + x2_off)
                        x1: vmi.Vreg[64, vmi.f32] = x1_16.extf(vmi.f32)
                        x2: vmi.Vreg[64, vmi.f32] = x2_16.extf(vmi.f32)

                        # ADVANTAGE (operator overloading + implicit masks): mulf/
                        # subf/addf become * - +, folding named intermediates into
                        # one expression. The three MLIR ops per output —
                        #   %x1_cos   = pto.vmi.mulf %x1, %cos1 : vreg<64xf32>, vreg<64xf32> -> vreg<64xf32>
                        #   %x2_sin   = pto.vmi.mulf %x2, %sin1 : ...
                        #   %out1_f32 = pto.vmi.subf %x1_cos, %x2_sin : ...
                        # collapse to the line below, with the compiler managing masks.
                        out1_f32: vmi.Vreg[64, vmi.f32] = x1 * cos1 - x2 * sin1
                        out2_f32: vmi.Vreg[64, vmi.f32] = x2 * cos2 + x1 * sin2

                        # ADVANTAGE (narrowing names only the target): `.truncf(bf16)`
                        # leaves rounding/saturation to the compiler. Replaces
                        #   %out1 = pto.vmi.truncf %out1_f32
                        #       : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xbf16>
                        out1: vmi.Vreg[64, vmi.bf16] = out1_f32.truncf(vmi.bf16)
                        out2: vmi.Vreg[64, vmi.bf16] = out2_f32.truncf(vmi.bf16)

                        # ADVANTAGE (value/addr/mask as plain args): `ptr + off`
                        # addressing keeps the store compact. Replaces
                        #   pto.vmi.masked_store %out1, %y_ub[%y1_off], %mask
                        #       : !pto.vmi.vreg<64xbf16>, !pto.ptr<bf16, ub>, !pto.vmi.mask<64xpred>
                        vmi.masked_store(out1, y_ub + y1_off, mask)
                        vmi.masked_store(out2, y_ub + y2_off, mask)
        else:
            # INTERLEAVE / GPT-J: y = x*cos + rot(x)*sin, rot(x) = [-x_odd, x_even]
            for s in range(s_count):
                x_s_off: int = s * x_s_step
                cs_off:  int = s * cs_s_step
                y_s_off: int = s * y_s_step

                mask = vmi.create_mask[128](64)

                cos16 = vmi.load[vmi.Vreg[128, vmi.f16]](cos_ub + cs_off)
                sin16 = vmi.load[vmi.Vreg[128, vmi.f16]](sin_ub + cs_off)
                cos: vmi.Vreg[128, vmi.f32] = cos16.extf(vmi.f32)
                sin: vmi.Vreg[128, vmi.f32] = sin16.extf(vmi.f32)

                for n in range(n_count):
                    x_off: int = x_s_off + n * x_n_step
                    y_off: int = y_s_off + n * y_n_step

                    x16 = vmi.load[vmi.Vreg[128, vmi.bf16]](x_ub + x_off)
                    x: vmi.Vreg[128, vmi.f32] = x16.extf(vmi.f32)

                    # ADVANTAGE (tuple unpacking + unary minus): the even/odd
                    # rotation reads as ordinary Python — `channel_split` returns a
                    # pair and `-x_odd` is negf. Replaces the quoted generic form
                    # with fully spelled tuple types:
                    #   %x_even, %x_odd = "pto.vmi.channel_split"(%x)
                    #       : (!pto.vmi.vreg<128xf32>) -> (!pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>)
                    #   %neg_x_odd = pto.vmi.negf %x_odd : ... -> ...
                    #   %rot = "pto.vmi.channel_merge"(%neg_x_odd, %x_even) : (...) -> ...
                    x_even: vmi.Vreg[64, vmi.f32]
                    x_odd:  vmi.Vreg[64, vmi.f32]
                    x_even, x_odd = vmi.channel_split(x)
                    rot: vmi.Vreg[128, vmi.f32] = vmi.channel_merge(-x_odd, x_even)

                    y_f32: vmi.Vreg[128, vmi.f32] = x * cos + rot * sin
                    y16: vmi.Vreg[128, vmi.bf16] = y_f32.truncf(vmi.bf16)

                    vmi.masked_store(y16, y_ub + y_off, mask)

    pto.set_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")
    pto.wait_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")

    pto.copy_ub_to_gm(y_ub, y_gm, length=xy_dma_bytes,
                      nburst=(1, xy_dma_bytes, xy_dma_bytes))
    pto.barrier("PIPE_ALL")
