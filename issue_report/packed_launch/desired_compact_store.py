#!/usr/bin/env python3
"""Minimal TileLang-free PTODSL compact-store reproducer.

The packed UE8M0 path needs a compact one-value-per-group store.  This probe
isolates the lower-level VMI form used by that path: a broadcast load followed
by arithmetic and ``group=1, stride=1`` storage.  No TileLang APIs are used.
"""
import argparse
import os

import numpy as np
from ptodsl import pto

WIDTH = 24
VL = 64
SCALE = 1.25
BIAS = 1.0
BYTES = WIDTH * 4


@pto.jit(
    name="compact_f32_store_desired",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
    insert_sync=False,
)
def compact_f32_store_desired(
    input_gm: pto.ptr(pto.f32, "gm"),
    output_gm: pto.ptr(pto.f32, "gm"),
):
    zero = pto.const(0, dtype=pto.i64)
    input_ub = pto.castptr(zero, pto.ptr(pto.f32, "ub"))
    output_ub = pto.addptr(input_ub, WIDTH)
    offset = pto.const(0, dtype=pto.index)
    stride = pto.const(1, dtype=pto.index)
    mask = pto.vmi.create_mask(VL, size=VL)

    pto.mte_gm_ub(input_gm, input_ub, 0, BYTES, nburst=(1, BYTES, BYTES))
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    for index in range(WIDTH):
        value = pto.vmi.vload(
            pto.addptr(input_ub, index), offset, size=VL, dist_mode="brc"
        )
        value = pto.vmi.vmuls(value, SCALE, mask)
        value = pto.vmi.vadds(value, BIAS, mask)
        pto.vmi.vstore(
            value, pto.addptr(output_ub, index), offset, group=1, stride=stride
        )

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(output_ub, output_gm, BYTES, nburst=(1, BYTES, BYTES))
    pto.pipe_barrier(pto.Pipe.ALL)


def compile_kernel():
    return compact_f32_store_desired.compile()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("--device", default=os.environ.get("NPU_TEST_DEVICE", "npu:0"))
    args = ap.parse_args(argv)
    compiled = compile_kernel()
    text = compiled.mlir_text()
    if "pto.vmi.vstore" not in text or 'group = 1' not in text:
        raise SystemExit("FAIL: expected VMI compact group=1 store in MLIR")
    if args.emit_mlir:
        print(text)
        return 0
    if args.compile_only:
        print(f"PTODSL_COMPILE_OK: {len(text)} bytes of verified MLIR")
        return 0

    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    host_input = np.linspace(-3.0, 4.0, WIDTH, dtype=np.float32)
    src = torch.from_numpy(host_input).to(args.device)
    dst = torch.full((WIDTH,), float("nan"), dtype=torch.float32, device=args.device)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    print("PTODSL_COMPILE_OK: VMI frontend accepted and compiled", flush=True)
    compiled[1, stream](src.data_ptr(), dst.data_ptr())
    try:
        torch.npu.synchronize()
    except RuntimeError as exc:
        print(f"VMI_RUNTIME_ERROR: {exc}", flush=True)
        return 2
    expected = host_input * np.float32(SCALE) + np.float32(BIAS)
    np.testing.assert_allclose(dst.cpu().numpy(), expected, rtol=1e-6, atol=1e-6)
    print("VMI_LAUNCH_OK: compact output matched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
