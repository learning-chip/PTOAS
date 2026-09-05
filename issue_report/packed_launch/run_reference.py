#!/usr/bin/env python3
"""Launch and verify the compiled ASC reference shared library."""
from __future__ import annotations

import argparse
import ctypes
import os
import numpy as np

WIDTH = 24


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--device", default=os.environ.get("NPU_TEST_DEVICE", "npu:0"))
    args = ap.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    host_input = np.linspace(-3.0, 4.0, WIDTH, dtype=np.float32)
    expected = host_input * np.float32(1.25) + np.float32(1.0)
    src = torch.from_numpy(host_input).to(args.device)
    dst = torch.full((WIDTH,), float("nan"), dtype=torch.float32, device=args.device)
    lib = ctypes.CDLL(os.path.abspath(args.library))
    launch = lib.launch_reference_compact_f32_store
    launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    launch(ctypes.c_void_p(int(getattr(stream, "value", stream))),
           ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()))
    torch.npu.synchronize()
    np.testing.assert_allclose(dst.cpu().numpy(), expected, rtol=1e-6, atol=1e-6)
    print("ASC_LAUNCH_OK: compact output matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
