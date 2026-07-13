"""PTODSL source-backed backend for rope VF sim tests.

Unlike ``rope_mi`` (the ``mi`` backend), the kernels below are not traced
from Python function bodies -- they are loaded from the static ``.pto`` VMI
surface IR text files co-located with this example
(``rope_{f16,bf16,f32}.vmi.pto``).
"""

import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
PTO_FILE_F16 = HERE / "rope_f16.vmi.pto"
PTO_FILE_BF16 = HERE / "rope_bf16.vmi.pto"
PTO_FILE_F32 = HERE / "rope_f32.vmi.pto"


def _candidate_ptodsl_roots() -> list[Path]:
    candidates: list[Path] = []

    env_keys = ("PTODSL_PKG_ROOT", "PTOAS_ROOT", "PTOAS_HOME")
    for key in env_keys:
        raw = os.environ.get(key)
        if not raw:
            continue
        base = Path(raw).expanduser()
        if key == "PTODSL_PKG_ROOT":
            candidates.append(base)
        else:
            candidates.append(base / "ptodsl")

    candidates.append(REPO_ROOT / "PTOAS" / "ptodsl")
    for sibling in sorted(REPO_ROOT.parent.glob("PTOAS*")):
        candidates.append(sibling / "ptodsl")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _import_ptodsl_pto():
    try:
        return importlib.import_module("ptodsl").pto
    except ModuleNotFoundError as first_error:
        for pkg_root in _candidate_ptodsl_roots():
            if not (pkg_root / "ptodsl" / "__init__.py").exists():
                continue
            pkg_root_str = str(pkg_root)
            if pkg_root_str not in sys.path:
                sys.path.insert(0, pkg_root_str)
            try:
                return importlib.import_module("ptodsl").pto
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "Unable to import ptodsl. Set PTODSL_PKG_ROOT or PTOAS_ROOT, "
            "or place a PTOAS checkout next to this repo."
        ) from first_error


pto = _import_ptodsl_pto()


@pto.jit(
    name="rope_vmi_f16",
    target="a5",
    kernel_kind="vector",
    source=str(PTO_FILE_F16),
)
def rope_vmi_f16(
    x_ptr: pto.ptr(pto.ui16, "gm"),
    cos_ptr: pto.ptr(pto.ui16, "gm"),
    sin_ptr: pto.ptr(pto.ui16, "gm"),
    y_ptr: pto.ptr(pto.ui16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    raise RuntimeError("source-backed PTODSL kernel body should not execute")


@pto.jit(
    name="rope_vmi_bf16",
    target="a5",
    kernel_kind="vector",
    source=str(PTO_FILE_BF16),
)
def rope_vmi_bf16(
    x_ptr: pto.ptr(pto.ui16, "gm"),
    cos_ptr: pto.ptr(pto.ui16, "gm"),
    sin_ptr: pto.ptr(pto.ui16, "gm"),
    y_ptr: pto.ptr(pto.ui16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    raise RuntimeError("source-backed PTODSL kernel body should not execute")


@pto.jit(
    name="rope_vmi_f32",
    target="a5",
    kernel_kind="vector",
    source=str(PTO_FILE_F32),
)
def rope_vmi_f32(
    x_ptr: pto.ptr(pto.ui32, "gm"),
    cos_ptr: pto.ptr(pto.ui32, "gm"),
    sin_ptr: pto.ptr(pto.ui32, "gm"),
    y_ptr: pto.ptr(pto.ui32, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    mode: pto.i32,
):
    raise RuntimeError("source-backed PTODSL kernel body should not execute")


_COMPILED: dict[str, object] = {}


def is_supported_variant(mode: str, dtype: str, cycle: bool = False) -> bool:
    del cycle
    return dtype in {"f16", "bf16", "f32"} and mode in {"half", "interleave"}


def describe() -> str:
    return "vmi"


def _kernel_for_dtype(dtype: str):
    if dtype == "f16":
        return "f16", rope_vmi_f16
    if dtype == "bf16":
        return "bf16", rope_vmi_bf16
    if dtype == "f32":
        return "f32", rope_vmi_f32
    raise ValueError(f"unsupported vmi dtype: {dtype}")


def prepare(dtype: str = "f16", force_rebuild: bool = False):
    del force_rebuild
    key, kernel = _kernel_for_dtype(dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = kernel.compile()
        _COMPILED[key] = compiled
    return compiled


def launch(ref: dict, cycle: bool = False):
    import torch

    from common.torch_runtime import device_str, empty_npu, stream_ptr, sync

    dtype_name = ref["dtype"]
    mode_name = ref["mode"]
    mode = 0 if mode_name == "half" else 1

    if dtype_name == "f16":
        x = torch.from_numpy(ref["x"]).to(torch.float16).to(device_str())
        cos = torch.from_numpy(ref["cos"]).to(torch.float16).to(device_str())
        sin = torch.from_numpy(ref["sin"]).to(torch.float16).to(device_str())
        y = empty_npu(ref["y"].shape, torch.float16)
        compiled = prepare("f16")
    elif dtype_name == "f32":
        x = torch.from_numpy(ref["x"]).to(torch.float32).to(device_str())
        cos = torch.from_numpy(ref["cos"]).to(torch.float32).to(device_str())
        sin = torch.from_numpy(ref["sin"]).to(torch.float32).to(device_str())
        y = empty_npu(ref["y"].shape, torch.float32)
        compiled = prepare("f32")
    elif dtype_name == "bf16":
        x = torch.from_numpy(ref["x"]).to(torch.bfloat16).to(device_str())
        cos = torch.from_numpy(ref["cos"]).to(torch.float16).to(device_str())
        sin = torch.from_numpy(ref["sin"]).to(torch.float16).to(device_str())
        y = empty_npu(ref["y"].shape, torch.bfloat16)
        compiled = prepare("bf16")
    else:
        raise ValueError(f"unsupported vmi launch dtype: {dtype_name}")

    s_count, n_count = [int(v) for v in ref["params"]]

    compiled[1, stream_ptr()](
        x.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        y.data_ptr(),
        s_count,
        n_count,
        mode,
    )
    sync()
    return y


def cache_tag() -> str:
    return (
        f"{describe()}:"
        f"{PTO_FILE_F16}:{os.path.getmtime(PTO_FILE_F16):.0f}:"
        f"{PTO_FILE_BF16}:{os.path.getmtime(PTO_FILE_BF16):.0f}:"
        f"{PTO_FILE_F32}:{os.path.getmtime(PTO_FILE_F32):.0f}"
    )
