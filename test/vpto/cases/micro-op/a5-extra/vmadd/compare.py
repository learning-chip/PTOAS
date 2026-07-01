#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import numpy as np


def compare_float(golden_path: str, output_path: str, label: str, atol: float) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        print(f"[ERROR] missing file for {label}")
        return False
    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    ok = golden.shape == output.shape and np.allclose(
        golden, output, atol=atol, rtol=atol, equal_nan=True
    )
    if not ok:
        diff = np.max(np.abs(golden.astype(np.float64) - output.astype(np.float64)))
        print(f"[ERROR] compare failed: {label}, max_abs_diff={diff}")
    return ok


def compare_exact(golden_path: str, output_path: str, dtype, label: str) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        print(f"[ERROR] missing file for {label}")
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    ok = golden.shape == output.shape and np.array_equal(golden, output)
    if not ok:
        mismatch = np.flatnonzero(golden != output)
        first = int(mismatch[0]) if mismatch.size else -1
        print(
            f"[ERROR] compare failed: {label}, first_mismatch={first}, "
            f"golden={golden[first] if first >= 0 else 'n/a'}, "
            f"output={output[first] if first >= 0 else 'n/a'}"
        )
    return ok


def main() -> None:
    checks = [
        compare_float("golden_vmadd.bin", "out_vmadd.bin", "vmadd", 2e-4),
    ]
    if not all(checks):
        sys.exit(2)
    print("[INFO] compare passed (a5 extra vmadd)")


if __name__ == "__main__":
    main()
