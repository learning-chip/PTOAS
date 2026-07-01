#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

ELEMS = 1024
SEED = 29


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    f_acc = rng.uniform(-2.0, 2.0, size=ELEMS).astype(np.float32)
    f_lhs = rng.uniform(-3.0, 3.0, size=ELEMS).astype(np.float32)
    f_rhs = rng.uniform(-1.0, 1.0, size=ELEMS).astype(np.float32)

    f_acc.tofile(out / "f_acc.bin")
    f_lhs.tofile(out / "f_lhs.bin")
    f_rhs.tofile(out / "f_rhs.bin")

    (f_lhs * f_acc + f_rhs).astype(np.float32).tofile(out / "golden_vmadd.bin")


if __name__ == "__main__":
    main()
