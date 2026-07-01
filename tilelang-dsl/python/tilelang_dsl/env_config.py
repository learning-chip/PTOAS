# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Environment configuration helper for TileLang DSL PybindBackend.

This module provides utilities to check and configure the environment
for using the PybindBackend which requires MLIR Python bindings.
"""

import os
import sys
from typing import Optional


def check_mlir_bindings_available() -> bool:
    """Check if MLIR Python bindings are available.

    Returns:
        True if mlir.ir module can be imported, False otherwise.
    """
    try:
        from mlir import ir
        return True
    except ImportError:
        return False


def check_pto_dialect_available() -> bool:
    """Check if PTO dialect bindings are available.

    Returns:
        True if pto.dialects.pto module can be imported, False otherwise.
    """
    try:
        from pto.dialects import pto
        return True
    except ImportError:
        return False


def get_environment_status() -> dict:
    """Get current environment status for PybindBackend.

    Returns:
        Dictionary with status information.
    """
    status = {
        "mlir_bindings": check_mlir_bindings_available(),
        "pto_dialect": check_pto_dialect_available(),
        "python_path": sys.path,
        "pythonpath_env": os.environ.get("PYTHONPATH", ""),
    }
    return status


def print_environment_help() -> None:
    """Print help message for setting up the environment."""
    print("=" * 60)
    print("PybindBackend Environment Configuration Guide")
    print("=" * 60)
    print()
    print("PybindBackend requires MLIR Python bindings from LLVM build.")
    print()
    print("To set up the environment, add the following to PYTHONPATH:")
    print()
    print("1. LLVM MLIR Python package:")
    print("   $LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core")
    print()
    print("2. PTO dialect bindings (optional, for PTO ops):")
    print("   $PTOAS_BUILD_DIR/python/mlir")
    print("   OR")
    print("   $PTO_INSTALL_DIR/mlir")
    print()
    print("Example setup:")
    print()
    print("  # Assuming LLVM is built at ~/llvm-workspace/llvm-project/build-shared")
    print("  export LLVM_BUILD_DIR=~/llvm-workspace/llvm-project/build-shared")
    print()
    print("  # Assuming PTOAS is at ~/llvm-workspace/pto-as")
    print("  export PTOAS_BUILD_DIR=~/llvm-workspace/pto-as/build")
    print()
    print("  # Set PYTHONPATH")
    print("  export PYTHONPATH=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$PTOAS_BUILD_DIR/python")
    print()
    print("After setup, verify with:")
    print("  python3 -c 'from mlir import ir; print(\"MLIR bindings OK\")'")
    print("  python3 -c 'from pto.dialects import pto; print(\"PTO dialect OK\")'")
    print()
    print("=" * 60)


def verify_environment() -> bool:
    """Verify the environment is properly configured.

    Returns:
        True if all required components are available, False otherwise.
    """
    status = get_environment_status()

    all_ok = status["mlir_bindings"]

    if not all_ok:
        print("Environment verification FAILED:")
        if not status["mlir_bindings"]:
            print("  - MLIR Python bindings NOT found")
        print()
        print_environment_help()
        return False

    print("Environment verification PASSED:")
    print("  - MLIR Python bindings: OK")
    if status["pto_dialect"]:
        print("  - PTO dialect: OK")
    else:
        print("  - PTO dialect: NOT available (PTO ops will be disabled)")

    return True


__all__ = [
    "check_mlir_bindings_available",
    "check_pto_dialect_available",
    "get_environment_status",
    "print_environment_help",
    "verify_environment",
]