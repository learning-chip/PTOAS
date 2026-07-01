# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Simple verification script without pytest dependency."""

import sys
import os
from pathlib import Path

# Setup path
script_path = Path(__file__).resolve()
repo_root = script_path.parents[1]
sys.path.insert(0, str(repo_root / "python"))

from tilelang_dsl.lowering_backend import (
    LoweringResult,
    LoweringBackend,
    TextBackend,
    PybindBackend,
    get_backend,
    lower_with_backend,
)
from tilelang_dsl.backend_validator import (
    VerifyResult,
    BackendComparisonResult,
    compare_backends,
    _normalize_mlir_text,
)


def test_lowering_result():
    """Test LoweringResult."""
    print("Test: LoweringResult")

    # Empty result
    empty = LoweringResult()
    assert not empty, "Empty result should be falsy"

    # Text result
    text_result = LoweringResult(text="module {}")
    assert text_result, "Result with text should be truthy"
    assert text_result.as_text() == "module {}", "as_text() returns text"

    print("  [PASS]")
    return True


def test_text_backend():
    """Test TextBackend."""
    print("Test: TextBackend")

    backend = TextBackend()
    assert backend.name() == "text"

    backend2 = get_backend("text")
    assert isinstance(backend2, TextBackend)

    print("  [PASS]")
    return True


def test_pybind_backend():
    """Test PybindBackend."""
    print("Test: PybindBackend")

    backend = PybindBackend()
    assert backend.name() == "pybind"

    available = backend.is_available()
    print(f"  is_available() = {available}")

    backend2 = get_backend("pybind")
    assert isinstance(backend2, PybindBackend)

    print("  [PASS]")
    return True


def test_get_backend():
    """Test get_backend factory."""
    print("Test: get_backend")

    # Default backend
    backend = get_backend()
    assert backend.name() == "text"

    # Unknown backend raises
    try:
        get_backend("unknown")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("  [PASS]")
    return True


def test_verify_result():
    """Test VerifyResult."""
    print("Test: VerifyResult")

    passed = VerifyResult(passed=True)
    assert passed.passed

    failed = VerifyResult(passed=False, error="test error")
    assert not failed.passed
    assert failed.error == "test error"

    print("  [PASS]")
    return True


def test_normalize_mlir_text():
    """Test MLIR text normalization."""
    print("Test: normalize_mlir_text")

    # Empty text
    assert _normalize_mlir_text("") == ""

    # Skip comments
    text = "// comment\nmodule {}"
    normalized = _normalize_mlir_text(text, skip_comments=True)
    assert "comment" not in normalized
    assert "module {}" in normalized

    # Skip blank lines
    text = "module {\n\n\n}"
    normalized = _normalize_mlir_text(text, skip_blank_lines=True)
    assert "\n\n" not in normalized

    # Normalize SSA names
    text = "%0 = arith.constant 0 : i32"
    normalized = _normalize_mlir_text(text, normalize_ssa_names=True)
    assert "%ssa" in normalized

    print("  [PASS]")
    return True


def test_compare_backends_mock():
    """Test compare_backends with mock kernel."""
    print("Test: compare_backends (mock)")

    from dataclasses import dataclass

    @dataclass
    class MockSemanticKernel:
        target: str = "a5"
        op: str = "test"
        symbol_name: str = "mock_kernel"  # Used by compare_backends
        verify_enabled: bool = True
        advanced_enabled: bool = False
        dtype_signature: tuple = ()
        parameters: tuple = ()
        tile_bindings: tuple = ()
        body: tuple = ()
        inline_helpers: tuple = ()
        kernel_family: str = "test_family"  # Required attribute

    mock_kernel = MockSemanticKernel()
    result = compare_backends(mock_kernel)

    assert isinstance(result, BackendComparisonResult)
    assert result.kernel_name == "mock_kernel"
    print(f"  pybind_available = {result.pybind_available}")

    print("  [PASS]")
    return True


def test_backend_comparison_result():
    """Test BackendComparisonResult properties."""
    print("Test: BackendComparisonResult")

    result_ok = BackendComparisonResult(
        kernel_name="test",
        text_output="module {}",
        pybind_output="module {}",
        text_normalized="module {}",
        pybind_normalized="module {}",
        matches=True,
        normalized_matches=True,
        differences=(),
        text_verify_passed=True,
        pybind_verify_passed=True,
        text_verify_error=None,
        pybind_verify_error=None,
        pybind_available=True,
    )
    assert result_ok.ok

    result_diff = BackendComparisonResult(
        kernel_name="test",
        text_output="",
        pybind_output="",
        text_normalized="",
        pybind_normalized="",
        matches=False,
        normalized_matches=False,
        differences=("diff line"),
        text_verify_passed=True,
        pybind_verify_passed=True,
        text_verify_error=None,
        pybind_verify_error=None,
        pybind_available=True,
    )
    assert result_diff.has_differences

    print("  [PASS]")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PybindBackend Verification Tests")
    print("=" * 60)

    tests = [
        test_lowering_result,
        test_text_backend,
        test_pybind_backend,
        test_get_backend,
        test_verify_result,
        test_normalize_mlir_text,
        test_compare_backends_mock,
        test_backend_comparison_result,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)