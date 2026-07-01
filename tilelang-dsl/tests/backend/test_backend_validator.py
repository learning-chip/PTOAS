# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for backend_validator (Issue #237 Phase 4).

Can run without pytest:
    python3 tests/backend/test_backend_validator.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from tilelang_dsl.backend_validator import (
    VerifyResult,
    BackendComparisonResult,
    compare_backends,
    run_backend_validation_suite,
    print_comparison_summary,
    _normalize_mlir_text,
    _find_text_differences,
)

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def test_verify_result_passed():
    """Passed result."""
    print("Test: VerifyResult passed")

    result = VerifyResult(passed=True)
    assert result.passed, "passed is True"
    assert result.error is None, "error is None"

    print("  [PASS]")
    return True


def test_verify_result_failed():
    """Failed result with error."""
    print("Test: VerifyResult failed")

    result = VerifyResult(passed=False, error="verification failed")
    assert not result.passed, "passed is False"
    assert result.error == "verification failed", "error matches"

    print("  [PASS]")
    return True


def test_backend_comparison_result_ok():
    """ok property checks alignment status."""
    print("Test: BackendComparisonResult ok")

    result = BackendComparisonResult(
        kernel_name="test_kernel",
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
    assert result.ok, "ok is True when matches"

    print("  [PASS]")
    return True


def test_backend_comparison_result_has_differences():
    """has_differences property."""
    print("Test: BackendComparisonResult has_differences")

    result_no_diff = BackendComparisonResult(
        kernel_name="test",
        text_output="",
        pybind_output="",
        text_normalized="",
        pybind_normalized="",
        matches=True,
        normalized_matches=True,
        differences=(),
        text_verify_passed=True,
        pybind_verify_passed=True,
        text_verify_error=None,
        pybind_verify_error=None,
        pybind_available=True,
    )
    assert not result_no_diff.has_differences, "No differences"

    result_with_diff = BackendComparisonResult(
        kernel_name="test",
        text_output="",
        pybind_output="",
        text_normalized="",
        pybind_normalized="",
        matches=False,
        normalized_matches=False,
        differences=("--- text_backend", "+++ pybind_backend"),
        text_verify_passed=True,
        pybind_verify_passed=True,
        text_verify_error=None,
        pybind_verify_error=None,
        pybind_available=True,
    )
    assert result_with_diff.has_differences, "Has differences"

    print("  [PASS]")
    return True


def test_normalize_mlir_empty():
    """Empty text returns empty."""
    print("Test: normalize_mlir_text empty")

    assert _normalize_mlir_text("") == "", "Empty returns empty"

    print("  [PASS]")
    return True


def test_normalize_mlir_skip_comments():
    """Comments are skipped."""
    print("Test: normalize_mlir_text skip_comments")

    text = "// comment\nmodule {}"
    normalized = _normalize_mlir_text(text, skip_comments=True)
    assert "comment" not in normalized, "Comment removed"
    assert "module {}" in normalized, "Module retained"

    print("  [PASS]")
    return True


def test_normalize_mlir_skip_blank_lines():
    """Blank lines are skipped."""
    print("Test: normalize_mlir_text skip_blank_lines")

    text = "module {\n\n\n}"
    normalized = _normalize_mlir_text(text, skip_blank_lines=True)
    assert "\n\n" not in normalized, "Multiple newlines removed"

    print("  [PASS]")
    return True


def test_normalize_mlir_ssa_names():
    """SSA names are normalized."""
    print("Test: normalize_mlir_text normalize_ssa_names")

    text = "%0 = arith.constant 0 : i32"
    normalized = _normalize_mlir_text(text, normalize_ssa_names=True)
    assert "%ssa" in normalized, "SSA normalized to %ssa"
    assert "%0" not in normalized, "Original SSA removed"

    print("  [PASS]")
    return True


def test_normalize_mlir_combined():
    """Combined normalization options."""
    print("Test: normalize_mlir_text combined")

    text = """
// Kernel: test
module {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
}
"""
    normalized = _normalize_mlir_text(
        text,
        skip_comments=True,
        skip_blank_lines=True,
        normalize_ssa_names=True,
    )
    assert "// Kernel" not in normalized, "Comment removed"
    assert "%ssa" in normalized, "SSA normalized"

    print("  [PASS]")
    return True


def test_find_text_differences_identical():
    """Identical texts have no differences."""
    print("Test: find_text_differences identical")

    text = "module {}"
    diffs = _find_text_differences(text, text)
    assert len(diffs) == 0, "No differences for identical"

    print("  [PASS]")
    return True


def test_find_text_differences_different():
    """Different texts have differences."""
    print("Test: find_text_differences different")

    text_a = "module { %0 = arith.constant 0 : i32 }"
    text_b = "module { %0 = arith.constant 1 : i32 }"
    diffs = _find_text_differences(text_a, text_b)
    assert len(diffs) > 0, "Has differences"

    print("  [PASS]")
    return True


def test_find_text_differences_empty():
    """Empty texts return appropriate messages."""
    print("Test: find_text_differences empty")

    diffs_a_empty = _find_text_differences("", "module {}")
    assert len(diffs_a_empty) > 0, "Has message for empty A"
    assert "empty" in diffs_a_empty[0].lower() or "Text backend" in diffs_a_empty[0], "Message about empty"

    diffs_b_empty = _find_text_differences("module {}", "")
    assert len(diffs_b_empty) > 0, "Has message for empty B"
    assert "empty" in diffs_b_empty[0].lower() or "Pybind backend" in diffs_b_empty[0], "Message about empty"

    print("  [PASS]")
    return True


def test_compare_backends_returns_result():
    """compare_backends returns BackendComparisonResult."""
    print("Test: compare_backends returns result")

    from dataclasses import dataclass

    @dataclass
    class MockSemanticKernel:
        target: str = "a5"
        op: str = "test"
        symbol_name: str = "mock_kernel"
        kernel_family: str = "test_family"
        verify_enabled: bool = True
        advanced_enabled: bool = False
        dtype_signature: tuple = ()
        parameters: tuple = ()
        tile_bindings: tuple = ()
        body: tuple = ()
        inline_helpers: tuple = ()

    mock_kernel = MockSemanticKernel()
    result = compare_backends(mock_kernel)

    assert isinstance(result, BackendComparisonResult), "Returns BackendComparisonResult"
    assert result.kernel_name == "mock_kernel", "Kernel name matches"
    assert isinstance(result.text_output, str), "text_output is string"
    assert isinstance(result.matches, bool), "matches is bool"
    assert isinstance(result.pybind_available, bool), "pybind_available is bool"

    print(f"  pybind_available = {result.pybind_available}")
    print("  [PASS]")
    return True


def test_run_validation_suite_empty():
    """Empty suite returns empty results."""
    print("Test: run_backend_validation_suite empty")

    results = run_backend_validation_suite([])
    assert len(results) == 0, "Empty suite returns empty"

    print("  [PASS]")
    return True


def test_print_comparison_summary():
    """print_comparison_summary works."""
    print("Test: print_comparison_summary")

    print_comparison_summary(())
    print("  (summary printed)")
    print("  [PASS]")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Backend Validator Tests")
    print("=" * 60)

    tests = [
        test_verify_result_passed,
        test_verify_result_failed,
        test_backend_comparison_result_ok,
        test_backend_comparison_result_has_differences,
        test_normalize_mlir_empty,
        test_normalize_mlir_skip_comments,
        test_normalize_mlir_skip_blank_lines,
        test_normalize_mlir_ssa_names,
        test_normalize_mlir_combined,
        test_find_text_differences_identical,
        test_find_text_differences_different,
        test_find_text_differences_empty,
        test_compare_backends_returns_result,
        test_run_validation_suite_empty,
        test_print_comparison_summary,
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