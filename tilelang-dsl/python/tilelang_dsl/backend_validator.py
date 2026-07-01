# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Backend validation and comparison utilities for Issue #237.

This module provides tools to compare text and pybind backend outputs,
enabling safe migration with continuous validation.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .semantic import SemanticKernel
    from .lowering_backend import LoweringResult


@dataclass(frozen=True)
class VerifyResult:
    """Result of MLIR verification."""

    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class BackendComparisonResult:
    """Result of comparing text and pybind backend outputs."""

    kernel_name: str
    text_output: str
    pybind_output: str
    text_normalized: str
    pybind_normalized: str
    matches: bool
    normalized_matches: bool
    differences: tuple[str, ...]
    text_verify_passed: bool
    pybind_verify_passed: bool
    text_verify_error: str | None
    pybind_verify_error: str | None
    pybind_available: bool

    @property
    def ok(self) -> bool:
        """Check if comparison indicates successful alignment."""
        return (
            self.matches
            or self.normalized_matches
            and self.text_verify_passed
            and self.pybind_verify_passed
        )

    @property
    def has_differences(self) -> bool:
        """Check if there are textual differences."""
        return len(self.differences) > 0


def compare_backends(
    kernel: "SemanticKernel",
    *,
    normalize_text: bool = True,
    skip_comments: bool = True,
    skip_blank_lines: bool = True,
    normalize_ssa_names: bool = False,
) -> BackendComparisonResult:
    """Compare text and pybind backend outputs for a SemanticKernel.

    Args:
        kernel: The semantic kernel to compare.
        normalize_text: Whether to normalize text before comparison.
        skip_comments: Whether to skip comment lines in comparison.
        skip_blank_lines: Whether to skip blank lines in comparison.
        normalize_ssa_names: Whether to normalize SSA name differences.

    Returns:
        BackendComparisonResult with detailed comparison info.
    """
    from .lowering_backend import get_backend

    text_backend = get_backend("text")
    pybind_backend = get_backend("pybind")

    kernel_name = kernel.symbol_name

    # Get text backend output (always available)
    text_result = text_backend.lower(kernel)
    text_output = text_result.as_text()

    # Check if pybind backend is available
    pybind_available = pybind_backend.is_available()

    if not pybind_available:
        return BackendComparisonResult(
            kernel_name=kernel_name,
            text_output=text_output,
            pybind_output="",
            text_normalized=_normalize_mlir_text(
                text_output,
                skip_comments=skip_comments,
                skip_blank_lines=skip_blank_lines,
                normalize_ssa_names=normalize_ssa_names,
            ),
            pybind_normalized="",
            matches=False,
            normalized_matches=False,
            differences=tuple(["PybindBackend not yet implemented"]),
            text_verify_passed=True,
            pybind_verify_passed=False,
            text_verify_error=None,
            pybind_verify_error="PybindBackend not yet implemented",
            pybind_available=False,
        )

    # Get pybind backend output
    try:
        pybind_result = pybind_backend.lower(kernel)
        pybind_output = pybind_result.as_text()
        pybind_verify_error = None
    except NotImplementedError as e:
        pybind_output = ""
        pybind_verify_error = str(e)

    # Normalize for comparison
    text_normalized = _normalize_mlir_text(
        text_output,
        skip_comments=skip_comments,
        skip_blank_lines=skip_blank_lines,
        normalize_ssa_names=normalize_ssa_names,
    )
    pybind_normalized = _normalize_mlir_text(
        pybind_output,
        skip_comments=skip_comments,
        skip_blank_lines=skip_blank_lines,
        normalize_ssa_names=normalize_ssa_names,
    )

    # Direct comparison
    matches = text_output == pybind_output
    normalized_matches = text_normalized == pybind_normalized

    # Find differences
    differences = _find_text_differences(text_output, pybind_output)

    # Verify both outputs
    text_verify = _verify_output(text_result)
    pybind_verify = _verify_output(None if not pybind_available else pybind_result)

    return BackendComparisonResult(
        kernel_name=kernel_name,
        text_output=text_output,
        pybind_output=pybind_output,
        text_normalized=text_normalized,
        pybind_normalized=pybind_normalized,
        matches=matches,
        normalized_matches=normalized_matches,
        differences=differences,
        text_verify_passed=text_verify.passed,
        pybind_verify_passed=pybind_verify.passed,
        text_verify_error=text_verify.error,
        pybind_verify_error=pybind_verify.error,
        pybind_available=pybind_available,
    )


def _normalize_mlir_text(
    text: str,
    *,
    skip_comments: bool = True,
    skip_blank_lines: bool = True,
    normalize_ssa_names: bool = False,
) -> str:
    """Normalize MLIR text for comparison.

    Args:
        text: The MLIR text to normalize.
        skip_comments: Whether to remove comment lines.
        skip_blank_lines: Whether to remove blank lines.
        normalize_ssa_names: Whether to normalize SSA naming (e.g., %0, %1 to %ssa).

    Returns:
        Normalized MLIR text.
    """
    if not text:
        return ""

    lines = text.splitlines()

    # Process each line
    processed_lines = []
    for line in lines:
        # Skip comments
        if skip_comments and line.lstrip().startswith("//"):
            continue

        # Skip blank lines
        if skip_blank_lines and not line.strip():
            continue

        # Normalize SSA names
        if normalize_ssa_names:
            line = _normalize_ssa_in_line(line)

        # Normalize whitespace
        line = line.strip()

        processed_lines.append(line)

    return "\n".join(processed_lines)


def _normalize_ssa_in_line(line: str) -> str:
    """Normalize SSA names in a line (e.g., %0, %1 to %ssa)."""
    # Replace SSA names like %0, %1, %arg0, etc. with %ssa
    return re.sub(r"%[a-zA-Z0-9_]+", "%ssa", line)


def _verify_output(result: "LoweringResult | None") -> VerifyResult:
    """Verify MLIR output.

    Args:
        result: The lowering result to verify.

    Returns:
        VerifyResult with pass/fail status.
    """
    if result is None:
        return VerifyResult(passed=False, error="No result to verify")

    try:
        module = result.as_module()
        module.operation.verify()
        return VerifyResult(passed=True)
    except Exception as e:
        return VerifyResult(passed=False, error=str(e))


def _find_text_differences(text_a: str, text_b: str) -> tuple[str, ...]:
    """Find line-level differences between two MLIR texts.

    Args:
        text_a: First MLIR text.
        text_b: Second MLIR text.

    Returns:
        Tuple of difference lines (unified diff format).
    """
    if not text_a or not text_b:
        if not text_a:
            return tuple(["Text backend output is empty"])
        if not text_b:
            return tuple(["Pybind backend output is empty"])

    a_lines = text_a.splitlines()
    b_lines = text_b.splitlines()

    diff = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile="text_backend",
        tofile="pybind_backend",
        lineterm="",
    )

    return tuple(diff)


def run_backend_validation_suite(
    kernels: list["SemanticKernel"],
    *,
    require_all_match: bool = False,
    normalize_text: bool = True,
    verbose: bool = False,
) -> tuple[BackendComparisonResult, ...]:
    """Run validation suite over multiple kernels.

    Args:
        kernels: List of semantic kernels to validate.
        require_all_match: Whether to raise error on mismatch.
        normalize_text: Whether to normalize text before comparison.
        verbose: Whether to print detailed output.

    Returns:
        Tuple of BackendComparisonResult for each kernel.

    Raises:
        AssertionError: If require_all_match and any kernel mismatches.
    """
    results = []

    for kernel in kernels:
        result = compare_backends(kernel, normalize_text=normalize_text)
        results.append(result)

        if verbose:
            print(f"Kernel: {result.kernel_name}")
            print(f"  Matches: {result.matches}")
            print(f"  Normalized matches: {result.normalized_matches}")
            print(f"  Text verify: {result.text_verify_passed}")
            print(f"  Pybind verify: {result.pybind_verify_passed}")
            if result.has_differences:
                print(f"  Differences ({len(result.differences)} lines):")
                for diff_line in result.differences[:10]:
                    print(f"    {diff_line}")

        if require_all_match and not result.ok:
            raise AssertionError(
                f"Backend mismatch for kernel {result.kernel_name}:\n"
                f"Differences:\n{chr(10).join(result.differences[:20])}"
            )

    return tuple(results)


def print_comparison_summary(results: tuple[BackendComparisonResult, ...]) -> None:
    """Print summary of validation results.

    Args:
        results: Tuple of BackendComparisonResult to summarize.
    """
    total = len(results)
    matched = sum(1 for r in results if r.matches)
    normalized_matched = sum(1 for r in results if r.normalized_matches)
    text_verify_ok = sum(1 for r in results if r.text_verify_passed)
    pybind_verify_ok = sum(1 for r in results if r.pybind_verify_passed)
    pybind_available = sum(1 for r in results if r.pybind_available)

    print(f"Backend Validation Summary:")
    print(f"  Total kernels: {total}")
    print(f"  Pybind available: {pybind_available}/{total}")
    print(f"  Exact matches: {matched}/{total}")
    print(f"  Normalized matches: {normalized_matched}/{total}")
    print(f"  Text verify passed: {text_verify_ok}/{total}")
    print(f"  Pybind verify passed: {pybind_verify_ok}/{total}")

    if total > 0:
        print(f"  Match rate: {matched / total * 100:.1f}%")
        print(f"  Normalized match rate: {normalized_matched / total * 100:.1f}%")


__all__ = [
    "VerifyResult",
    "BackendComparisonResult",
    "compare_backends",
    "run_backend_validation_suite",
    "print_comparison_summary",
]