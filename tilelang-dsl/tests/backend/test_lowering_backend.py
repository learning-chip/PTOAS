# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for lowering_backend abstraction (Issue #237 Phase 1).

Can run without pytest:
    python3 tests/backend/test_lowering_backend.py
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from tilelang_dsl.lowering_backend import (
    LoweringBackend,
    LoweringResult,
    TextBackend,
    PybindBackend,
    get_backend,
    lower_with_backend,
    _SUPPORTED_BACKENDS,
)

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def test_lowering_result_empty():
    """Empty result should raise ValueError."""
    print("Test: LoweringResult empty")

    result = LoweringResult()
    assert not result, "Empty result should be falsy"

    try:
        result.as_text()
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "no output" in str(e)

    try:
        result.as_module()
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "no output" in str(e)

    print("  [PASS]")
    return True


def test_lowering_result_text():
    """Result with text only."""
    print("Test: LoweringResult text")

    text = "module {}"
    result = LoweringResult(text=text)
    assert result.as_text() == text, "as_text() returns text"
    assert result.text == text, "text attribute matches"
    assert result.module is None, "module is None"

    print("  [PASS]")
    return True


def test_lowering_result_bool():
    """Boolean conversion."""
    print("Test: LoweringResult bool conversion")

    empty = LoweringResult()
    assert not empty, "Empty is falsy"

    with_text = LoweringResult(text="module {}")
    assert with_text, "With text is truthy"

    with_module = LoweringResult(module=object())
    assert with_module, "With module is truthy"

    print("  [PASS]")
    return True


def test_lowering_result_str():
    """String conversion."""
    print("Test: LoweringResult str conversion")

    result = LoweringResult(text="module {}")
    assert str(result) == "module {}", "str() returns text"

    print("  [PASS]")
    return True


def test_lowering_backend_abstract():
    """LoweringBackend cannot be instantiated directly."""
    print("Test: LoweringBackend abstract")

    try:
        LoweringBackend()
        assert False, "Should raise TypeError"
    except TypeError:
        pass

    print("  [PASS]")
    return True


def test_text_backend_instantiation():
    """TextBackend can be instantiated."""
    print("Test: TextBackend instantiation")

    backend = TextBackend()
    assert backend.name() == "text", "name() returns 'text'"
    assert "text" in repr(backend), "repr contains 'text'"

    print("  [PASS]")
    return True


def test_text_backend_has_lower():
    """TextBackend has lower method."""
    print("Test: TextBackend has lower")

    backend = TextBackend()
    assert hasattr(backend, "lower"), "Has lower method"

    print("  [PASS]")
    return True


def test_pybind_backend_instantiation():
    """PybindBackend can be instantiated."""
    print("Test: PybindBackend instantiation")

    backend = PybindBackend()
    assert backend.name() == "pybind", "name() returns 'pybind'"
    assert "pybind" in repr(backend), "repr contains 'pybind'"

    print("  [PASS]")
    return True


def test_pybind_backend_is_available():
    """is_available() returns bool."""
    print("Test: PybindBackend is_available")

    backend = PybindBackend()
    available = backend.is_available()
    assert isinstance(available, bool), "is_available() returns bool"
    print(f"  is_available() = {available}")

    print("  [PASS]")
    return True


def test_pybind_backend_lower():
    """lower() behavior depends on availability."""
    print("Test: PybindBackend lower")

    backend = PybindBackend()
    if not backend.is_available():
        try:
            backend.lower(None)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            print("  (NotImplementedError raised as expected)")
    else:
        print("  (backend available, skipping NotImplementedError test)")

    print("  [PASS]")
    return True


def test_get_backend_text():
    """get_backend('text') returns TextBackend."""
    print("Test: get_backend('text')")

    backend = get_backend("text")
    assert isinstance(backend, TextBackend), "Returns TextBackend"
    assert backend.name() == "text", "name() is 'text'"

    print("  [PASS]")
    return True


def test_get_backend_pybind():
    """get_backend('pybind') returns PybindBackend."""
    print("Test: get_backend('pybind')")

    backend = get_backend("pybind")
    assert isinstance(backend, PybindBackend), "Returns PybindBackend"
    assert backend.name() == "pybind", "name() is 'pybind'"

    print("  [PASS]")
    return True


def test_get_backend_unknown():
    """Unknown backend name raises ValueError."""
    print("Test: get_backend('unknown')")

    try:
        get_backend("unknown")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown backend" in str(e), "Error message contains 'Unknown backend'"

    print("  [PASS]")
    return True


def test_get_backend_default():
    """Default backend is text."""
    print("Test: get_backend() default")

    backend = get_backend()
    assert backend.name() == "text", "Default is 'text'"

    print("  [PASS]")
    return True


def test_supported_backends():
    """Supported backends are text and pybind."""
    print("Test: _SUPPORTED_BACKENDS")

    assert "text" in _SUPPORTED_BACKENDS, "'text' in supported"
    assert "pybind" in _SUPPORTED_BACKENDS, "'pybind' in supported"
    assert len(_SUPPORTED_BACKENDS) == 2, "Exactly 2 backends"

    print("  [PASS]")
    return True


def test_lower_with_backend_callable():
    """lower_with_backend is callable."""
    print("Test: lower_with_backend callable")

    assert callable(lower_with_backend), "lower_with_backend is callable"

    print("  [PASS]")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Lowering Backend Tests")
    print("=" * 60)

    tests = [
        test_lowering_result_empty,
        test_lowering_result_text,
        test_lowering_result_bool,
        test_lowering_result_str,
        test_lowering_backend_abstract,
        test_text_backend_instantiation,
        test_text_backend_has_lower,
        test_pybind_backend_instantiation,
        test_pybind_backend_is_available,
        test_pybind_backend_lower,
        test_get_backend_text,
        test_get_backend_pybind,
        test_get_backend_unknown,
        test_get_backend_default,
        test_supported_backends,
        test_lower_with_backend_callable,
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