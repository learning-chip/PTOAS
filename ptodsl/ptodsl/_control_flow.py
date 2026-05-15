# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Control-flow context managers and functional helpers for PTO kernels.

All helpers work with the current MLIR insertion point; no context threading needed.

Public API
──────────
``vecscope()``              – ``pto.vecscope { … }``
``for_(lo, hi, *, step)``   – simple ``scf.for`` (yields induction variable)
``if_(cond)``               – simple ``scf.if`` without results
``reduce(lo, hi, *, step, init, fn)``
                            – ``scf.for`` with iter_args expressed as a fold:
                              ``fn(iv, *state) → new_state``
``cond(condition, then_, else_)``
                            – ``scf.if`` with results expressed as a conditional:
                              each branch is a zero-arg callable returning values
``yield_(*vals)``           – ``scf.yield``
"""

from ._bootstrap import make_context  # noqa: F401
from ._types import _resolve

from mlir.dialects import pto as _pto, scf
from mlir.ir import InsertionPoint


# ── vecscope ──────────────────────────────────────────────────────────────────

class _VecScopeCM:
    """Context manager for ``pto.vecscope { … }``."""

    def __enter__(self):
        self._op = _pto.VecScopeOp()
        self._block = self._op.body.blocks.append()
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()
        return None

    def __exit__(self, *exc):
        self._ip.__exit__(*exc)


def vecscope() -> _VecScopeCM:
    """Return a context manager that emits ``pto.vecscope { … }``."""
    return _VecScopeCM()


# ── for_ (simple loop) ────────────────────────────────────────────────────────

class _ForCM:
    def __init__(self, start, stop, step):
        self._start = start
        self._stop = stop
        self._step = step
        self._for_op = None
        self._ip = None

    def __enter__(self):
        self._for_op = scf.ForOp(self._start, self._stop, self._step)
        self._ip = InsertionPoint(self._for_op.body)
        self._ip.__enter__()
        return self._for_op.induction_variable

    def __exit__(self, *exc):
        scf.YieldOp([])
        self._ip.__exit__(*exc)


def for_(start, stop, *, step) -> _ForCM:
    """
    Simple ``scf.for`` context manager.  Yields the induction variable;
    ``scf.yield`` is inserted automatically on exit::

        with pto.for_(c0, c16, step=c1) as i:
            off = pto.scalar.muli(i, c64)
            ...

    For loops that carry loop-carried values (iter_args), use
    :func:`reduce` instead.
    """
    return _ForCM(start, stop, step)


# ── fori_loop – scf.for with iter_args as a functional fold ──────────────────

def fori_loop(lower, upper, body_fun, init_val, *, step):
    """
    ``scf.for`` with iter_args, modelled after ``jax.lax.fori_loop``.

    Signature mirrors JAX::

        fori_loop(lower, upper, body_fun, init_val, *, step)

    ``body_fun(iv, *state) → new_state`` is called once with the InsertionPoint
    set to the loop body; the returned values become the ``scf.yield`` operands.
    ``init_val`` is the initial loop-carried state (a tuple or single value).

    Returns the final state after the loop (tuple, or a single value if
    ``init_val`` was a single value)::

        final_max, final_sum = pto.fori_loop(
            c0, c128, chunk_step, (oldmax_bc, oldsum_bc), step=c64
        )
        # where: def chunk_step(chunk, rmax, rsum): … return new_max, new_sum
    """
    init_vals = list(init_val) if isinstance(init_val, (tuple, list)) else [init_val]
    for_op = scf.ForOp(lower, upper, step, init_vals)
    with InsertionPoint(for_op.body):
        iv    = for_op.induction_variable
        state = tuple(for_op.inner_iter_args)
        new_state = body_fun(iv, *state)
        new_state = list(new_state) if isinstance(new_state, (tuple, list)) else [new_state]
        scf.YieldOp(new_state)
    results = tuple(for_op.results)
    return results if len(results) != 1 else results[0]


# ── if_ (simple conditional, no results) ─────────────────────────────────────

class _IfCM:
    def __init__(self, cond_val):
        self._cond = cond_val
        self._if_op = None
        self._ip = None

    def __enter__(self):
        self._if_op = scf.IfOp(self._cond)
        self._ip = InsertionPoint(self._if_op.then_block)
        self._ip.__enter__()
        return None

    def __exit__(self, *exc):
        scf.YieldOp([])
        self._ip.__exit__(*exc)


def if_(cond_val) -> _IfCM:
    """
    Simple ``scf.if`` without results.  ``scf.yield`` is inserted
    automatically on exit::

        with pto.if_(has_rows):
            pto.tload(part, tile)
            ...

    For conditionals that produce values, use :func:`cond` instead.
    """
    return _IfCM(cond_val)


# ── cond – scf.if with results as a conditional expression ───────────────────

def _to_list(vals):
    """Normalise a single value or a tuple/list to a list."""
    return list(vals) if isinstance(vals, (tuple, list)) else [vals]


def cond(condition, then_, else_):
    """
    ``scf.if`` with results expressed as a conditional expression.

    ``then_()`` and ``else_()`` are zero-arg callables that emit MLIR ops
    and return their result values (tuple or single value).

    **Type inference**: ``else_`` is called *once* outside any block before
    the IfOp is created to discover the result types.  ``else_`` must
    therefore be non-emitting at that call (the typical ``lambda: (a, b)``
    pattern satisfies this automatically).  Both branches are then called
    properly inside their respective blocks.

    Usage::

        next_max, next_sum = pto.cond(
            has_chunk,
            then_=compute_chunk,              # def that emits ops
            else_=lambda: (rmax, rsum),       # trivial: just passes state through
        )
    """
    # ── Type inference pass (else_ must not emit ops here) ──
    hint       = _to_list(else_())
    res_types  = [v.type for v in hint]

    # ── Create the IfOp at the current insertion point ──────
    if_op = scf.IfOp(condition, res_types, hasElse=True)

    # ── Fill the then-block ──────────────────────────────────
    with InsertionPoint(if_op.then_block):
        scf.YieldOp(_to_list(then_()))

    # ── Fill the else-block ──────────────────────────────────
    with InsertionPoint(if_op.else_block):
        scf.YieldOp(_to_list(else_()))

    results = tuple(if_op.results)
    return results if len(results) != 1 else results[0]


# ── yield_ ────────────────────────────────────────────────────────────────────

def yield_(*vals):
    """Emit ``scf.yield`` with the given values."""
    scf.YieldOp(list(vals))


__all__ = [
    "vecscope",
    "for_", "fori_loop",
    "if_", "cond",
    "yield_",
]
