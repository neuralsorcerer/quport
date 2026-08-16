# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Computational-basis (Z) diagonality analysis for distributed quantum gates.

Why this exists
---------------
Remote (inter-QPU) two-qubit gates in a distributed quantum computer are not
executed by moving quantum state across the network. The standard construction
is *cat-entanglement* (also called telegate, or "cat-entangler/cat-disentangler"):

1. Distribute one EPR pair between QPU ``A`` (holding the root qubit ``c``) and
   QPU ``B``.
2. ``A`` applies ``CX(c -> a)`` on its half of the pair, measures ``a`` in the
   Z basis, and sends the outcome to ``B``, which applies a conditional ``X``.
   ``B``'s half ``b`` now carries a *cat copy* of ``c``: the joint state is

   .. math:: \\sum_z \\alpha_z |z\\rangle_c |z\\rangle_b \\otimes |\\psi_z\\rangle.

3. Every gate that uses ``c`` only through its computational-basis label can now
   be run **locally on B** against ``b``.
4. ``B`` measures ``b`` in the X basis and sends the outcome back, and ``A``
   applies a conditional ``Z`` to ``c`` (cat-disentangler).

The correctness condition for step 3 is exactly:

    every operation applied to ``c`` while the cat copy is live must commute
    with :math:`Z_c`.

If that holds, the operation maps :math:`|z\\rangle_c \\otimes |\\psi\\rangle` to
:math:`|z\\rangle_c \\otimes U_z|\\psi\\rangle`, so the ``c``/``b`` label
correspondence survives and the disentangler restores ``c`` exactly. If it does
not hold (an ``X``, ``H``, ``SX``, or a gate that uses ``c`` as a CX *target*),
the cat copy must be released first.

This module answers one question, rigorously and conservatively: **which operand
positions of an operation act diagonally in the computational basis?** Every
other entanglement-aware component in QuPort (packet construction, communication
aggregation, e-bit costing, entanglement scheduling) is built on top of it, so
a single shared and conservative answer keeps them consistent by construction.

Conservatism
------------
An operation whose diagonality QuPort cannot establish is reported as acting
non-diagonally on every operand. That can only *over*-count entanglement
requirements; it can never claim a cat copy survives an operation that would
destroy it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from qiskit.circuit import ControlledGate, Operation

__all__ = [
    "DIAGONAL_1Q_GATES",
    "DIAGONAL_QARG_POSITIONS",
    "acts_diagonally",
    "breaks_cat_copy",
    "diagonal_operands",
    "diagonal_positions",
    "is_directive",
]

#: Single-qubit operations that commute with ``Z`` (diagonal in the
#: computational basis, up to an irrelevant global phase).
DIAGONAL_1Q_GATES: frozenset[str] = frozenset(
    {
        "id",
        "i",
        "delay",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "rz",
        "p",
        "u1",
    }
)

#: Operand positions that act diagonally, for operations whose structure cannot
#: be derived from :class:`~qiskit.circuit.ControlledGate` metadata.
#:
#: ``rzz(theta) = exp(-i theta/2 Z (x) Z)`` is diagonal on both operands;
#: ``rzx(theta) = exp(-i theta/2 Z (x) X)`` is diagonal on operand 0 only.
#: Explicitly listing the non-diagonal two-qubit gates keeps the intent visible
#: and documents that the omission is deliberate rather than an oversight.
DIAGONAL_QARG_POSITIONS: dict[str, tuple[int, ...]] = {
    "rzz": (0, 1),
    "rzx": (0,),
    "rxx": (),
    "ryy": (),
    "swap": (),
    "iswap": (),
    "dcx": (),
    "ecr": (),
    # Relative-phase Toffolis are not ControlledGate instances, but their
    # unitaries do commute with Z on every control operand.
    "rccx": (0, 1),
    "rcccx": (0, 1, 2),
    "measure": (),
    "reset": (),
    "initialize": (),
}


def is_directive(operation: Any) -> bool:
    """Return True for compiler directives (barriers) that do not act on state.

    Directives consume no runtime resources and cannot disturb a live cat copy,
    which is why every QuPort estimator skips them with this same check.
    """
    return bool(getattr(operation, "_directive", False))


def _num_qubits(operation: Any) -> int:
    """Return an operation's operand count, defaulting to zero when unknown."""
    num_qubits = getattr(operation, "num_qubits", 0)
    if type(num_qubits) is bool or not isinstance(num_qubits, int):
        return 0
    return max(0, num_qubits)


def diagonal_positions(operation: Operation | Any) -> frozenset[int]:
    """Return the operand positions on which ``operation`` commutes with ``Z``.

    A position ``i`` is included when the operation's unitary ``U`` satisfies
    ``[U, Z_i] = 0``, i.e. when ``U`` cannot change the computational-basis
    label of operand ``i``. Those are exactly the operands that may serve as, or
    coexist with, a live cat copy.

    The result is derived from three rules, in order:

    1. An explicit entry in :data:`DIAGONAL_QARG_POSITIONS`.
    2. :class:`~qiskit.circuit.ControlledGate` structure. A controlled gate is
       block diagonal in its control basis, so all ``num_ctrl_qubits`` control
       operands are diagonal regardless of ``ctrl_state``; and because
       ``C(U) = P_0 (x) I + P_1 (x) U``, every operand that is diagonal for the
       base gate stays diagonal once controlled.
    3. :data:`DIAGONAL_1Q_GATES` for single-qubit operations.

    Anything else is reported as non-diagonal on every operand.
    """
    if is_directive(operation):
        # A barrier applies no unitary, so it trivially commutes with every Z.
        return frozenset(range(_num_qubits(operation)))

    name = getattr(operation, "name", None)
    if isinstance(name, str):
        explicit = DIAGONAL_QARG_POSITIONS.get(name)
        if explicit is not None:
            return frozenset(explicit)

    if isinstance(operation, ControlledGate):
        num_ctrl = operation.num_ctrl_qubits
        if type(num_ctrl) is bool or not isinstance(num_ctrl, int) or num_ctrl < 0:
            return frozenset()
        positions = set(range(num_ctrl))
        base_gate = getattr(operation, "base_gate", None)
        if base_gate is not None:
            positions.update(
                num_ctrl + position for position in diagonal_positions(base_gate)
            )
        return frozenset(positions)

    if _num_qubits(operation) == 1 and isinstance(name, str):
        if name in DIAGONAL_1Q_GATES:
            return frozenset({0})

    return frozenset()


def acts_diagonally(operation: Operation | Any, position: int) -> bool:
    """Return True when ``operation`` commutes with ``Z`` on operand ``position``."""
    if type(position) is bool or not isinstance(position, int):
        raise ValueError("position must be an integer operand index")
    return position in diagonal_positions(operation)


def diagonal_operands(
    operation: Operation | Any, qubits: Sequence[int]
) -> tuple[int, ...]:
    """Return the entries of ``qubits`` whose operand position is diagonal."""
    positions = diagonal_positions(operation)
    return tuple(qubit for index, qubit in enumerate(qubits) if index in positions)


def breaks_cat_copy(operation: Operation | Any, positions: Iterable[int]) -> bool:
    """Return True when ``operation`` disturbs a cat copy of any listed operand.

    ``positions`` are operand indices of the operation that currently host, or
    are mirrored by, a live cat copy. The copy survives only while every one of
    them is acted on diagonally.
    """
    diagonal = diagonal_positions(operation)
    return any(position not in diagonal for position in positions)
