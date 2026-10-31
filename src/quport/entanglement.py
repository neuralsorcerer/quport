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
from functools import cache
from typing import Any

import numpy as np
from qiskit.circuit import ControlledGate, Delay, Operation, QuantumCircuit

__all__ = [
    "DIAGONAL_1Q_GATES",
    "DIAGONAL_QARG_POSITIONS",
    "acts_diagonally",
    "breaks_cat_copy",
    "diagonal_operands",
    "diagonal_positions",
    "is_directive",
    "is_standard_gate",
    "is_standard_namesake",
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

#: Widest controlled gate outside Qiskit's library whose unitary is built to
#: find its diagonal operands; a wider one is judged gate by gate.
_MAX_UNITARY_QUBITS = 6


def is_directive(operation: Any) -> bool:
    """Return True for compiler directives (barriers) that do not act on state.

    Directives consume no runtime resources and cannot disturb a live cat copy,
    which is why every QuPort estimator skips them with this same check.
    """
    return bool(getattr(operation, "_directive", False))


@cache
def _standard_gates_by_name() -> dict[str, Any]:
    """Qiskit's standard gates, as the ``StandardGate`` mark each name carries."""
    from qiskit.circuit.library import get_standard_gate_name_mapping

    return {
        name: gate._standard_gate
        for name, gate in get_standard_gate_name_mapping().items()
        if getattr(gate, "_standard_gate", None) is not None
    }


def is_standard_gate(operation: Any, name: str | None = None) -> bool:
    """True when ``operation`` is Qiskit's own standard gate of its name.

    Qiskit marks its standard gates, and the mark must agree with the name:
    an ``X`` copied under the name ``t`` is still an ``X``. With ``name``, the
    operation must also be that gate.
    """
    standard = getattr(operation, "_standard_gate", None)
    own = getattr(operation, "name", None)
    if standard is None or not isinstance(own, str):
        return False
    if name is not None and own != name:
        return False
    return bool(_standard_gates_by_name().get(own) == standard)


@cache
def _standard_non_gates_by_name() -> dict[str, type]:
    """Qiskit's standard operations that carry no gate mark, by their class.

    ``measure``, ``reset`` and ``delay`` are standard but not gates, so they
    are recognised by class, as the simulator recognises them.
    """
    from qiskit.circuit.library import get_standard_gate_name_mapping

    return {
        name: operation.base_class
        for name, operation in get_standard_gate_name_mapping().items()
        if getattr(operation, "_standard_gate", None) is None
    }


def is_standard_namesake(operation: Any) -> bool:
    """True for an operation named like a Qiskit standard one without being it.

    Names are a lookup key, not an identity: a custom gate called ``t`` whose
    definition is an ``X`` is no ``T`` gate, and neither is Qiskit's ``X``
    copied under that name. Nor is a gate called ``measure`` a measurement,
    though passes that match by name treat it as one.
    """
    name = getattr(operation, "name", None)
    if not isinstance(name, str):
        return False
    kind = _standard_non_gates_by_name().get(name)
    if kind is not None:
        return not isinstance(operation, kind)
    return name in _standard_gates_by_name() and not is_standard_gate(operation)


def _is_named_operation(operation: Any) -> bool:
    """True when ``operation`` is Qiskit's own operation of its name.

    Only then can a table keyed by name speak for it. ``delay`` is not a gate,
    so it is recognised by its class.
    """
    return is_standard_gate(operation) or (
        isinstance(operation, Delay) and operation.name == "delay"
    )


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

    1. An explicit entry in :data:`DIAGONAL_QARG_POSITIONS`, for Qiskit's own
       gate of that name -- a custom gate that merely shares the name gets
       none of the tables' entries.
    2. :class:`~qiskit.circuit.ControlledGate` structure, for Qiskit's own
       controlled gates. A controlled gate is block diagonal in its control
       basis, so all ``num_ctrl_qubits`` control operands are diagonal
       regardless of ``ctrl_state``; and because
       ``C(U) = P_0 (x) I + P_1 (x) U``, every operand that is diagonal for the
       base gate stays diagonal once controlled. Any other controlled gate --
       one built by :meth:`~qiskit.circuit.Gate.control`, or by hand from a
       definition its ``base_gate`` need not describe -- is judged by what it
       does: by its unitary when it has at most six qubits and bound
       parameters, otherwise gate by gate through its definition.
    3. :data:`DIAGONAL_1Q_GATES` for Qiskit's own single-qubit operations.

    Anything else is reported as non-diagonal on every operand.
    """
    if is_directive(operation):
        # A barrier applies no unitary, so it trivially commutes with every Z.
        return frozenset(range(_num_qubits(operation)))

    name = getattr(operation, "name", None)
    named = _is_named_operation(operation)
    if isinstance(name, str) and named:
        explicit = DIAGONAL_QARG_POSITIONS.get(name)
        if explicit is not None:
            return frozenset(explicit)

    if isinstance(operation, ControlledGate):
        if not _is_qiskit_operation(operation):
            # Anyone can build a ControlledGate from any definition, and the
            # definition is what runs: its base gate is only a label.
            return _definition_diagonal_positions(operation)
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

    if _num_qubits(operation) == 1 and isinstance(name, str) and named:
        if name in DIAGONAL_1Q_GATES:
            return frozenset({0})

    return frozenset()


def _is_qiskit_operation(operation: Any) -> bool:
    """True for Qiskit's standard gates and the classes of its circuit library.

    Their definitions are built from what they are, so their structure can be
    read off them. Qiskit's singleton classes carry no module, and are all
    standard gates.
    """
    module = type(operation).__module__ or ""
    return is_standard_gate(operation) or module.startswith("qiskit.circuit.library")


def _definition_diagonal_positions(operation: Any) -> frozenset[int]:
    """Diagonal operands of an operation, from what its definition does.

    A narrow enough operation with bound parameters is judged by its unitary,
    exactly. Otherwise an operand is diagonal when every gate of the definition
    acts on it diagonally, since a product of operators that each commute with
    ``Z`` commutes with it too. An operation whose unitary and definition both
    fail to build, however they fail, is reported non-diagonal everywhere.
    """
    num_qubits = _num_qubits(operation)
    if num_qubits <= _MAX_UNITARY_QUBITS:
        from qiskit.quantum_info import Operator

        try:
            unitary = Operator(operation).data
        except Exception:  # noqa: BLE001 - a gate that cannot be built is judged below
            pass
        else:
            labels = np.arange(2**num_qubits)
            positions = set()
            for position in range(num_qubits):
                bit = (labels >> position) & 1
                # U commutes with Z on an operand exactly when it never maps a
                # basis state to one with that operand's bit flipped.
                if np.allclose(unitary[bit[:, None] != bit[None, :]], 0.0, atol=1e-10):
                    positions.add(position)
            return frozenset(positions)

    try:
        definition = operation.definition
    except Exception:  # noqa: BLE001 - no definition to read: judged non-diagonal
        return frozenset()
    if not isinstance(definition, QuantumCircuit):
        return frozenset()
    if definition.num_qubits != num_qubits:
        return frozenset()
    index = {qubit: position for position, qubit in enumerate(definition.qubits)}
    disturbed: set[int] = set()
    for instruction in definition.data:
        diagonal = diagonal_positions(instruction.operation)
        disturbed.update(
            index[qubit]
            for position, qubit in enumerate(instruction.qubits)
            if position not in diagonal
        )
    return frozenset(set(range(num_qubits)) - disturbed)


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
