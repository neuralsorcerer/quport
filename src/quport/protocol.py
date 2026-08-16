# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Executable cat-entanglement circuits for an aggregation plan.

:mod:`quport.aggregation` decides *which* entanglement transactions a mapped
circuit needs. This module emits the circuits that actually perform them, so a
communication plan stops being an accounting artifact and becomes something a
simulator or a backend can run -- and, crucially, something whose correctness
can be checked rather than argued.

The cat-entanglement gadget
---------------------------
For a block with root ``r`` on QPU ``A`` and cat copy on QPU ``B``, with EPR
halves ``a`` (on ``A``) and ``b`` (on ``B``)::

    entangler:     h(a); cx(a, b); cx(r, a); cx(a, b)
    block gates:   every gate of the block, with r replaced by b
    disentangler:  h(b); cz(b, r)

This is the deferred-measurement form of the usual protocol: the ``measure a``
and ``if m: x(b)`` pair becomes ``cx(a, b)``, and ``measure b in X`` with
``if m: z(r)`` becomes ``h(b); cz(b, r)``. Writing it unitarily is what makes
the whole construction checkable with a state vector.

Tracing the algebra through, with :math:`|\\psi\\rangle=\\sum_z\\alpha_z|z\\rangle`:

.. math::

    |\\psi\\rangle_r|0\\rangle_a|0\\rangle_b
    \\;\\longrightarrow\\;
    \\Bigl(\\sum_z \\alpha_z |z\\rangle_r |z\\rangle_b\\Bigr)\\otimes|+\\rangle_a

after the entangler -- ``a`` factors out as :math:`|+\\rangle` and ``b`` carries
``r``'s computational-basis label -- and the disentangler returns ``b`` to
:math:`|+\\rangle` while leaving ``r`` holding the result. Both ancillas end in
a known product state, independent of the data, so an ``h`` restores them to
:math:`|0\\rangle` and they can be recycled by the next block.

That factorisation is exactly what fails when the root is touched
non-diagonally mid-block: ``r`` and ``b`` stay entangled, the disentangler
cannot separate them, and the emitted circuit computes something else.
:func:`verify_telegate_equivalence` detects that, which is what turns
:mod:`quport.entanglement`'s diagonality rule from a stated assumption into a
tested one.

Teleport blocks
---------------
Blocks that no cat copy can serve move the operand instead of copying it. The
emitted circuit shows the state movement as a ``swap`` in and out of the host's
ancilla, which is what teleportation achieves; the two e-bits it costs are
accounted for by the plan. QuPort does not expand the Bell-measurement gadget
itself, because the return trip needs a mid-circuit reset that would make the
program non-unitary and therefore unverifiable by the same route.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ClassicalRegister

from quport.aggregation import AggregationPlan, RemoteBlock, aggregate_remote_operations
from quport.architecture import MultiQPUArchitecture
from quport.entanglement import is_directive

__all__ = [
    "TelegateProgram",
    "build_telegate_circuit",
    "verify_telegate_equivalence",
]

#: Total qubit count above which state-vector verification is refused.
#: 2**24 amplitudes is already a gigabyte of complex128.
MAX_VERIFIABLE_QUBITS: int = 24


@dataclass(frozen=True)
class TelegateProgram:
    """A circuit that realises an aggregation plan with explicit entanglement.

    Attributes
    ----------
    circuit:
        The emitted circuit. Its first ``n_data`` qubits are the mapped
        circuit's physical qubits, in the same order; the rest are protocol
        ancillas that start and end in the ground state.
    n_data:
        Number of data qubits, i.e. the width of the input mapped circuit.
    ancillas:
        Qubit indices of the protocol ancillas within ``circuit``.
    blocks / epr_pairs:
        Blocks expanded and EPR pairs they consume, copied from the plan.
    unschedulable_gates:
        Cross-QPU gates the plan could not serve. They are emitted verbatim, so
        the circuit stays semantically faithful, but a real machine could not
        run them as written.
    measured:
        True when the circuit uses mid-circuit measurement and classical
        feedforward instead of the coherent form.
    """

    circuit: QuantumCircuit
    n_data: int
    ancillas: tuple[int, ...]
    blocks: int
    epr_pairs: int
    unschedulable_gates: int
    measured: bool

    @property
    def n_ancillas(self) -> int:
        """Number of protocol ancillas the expansion needed."""
        return len(self.ancillas)


class _AncillaPool:
    """Hand out ancilla qubit indices, reusing ones returned in ``|0>``."""

    __slots__ = ("_free", "_allocated", "_base")

    def __init__(self, base: int) -> None:
        self._base = base
        self._free: list[int] = []
        self._allocated = 0

    def acquire(self) -> int:
        if self._free:
            return self._free.pop()
        index = self._base + self._allocated
        self._allocated += 1
        return index

    def release(self, index: int) -> None:
        self._free.append(index)

    @property
    def allocated(self) -> int:
        return self._allocated


def _validate_plan(
    plan: AggregationPlan | None,
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    ports_per_qpu: int | Sequence[int] | None,
) -> AggregationPlan:
    if plan is None:
        return aggregate_remote_operations(mapped, arch, ports_per_qpu=ports_per_qpu)
    if not isinstance(plan, AggregationPlan):
        raise ValueError("plan must be an AggregationPlan")
    return plan


def build_telegate_circuit(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    plan: AggregationPlan | None = None,
    *,
    coherent: bool = True,
    ports_per_qpu: int | Sequence[int] | None = None,
) -> TelegateProgram:
    """Expand an aggregation plan into an executable circuit.

    Parameters
    ----------
    mapped:
        A circuit whose qubit indices are physical indices of ``arch``.
    arch:
        The architecture that defines the physical-to-QPU mapping.
    plan:
        A pre-computed plan; one is built from the architecture's own port
        budget when omitted.
    coherent:
        ``True`` (default) emits the deferred-measurement form: unitary, and
        therefore checkable by :func:`verify_telegate_equivalence`. ``False``
        emits real mid-circuit measurements with classical feedforward, which is
        what a backend executes and what OpenQASM 3 export should carry.
    ports_per_qpu:
        Forwarded to :func:`quport.aggregation.aggregate_remote_operations` when
        ``plan`` is omitted.

    Returns
    -------
    TelegateProgram

    Notes
    -----
    Ancillas are recycled between blocks, so the emitted width is driven by how
    many cat copies are live at once rather than by the block count. Both forms
    return every ancilla to the ground state, in the coherent form by an ``h``
    (each ancilla provably ends in :math:`|+\\rangle`) and in the measured form
    by an explicit ``reset``.
    """
    if not isinstance(mapped, QuantumCircuit):
        raise ValueError("mapped must be a QuantumCircuit")
    if not isinstance(arch, MultiQPUArchitecture):
        raise ValueError("arch must be a MultiQPUArchitecture")
    if not isinstance(coherent, bool):
        raise ValueError("coherent must be a boolean")

    resolved = _validate_plan(plan, mapped, arch, ports_per_qpu)

    n_data = len(mapped.qubits)
    qindex = {qubit: index for index, qubit in enumerate(mapped.qubits)}
    cindex = {clbit: index for index, clbit in enumerate(mapped.clbits)}

    starts: dict[int, list[RemoteBlock]] = {}
    ends: dict[int, list[RemoteBlock]] = {}
    members: dict[int, list[RemoteBlock]] = {}
    for block in resolved.blocks:
        starts.setdefault(block.start_index, []).append(block)
        ends.setdefault(block.end_index, []).append(block)
        for gate_index in block.gate_indices:
            members.setdefault(gate_index, []).append(block)

    # Two classical bits per cat block in the measured form; the coherent form
    # needs none. Widths are counted before the register is created because a
    # QuantumCircuit cannot grow a register mid-build without invalidating bits.
    cat_blocks = sum(1 for block in resolved.blocks if block.protocol == "cat")
    protocol_bits = 0 if coherent else 2 * cat_blocks

    # Worst case one ancilla per block; the pool recycles, so the real width is
    # discovered during the walk and the register is trimmed afterwards.
    max_ancillas = max(1, len(resolved.blocks) + 1)
    work = QuantumRegister(n_data, "q")
    ancilla_register = QuantumRegister(max_ancillas, "cat")
    circuit = QuantumCircuit(work, ancilla_register)
    if mapped.clbits:
        circuit.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        circuit.add_register(creg)
    protocol_register: ClassicalRegister | None = None
    if protocol_bits:
        protocol_register = ClassicalRegister(protocol_bits, "cat_c")
        circuit.add_register(protocol_register)

    pool = _AncillaPool(n_data)
    # Blocks are frozen dataclasses, but two teleport blocks emitted for one
    # wide gate can compare equal on everything except their root, so the map is
    # keyed on a tuple that includes it rather than on the block itself.
    _BlockKey = tuple[str, int, int, tuple[int, ...]]
    copy_of: dict[_BlockKey, int] = {}  # block identity -> ancilla holding the copy
    next_protocol_bit = 0

    def block_key(block: RemoteBlock) -> _BlockKey:
        return (block.protocol, block.root_phys, block.remote_qpu, block.gate_indices)

    def open_block(block: RemoteBlock) -> None:
        nonlocal next_protocol_bit
        root = block.root_phys
        copy = pool.acquire()
        copy_of[block_key(block)] = copy

        if block.protocol == "teleport":
            # The two e-bits move the operand; the circuit shows the move.
            circuit.swap(root, copy)
            return

        helper = pool.acquire()
        circuit.h(helper)
        circuit.cx(helper, copy)
        circuit.cx(root, helper)
        if coherent:
            circuit.cx(helper, copy)
            # The helper is provably left in |+>; return it to |0> and recycle.
            circuit.h(helper)
        else:
            assert protocol_register is not None
            bit = protocol_register[next_protocol_bit]
            next_protocol_bit += 1
            circuit.measure(helper, bit)
            with circuit.if_test((bit, 1)):
                circuit.x(copy)
            circuit.reset(helper)
        pool.release(helper)

    def close_block(block: RemoteBlock) -> None:
        nonlocal next_protocol_bit
        root = block.root_phys
        copy = copy_of.pop(block_key(block))

        if block.protocol == "teleport":
            circuit.swap(copy, root)
            pool.release(copy)
            return

        circuit.h(copy)
        if coherent:
            circuit.cz(copy, root)
            # The copy is provably left in |+>; return it to |0> and recycle.
            circuit.h(copy)
        else:
            assert protocol_register is not None
            bit = protocol_register[next_protocol_bit]
            next_protocol_bit += 1
            circuit.measure(copy, bit)
            with circuit.if_test((bit, 1)):
                circuit.z(root)
            circuit.reset(copy)
        pool.release(copy)

    for index, instruction in enumerate(mapped.data):
        operation = instruction.operation
        qubits = [qindex[qubit] for qubit in instruction.qubits]
        clbits = [circuit.clbits[cindex[clbit]] for clbit in instruction.clbits]

        if is_directive(operation) or not qubits:
            circuit.append(operation, [circuit.qubits[q] for q in qubits], clbits)
            continue

        for block in starts.get(index, ()):
            open_block(block)

        serving = members.get(index)
        if serving:
            substitution = {
                block.root_phys: copy_of[block_key(block)] for block in serving
            }
            operands = [substitution.get(qubit, qubit) for qubit in qubits]
        else:
            operands = qubits

        circuit.append(operation, [circuit.qubits[q] for q in operands], clbits)

        for block in ends.get(index, ()):
            close_block(block)

    used = pool.allocated
    trimmed = _trim_unused_ancillas(circuit, work, mapped, protocol_register, used)

    return TelegateProgram(
        circuit=trimmed,
        n_data=n_data,
        ancillas=tuple(range(n_data, n_data + used)),
        blocks=len(resolved.blocks),
        epr_pairs=resolved.epr_pairs,
        unschedulable_gates=resolved.unschedulable_gates,
        measured=not coherent,
    )


def _trim_unused_ancillas(
    circuit: QuantumCircuit,
    work: QuantumRegister,
    mapped: QuantumCircuit,
    protocol_register: ClassicalRegister | None,
    used: int,
) -> QuantumCircuit:
    """Rebuild ``circuit`` with only the ancillas the expansion actually used.

    The ancilla register has to be sized before the walk, but recycling means
    most of it usually goes untouched. Idle qubits would inflate every state
    vector by a factor of two each, so they are dropped rather than kept.
    """
    n_data = len(work)
    total = len(circuit.qubits)
    if used == total - n_data:
        return circuit

    ancillas = QuantumRegister(used, "cat")
    trimmed = QuantumCircuit(work, ancillas) if used else QuantumCircuit(work)
    if mapped.clbits:
        trimmed.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        trimmed.add_register(creg)
    if protocol_register is not None:
        trimmed.add_register(protocol_register)

    keep = n_data + used
    index_of = {qubit: position for position, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        positions = [index_of[qubit] for qubit in instruction.qubits]
        if any(position >= keep for position in positions):  # pragma: no cover
            raise RuntimeError("emitted circuit touched a trimmed ancilla")
        trimmed.append(
            instruction.operation,
            [trimmed.qubits[position] for position in positions],
            list(instruction.clbits),
        )
    return trimmed


def verify_telegate_equivalence(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    plan: AggregationPlan | None = None,
    *,
    seed: int = 0,
    atol: float = 1e-9,
    ports_per_qpu: int | Sequence[int] | None = None,
) -> bool:
    """Check that the emitted protocol circuit computes the mapped circuit.

    The coherent expansion is run on a pseudo-random product input, the
    ancillas are traced out, and the resulting state of the data qubits is
    compared with the mapped circuit's state on the same input. Returning
    ``True`` therefore certifies two things at once: the data come out right,
    **and** the ancillas are left unentangled from them -- residual entanglement
    would show up as a mixed reduced state and drive the fidelity below one.

    This is the empirical counterpart of :mod:`quport.entanglement`'s
    diagonality rule. Aggregating across an operation that breaks the rule sends
    the fidelity to zero rather than merely degrading it.

    Parameters
    ----------
    seed:
        Chooses the random product input state. Verification is deterministic
        for a given seed.
    atol:
        Tolerance on ``1 - fidelity``.

    Raises
    ------
    ValueError
        If the expanded circuit is too wide to simulate
        (:data:`MAX_VERIFIABLE_QUBITS`), or if the plan leaves gates
        unschedulable, which would make the comparison meaningless.
    """
    from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

    program = build_telegate_circuit(
        mapped, arch, plan, coherent=True, ports_per_qpu=ports_per_qpu
    )
    if program.unschedulable_gates:
        raise ValueError(
            "cannot verify a plan with unschedulable gates; give the "
            "architecture enough comm ports first"
        )

    total = len(program.circuit.qubits)
    if total > MAX_VERIFIABLE_QUBITS:
        raise ValueError(
            f"circuit has {total} qubits, above the {MAX_VERIFIABLE_QUBITS}-qubit "
            "state-vector verification limit"
        )

    preparation = _random_product_state(program.n_data, seed)

    protocol = preparation.copy()
    protocol = _widen(protocol, total)
    protocol.compose(program.circuit, qubits=range(total), inplace=True)

    reference = preparation.copy()
    reference.compose(mapped, qubits=range(program.n_data), inplace=True)

    ancillas = list(range(program.n_data, total))
    actual = Statevector(protocol)
    expected = Statevector(reference)
    if ancillas:
        actual = partial_trace(actual, ancillas)  # type: ignore[assignment]

    return bool(state_fidelity(actual, expected, validate=False) >= 1.0 - atol)


def _random_product_state(n_qubits: int, seed: int) -> QuantumCircuit:
    """Deterministic single-qubit rotations covering the whole Bloch sphere.

    A product state is enough: the protocol is linear, so agreeing on a
    spanning set of inputs is agreement everywhere, and the angles below give
    every qubit a generic state with non-zero amplitude on both basis vectors.
    """
    if type(seed) is bool or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    circuit = QuantumCircuit(QuantumRegister(n_qubits, "q"))
    for qubit in range(n_qubits):
        step = (seed * 7 + qubit * 13 + 1) % 17
        circuit.ry(0.31 + step * math.pi / 11.0, qubit)
        circuit.rz(0.17 + step * math.pi / 13.0, qubit)
    return circuit


def _widen(circuit: QuantumCircuit, total: int) -> QuantumCircuit:
    """Return ``circuit`` padded with idle qubits up to ``total`` width."""
    current = len(circuit.qubits)
    if current == total:
        return circuit
    widened = QuantumCircuit(QuantumRegister(total, "q"))
    widened.compose(circuit, qubits=range(current), inplace=True)
    return widened
