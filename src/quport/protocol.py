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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ClassicalRegister, ControlFlowOp

from quport.aggregation import AggregationPlan, RemoteBlock, aggregate_remote_operations
from quport.architecture import MultiQPUArchitecture
from quport.distributed import RemoteOp, reassemble_distributed_program
from quport.entanglement import is_directive

__all__ = [
    "TelegateProgram",
    "build_telegate_circuit",
    "verify_distributed_program",
    "verify_telegate_equivalence",
]

#: Total qubit count above which state-vector verification is refused.
#: 2**24 amplitudes is 256 MiB of complex128 per state vector, and a check
#: holds a few at once.
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
    ``True`` therefore establishes two things at once for that input: the data
    come out right, **and** the ancillas are left unentangled from them --
    residual entanglement would show up as a mixed reduced state and drive the
    fidelity below one. One generic input is strong evidence for every input,
    not a proof of it; see :func:`_random_product_state`.

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
    import numpy as np
    from qiskit.quantum_info import Statevector

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

    unitary_mapped = _unitary_part(mapped, label="the mapped circuit")
    unitary_program = _unitary_part(program.circuit, label="the emitted protocol")

    preparation = _random_product_state(program.n_data, seed)

    protocol = _widen(preparation.copy(), total)
    protocol.compose(unitary_program, qubits=range(total), inplace=True)

    reference = preparation.copy()
    reference.compose(unitary_mapped, qubits=range(program.n_data), inplace=True)

    # Fidelity of the data qubits' reduced state with the expected one,
    # <psi|rho|psi> = sum over ancilla basis states a of |<psi, a|phi>|^2.
    # Summing the overlaps directly gives the same number as tracing the
    # ancillas out, without building the reduced density matrix, whose
    # 4**n_data entries outgrow memory long before the state vector does
    # (16 GiB at 15 data qubits). Qubits are little-endian and the ancillas
    # come after the data, so a row of the reshaped vector fixes the ancillas
    # and runs over the data.
    actual = Statevector(protocol).data
    expected = Statevector(reference).data
    by_ancilla = actual.reshape(2 ** (total - program.n_data), 2**program.n_data)
    overlaps = by_ancilla @ expected.conj()
    fidelity = float(np.vdot(overlaps, overlaps).real)
    return bool(fidelity >= 1.0 - atol)


def verify_distributed_program(
    mapped: QuantumCircuit,
    local_routed: Mapping[int, QuantumCircuit],
    remote_ops: Sequence[RemoteOp],
    arch: MultiQPUArchitecture,
    *,
    seed: int = 0,
    atol: float = 1e-9,
) -> bool:
    """Check that a distributed compile's artifacts still compute their circuit.

    This is the central claim of distributed compilation: the per-QPU programs
    plus the remote-operation manifest, taken together, are the circuit they
    were split from. It is checked by
    :func:`quport.distributed.reassemble_distributed_program` -- which merges
    them back under qubit and classical-bit dataflow and undoes each QPU's
    routing permutation --
    and comparing the result against the mapped circuit on a pseudo-random
    product input.

    Unlike :func:`verify_telegate_equivalence`, this says nothing about how
    remote gates are physically realised; it checks the splitting, the local
    routing, and the manifest that ties them together.

    Terminating measurements are compared by what they record, not by the
    state they read: each one is swapped into an ancilla standing for its
    classical bit, and those ancillas are dephased before the comparison. The
    pre-measurement state is not something the programs promise to keep --
    from ``optimization_level`` 2 Qiskit drops diagonal gates right before a
    measurement and re-targets the classical bit of a measurement that follows
    a ``swap`` -- and either would fail a comparison of raw state vectors
    although no outcome changes. Comparing outcomes per classical bit also
    checks which bit each measurement writes, which dropping the measurements
    could not.

    Parameters
    ----------
    remote_ops:
        The manifest matching ``local_routed``: ``routed_remote_ops`` for routed
        programs, ``program.remote_ops`` for unrouted ones.

    Raises
    ------
    ValueError
        If the circuit is too wide to simulate -- its qubits plus one ancilla
        per measured classical bit exceed :data:`MAX_VERIFIABLE_QUBITS` -- or if
        the artifacts are inconsistent enough that they cannot be merged at all.
    """
    from qiskit.quantum_info import Statevector

    merged = reassemble_distributed_program(
        mapped, local_routed, remote_ops, arch, restore_layout=True
    )
    # Refuses mid-circuit measurement, reset and classical control, so every
    # measurement left in `mapped` -- and so in `merged` -- is terminating.
    _unitary_part(mapped, label="the mapped circuit")

    width = len(merged.qubits)
    outcome_slots = _outcome_slots(width, mapped, merged)
    total = width + len(outcome_slots)
    if total > MAX_VERIFIABLE_QUBITS:
        raise ValueError(
            f"circuit has {width} qubits and {len(outcome_slots)} measured classical "
            f"bits, above the {MAX_VERIFIABLE_QUBITS}-qubit state-vector "
            "verification limit"
        )

    preparation = _widen(_random_product_state(width, seed), total)

    actual = preparation.copy()
    actual.compose(
        _measurements_into_ancillas(merged, total, outcome_slots),
        qubits=range(total),
        inplace=True,
    )

    expected = preparation.copy()
    expected.compose(
        _measurements_into_ancillas(mapped, total, outcome_slots),
        qubits=range(total),
        inplace=True,
    )

    fidelity = _dephased_fidelity(
        Statevector(actual).data, Statevector(expected).data, n_data=width
    )
    return bool(fidelity >= 1.0 - atol)


_NON_UNITARY_OPS = frozenset({"measure", "reset", "initialize"})


def _unitary_part(circuit: QuantumCircuit, *, label: str) -> QuantumCircuit:
    """Return ``circuit`` without its terminating measurements.

    Verification compares state vectors, so it can only speak about the state a
    circuit prepares. Measurements that come last are dropped -- they read that
    state out without changing it -- while a measurement or reset that other
    operations depend on genuinely changes what the circuit computes, and is
    refused rather than quietly ignored. So is classical control: an operation
    conditioned on a classical bit, or any other control-flow block, has no
    state-vector evolution to compare.
    """
    trimmed = circuit.remove_final_measurements(inplace=False)
    if trimmed is None:  # pragma: no cover - defensive across Qiskit versions
        trimmed = circuit
    surviving = {
        instruction.operation.name
        for instruction in trimmed.data
        if instruction.operation.name in _NON_UNITARY_OPS
    }
    if surviving:
        raise ValueError(
            f"cannot verify {label}: it contains mid-circuit "
            f"{', '.join(sorted(surviving))}, which state-vector comparison "
            "cannot represent"
        )
    controlled = {
        instruction.operation.name
        for instruction in trimmed.data
        if instruction.clbits or isinstance(instruction.operation, ControlFlowOp)
    }
    if controlled:
        raise ValueError(
            f"cannot verify {label}: it contains classically controlled "
            f"{', '.join(sorted(controlled))}, which state-vector comparison "
            "cannot represent"
        )
    return trimmed


def _random_product_state(n_qubits: int, seed: int) -> QuantumCircuit:
    """Deterministic single-qubit rotations giving every qubit a generic state.

    The angles give each qubit non-zero amplitude on both basis vectors, so the
    input overlaps every computational basis state. That makes agreement on it
    a strong check rather than a proof: one input is not a spanning set, and two
    circuits can agree on a particular state while differing elsewhere. A wrong
    protocol passes only if it happens to map this one generic input correctly.
    """
    if type(seed) is bool or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    circuit = QuantumCircuit(QuantumRegister(n_qubits, "q"))
    for qubit in range(n_qubits):
        step = (seed * 7 + qubit * 13 + 1) % 17
        circuit.ry(0.31 + step * math.pi / 11.0, qubit)
        circuit.rz(0.17 + step * math.pi / 13.0, qubit)
    return circuit


def _outcome_slots(width: int, *circuits: QuantumCircuit) -> dict[Any, int]:
    """Give every classical bit any of ``circuits`` measures into an ancilla.

    The ancillas follow the ``width`` data qubits, in the order of the first
    circuit's classical bits, so the circuits being compared -- which share
    their classical bits -- assign every bit the same ancilla.
    """
    measured = {
        clbit
        for circuit in circuits
        for instruction in circuit.data
        if instruction.operation.name == "measure"
        for clbit in instruction.clbits
    }
    order = [clbit for circuit in circuits for clbit in circuit.clbits]
    slots: dict[Any, int] = {}
    for clbit in order:
        if clbit in measured and clbit not in slots:
            slots[clbit] = width + len(slots)
    return slots


def _measurements_into_ancillas(
    circuit: QuantumCircuit, total: int, outcome_slots: Mapping[Any, int]
) -> QuantumCircuit:
    """Return ``circuit`` with each measurement swapped into its bit's ancilla.

    A terminating measurement only records its qubit's computational-basis
    value, so moving the qubit into an ancilla that is dephased afterwards
    keeps exactly what the measurement observes and nothing it does not. The
    qubit is left in ``|0>`` -- the same in every circuit compared -- so
    whatever moves it afterwards, such as the swaps undoing routing in a merged
    circuit, cannot make two equivalent programs differ.
    """
    out = QuantumCircuit(QuantumRegister(total, "q"))
    position = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [position[qubit] for qubit in instruction.qubits]
        if operation.name == "measure":
            out.swap(qubits[0], outcome_slots[instruction.clbits[0]])
            continue
        if operation.name in _NON_UNITARY_OPS or instruction.clbits:
            raise ValueError(
                f"cannot verify a program containing {operation.name}, which "
                "state-vector comparison cannot represent"
            )
        out.append(operation, [out.qubits[qubit] for qubit in qubits])
    return out


def _dephased_fidelity(actual: Any, expected: Any, *, n_data: int) -> float:
    """Fidelity of two states once the qubits above ``n_data`` are measured.

    Qubits are little-endian, so a row of the reshaped vector fixes the
    measured ancillas and runs over the data qubits. Measuring the ancillas
    leaves one conditional data state per outcome, and the fidelity of the
    two resulting block-diagonal states is the squared sum of the conditional
    overlaps' magnitudes: one exactly when every outcome is equally likely and
    leaves the data in the same state, up to a phase per outcome.
    """
    import numpy as np

    rows = actual.size >> n_data
    by_outcome_actual = actual.reshape(rows, 2**n_data)
    by_outcome_expected = expected.reshape(rows, 2**n_data)
    overlaps = np.einsum("ij,ij->i", by_outcome_expected.conj(), by_outcome_actual)
    return float(np.sum(np.abs(overlaps)) ** 2)


def _widen(circuit: QuantumCircuit, total: int) -> QuantumCircuit:
    """Return ``circuit`` padded with idle qubits up to ``total`` width."""
    current = len(circuit.qubits)
    if current == total:
        return circuit
    widened = QuantumCircuit(QuantumRegister(total, "q"))
    widened.compose(circuit, qubits=range(current), inplace=True)
    return widened
