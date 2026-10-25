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
from numbers import Integral
from typing import Any, SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import (
    Barrier,
    ClassicalRegister,
    ControlFlowOp,
    Gate,
    Measure,
    Reset,
    Store,
)
from qiskit.circuit.library import Initialize

from quport._validation import is_boolean
from quport.aggregation import AggregationPlan, RemoteBlock, aggregate_remote_operations
from quport.architecture import MultiQPUArchitecture
from quport.distributed import (
    RemoteOp,
    _fresh_register_name,
    reassemble_distributed_program,
)
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
        Cross-QPU gates the plan leaves unserved -- those it marks
        unschedulable, and any a plan built for another circuit does not cover.
        Counted on the emitted circuit: they are emitted verbatim, so the
        circuit stays semantically faithful, but a real machine could not run
        them as written.
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
    _require_plan_for(plan, mapped, arch)
    return plan


def _require_plan_for(
    plan: AggregationPlan, mapped: QuantumCircuit, arch: MultiQPUArchitecture
) -> None:
    """Refuse a supplied plan that does not describe ``mapped``.

    A plan built for another circuit, or by hand, can name gates that are not
    there or a root that is not their operand; expanding it would fail deep
    inside, or rewire the wrong qubits. Everything the expansion relies on is
    checked here instead.
    """
    n_qpus = arch.cfg.n_qpus
    width = len(mapped.qubits)
    qindex = {qubit: index for index, qubit in enumerate(mapped.qubits)}
    seen: set[tuple[str, int, int, tuple[int, ...]]] = set()
    if not isinstance(plan.blocks, Sequence) or isinstance(plan.blocks, str):
        raise ValueError("plan.blocks must be a sequence of RemoteBlock")
    for ordinal, block in enumerate(plan.blocks):
        where = f"plan block {ordinal}"
        if not isinstance(block, RemoteBlock):
            raise ValueError(f"{where} must be a RemoteBlock")
        if block.protocol not in ("cat", "teleport"):
            raise ValueError(f"{where} has unknown protocol {block.protocol!r}")
        indices = block.gate_indices
        if (
            not isinstance(indices, tuple)
            or not indices
            or any(type(index) is not int for index in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))
            or indices[0] < 0
            or indices[-1] >= len(mapped.data)
        ):
            raise ValueError(
                f"{where} must name a non-empty, increasing run of the circuit's "
                f"{len(mapped.data)} instructions"
            )
        root = block.root_phys
        if type(root) is not int or not 0 <= root < min(width, arch.n_phys):
            raise ValueError(f"{where} roots a qubit the circuit does not have")
        if block.root_qpu != arch.qpu_of_phys(root):
            raise ValueError(f"{where} places its root on the wrong QPU")
        if (
            type(block.remote_qpu) is not int
            or not 0 <= block.remote_qpu < n_qpus
            or block.remote_qpu == block.root_qpu
        ):
            raise ValueError(f"{where} names an invalid remote QPU")
        for index in indices:
            instruction = mapped.data[index]
            operands = [qindex[qubit] for qubit in instruction.qubits]
            if (
                is_directive(instruction.operation)
                or root not in operands
                or not any(
                    operand < arch.n_phys
                    and arch.qpu_of_phys(operand) == block.remote_qpu
                    for operand in operands
                )
            ):
                raise ValueError(
                    f"{where} serves instruction {index}, which is not a gate "
                    f"between its root and QPU {block.remote_qpu}"
                )
        key = (block.protocol, root, block.remote_qpu, indices)
        if key in seen:
            raise ValueError(f"{where} repeats an earlier block")
        seen.add(key)


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
    if mapped.num_vars or getattr(mapped, "num_stretches", 0):
        raise ValueError(
            "build_telegate_circuit does not support circuits with classical "
            "variables or stretches"
        )

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
    # The mapped circuit's classical registers keep their names, so the
    # registers added here take names none of them uses.
    taken = {creg.name for creg in mapped.cregs}
    work = QuantumRegister(n_data, _fresh_register_name("q", taken))
    taken.add(work.name)
    ancilla_register = QuantumRegister(max_ancillas, _fresh_register_name("cat", taken))
    taken.add(ancilla_register.name)
    circuit = QuantumCircuit(work, ancilla_register, global_phase=mapped.global_phase)
    if mapped.clbits:
        circuit.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        circuit.add_register(creg)
    protocol_register: ClassicalRegister | None = None
    if protocol_bits:
        protocol_register = ClassicalRegister(
            protocol_bits, _fresh_register_name("cat_c", taken)
        )
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

    unserved = 0
    for index, instruction in enumerate(mapped.data):
        operation = instruction.operation
        qubits = [qindex[qubit] for qubit in instruction.qubits]
        clbits = [circuit.clbits[cindex[clbit]] for clbit in instruction.clbits]

        if is_directive(operation) or not qubits:
            circuit.append(operation, [circuit.qubits[q] for q in qubits], clbits)
            continue

        for block in starts.get(index, ()):
            open_block(block)

        serving = members.get(index) or []
        substitution = {block.root_phys: copy_of[block_key(block)] for block in serving}
        operands = [substitution.get(qubit, qubit) for qubit in qubits]
        # Where each operand sits once the plan has served the gate: a root on
        # its copy's QPU, everything else where it lives. A gate still spanning
        # QPUs is emitted verbatim, which no machine can run as written.
        hosted = {block.root_phys: block.remote_qpu for block in serving}
        if len(qubits) > 1 and (
            len(
                {
                    hosted[qubit] if qubit in hosted else arch.qpu_of_phys(qubit)
                    for qubit in qubits
                }
            )
            > 1
        ):
            unserved += 1

        circuit.append(operation, [circuit.qubits[q] for q in operands], clbits)

        for block in ends.get(index, ()):
            close_block(block)

    used = pool.allocated
    trimmed = _trim_unused_ancillas(
        circuit, work, ancilla_register.name, mapped, protocol_register, used
    )

    return TelegateProgram(
        circuit=trimmed,
        n_data=n_data,
        ancillas=tuple(range(n_data, n_data + used)),
        blocks=len(resolved.blocks),
        epr_pairs=resolved.epr_pairs,
        unschedulable_gates=unserved,
        measured=not coherent,
    )


def _trim_unused_ancillas(
    circuit: QuantumCircuit,
    work: QuantumRegister,
    ancilla_name: str,
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

    ancillas = QuantumRegister(used, ancilla_name)
    registers = (work, ancillas) if used else (work,)
    trimmed = QuantumCircuit(*registers, global_phase=circuit.global_phase)
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
    diagonality rule. Aggregating across an operation that breaks the rule
    leaves the root entangled with its copy and lowers the fidelity by however
    much that operation moves the root's computational-basis label: an ``X``
    sends it far below one -- to 0.002 with the default input seed, and no
    higher than 0.72 for any seed from 0 to 19 -- a small rotation only
    slightly below one. Either fails the check, whose tolerance is ``atol``.

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

    if not isinstance(mapped, QuantumCircuit):
        raise ValueError("mapped must be a QuantumCircuit")
    tolerance = _validated_atol(atol)
    _require_bound_parameters(mapped)
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
    return bool(fidelity >= 1.0 - tolerance)


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
    could not. That model describes a measurement only when every qubit is
    measured at most once and every classical bit written at most once, so a
    mapped circuit outside that is refused rather than compared on it.

    A measured qubit is left in a fixed generic state, the same whichever bit
    it was measured into. Moving it afterwards -- as routing and the swaps
    undoing it do -- changes nothing, but any other operation on it changes
    the compared state, so a program that uses a qubit after measuring it
    fails the check. That is conservative: deferring a measurement past the
    operations it controls would be equivalent, but the mapped circuit's
    measurements are all terminating, and a faithful split keeps them so.

    Parameters
    ----------
    remote_ops:
        The manifest matching ``local_routed``: ``routed_remote_ops`` for routed
        programs, ``program.remote_ops`` for unrouted ones.

    Raises
    ------
    ValueError
        If the circuit is too wide to simulate -- the qubits the programs or
        the circuit act on, plus one ancilla per measured classical bit,
        exceed :data:`MAX_VERIFIABLE_QUBITS` -- if
        it measures a qubit or writes a classical bit more than once, or if the
        artifacts are inconsistent enough that they cannot be merged at all.
    """
    from qiskit.quantum_info import Statevector

    if not isinstance(mapped, QuantumCircuit):
        raise ValueError("mapped must be a QuantumCircuit")
    tolerance = _validated_atol(atol)
    _require_bound_parameters(mapped)
    merged = reassemble_distributed_program(
        mapped, local_routed, remote_ops, arch, restore_layout=True
    )
    _require_bound_parameters(merged)
    # Refuses mid-circuit measurement, reset and classical control, so every
    # measurement left in `mapped` -- and so in `merged` -- is terminating.
    _unitary_part(mapped, label="the mapped circuit")
    expected_writes = _single_measurement_writes(mapped)
    if _measurement_writes(merged) != expected_writes:
        # The programs record a different set of classical bits, or record one
        # of them a different number of times: not the same computation.
        return False

    # The merged circuit spans the whole architecture, but a qubit nothing acts
    # on keeps its input state on both sides, so only the qubits something
    # acts on are simulated.
    active = _acted_on_qubits(merged, mapped)
    merged_active = _on_qubits(merged, active)
    mapped_active = _on_qubits(mapped, active)
    width = len(active)
    outcome_slots = _outcome_slots(width, mapped_active, merged_active)
    total = width + len(outcome_slots)
    if total > MAX_VERIFIABLE_QUBITS:
        raise ValueError(
            f"the programs act on {width} qubits, with {len(outcome_slots)} "
            f"measured classical bits, above the {MAX_VERIFIABLE_QUBITS}-qubit "
            "state-vector verification limit"
        )

    preparation = _widen(_random_product_state(width, seed), total)
    for slot in range(width, total):
        _prepare_left_behind_state(preparation, slot)

    actual = preparation.copy()
    actual.compose(
        _measurements_into_ancillas(merged_active, total, outcome_slots),
        qubits=range(total),
        inplace=True,
    )

    expected = preparation.copy()
    expected.compose(
        _measurements_into_ancillas(mapped_active, total, outcome_slots),
        qubits=range(total),
        inplace=True,
    )

    fidelity = _dephased_fidelity(
        Statevector(actual).data, Statevector(expected).data, n_data=width
    )
    return bool(fidelity >= 1.0 - tolerance)


#: Qiskit's non-unitary operations, recognised by class -- as the simulator
#: recognises them -- so that a unitary gate merely named ``measure`` is a gate.
_NON_UNITARY_KINDS = (
    (Measure, "measure"),
    (Reset, "reset"),
    (Initialize, "initialize"),
)


def _non_unitary_kind(operation: Any) -> str | None:
    """Name the measurement, reset or initialization ``operation`` performs.

    A custom instruction can hide one in its definition, which the simulator
    would apply by sampling an outcome, so definitions are searched too. Gates
    are unitary by contract, and control flow is refused separately.
    """
    for kind, name in _NON_UNITARY_KINDS:
        if isinstance(operation, kind):
            return name
    if isinstance(operation, (Gate, ControlFlowOp, Barrier, Store)):
        return None
    definition = getattr(operation, "definition", None)
    if not isinstance(definition, QuantumCircuit):
        return None
    for instruction in definition.data:
        kind = _non_unitary_kind(instruction.operation)
        if kind is not None:
            return kind
    return None


def _describe_non_unitary(operation: Any) -> str | None:
    """Describe the non-unitary operation ``operation`` is or hides, if any."""
    kind = _non_unitary_kind(operation)
    if kind is None or isinstance(operation, (Measure, Reset, Initialize)):
        return kind
    return f"{kind} (inside {operation.name})"


def _require_bound_parameters(circuit: QuantumCircuit) -> None:
    """Refuse a circuit with unbound parameters, which has no state to simulate."""
    if circuit.parameters:
        names = ", ".join(sorted(parameter.name for parameter in circuit.parameters))
        raise ValueError(
            f"cannot verify a circuit with unbound parameters ({names}); assign "
            "them values first"
        )


#: The smallest tolerance on ``1 - fidelity`` a verifier accepts. Correct
#: programs at the verifiable widths land within about ``1e-13`` of a perfect
#: fidelity from floating-point rounding alone, so anything tighter would fail
#: them on rounding rather than on what they compute.
MIN_VERIFICATION_ATOL = 1e-12


def _validated_atol(atol: object) -> float:
    """Return a tolerance on ``1 - fidelity`` that can still decide a check.

    A boolean is not a tolerance, and one of 1 or more -- or an infinite one --
    passes every circuit, while NaN fails every one; each would turn the
    verification into a constant. So would one below the simulation's rounding
    error, from the other side: ``0`` fails even an empty circuit.
    """
    if is_boolean(atol):
        raise ValueError("atol must be numeric, not boolean")
    try:
        value = float(cast(SupportsFloat | SupportsIndex | str, atol))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("atol must be numeric") from None
    if not math.isfinite(value) or not MIN_VERIFICATION_ATOL <= value < 1.0:
        raise ValueError(
            f"atol must be a finite tolerance in [{MIN_VERIFICATION_ATOL:g}, 1): "
            "below that, rounding in the simulation fails correct programs"
        )
    return value


def _unitary_part(circuit: QuantumCircuit, *, label: str) -> QuantumCircuit:
    """Return ``circuit`` without its terminating measurements.

    Verification compares state vectors, so it can only speak about the state a
    circuit prepares. Terminating measurements are dropped -- they read that
    state out without changing it -- while a measurement or reset that other
    operations depend on genuinely changes what the circuit computes, and is
    refused rather than quietly ignored. So is classical control: an operation
    conditioned on a classical bit, or any other control-flow block, has no
    state-vector evolution to compare. Barriers are dropped as well; they
    order operations and change no state.
    """
    terminating = _terminating_measurements(circuit)
    trimmed = QuantumCircuit(len(circuit.qubits), global_phase=circuit.global_phase)
    position = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    surviving: set[str] = set()
    controlled: set[str] = set()
    stores = False
    for index, instruction in enumerate(circuit.data):
        operation = instruction.operation
        if index in terminating or isinstance(operation, Barrier):
            continue
        non_unitary = _describe_non_unitary(operation)
        if non_unitary is not None:
            surviving.add(non_unitary)
        elif instruction.clbits or isinstance(operation, ControlFlowOp):
            controlled.add(operation.name)
        elif isinstance(operation, Store):
            # A store writes a classical bit or variable without touching a
            # qubit, and names its target inside the operation rather than in
            # its classical arguments, so the check above does not see it.
            stores = True
        else:
            trimmed.append(
                operation, [trimmed.qubits[position[q]] for q in instruction.qubits]
            )
    if surviving:
        raise ValueError(
            f"cannot verify {label}: it contains mid-circuit "
            f"{', '.join(sorted(surviving))}, which state-vector comparison "
            "cannot represent"
        )
    if controlled:
        raise ValueError(
            f"cannot verify {label}: it contains classically controlled "
            f"{', '.join(sorted(controlled))}, which state-vector comparison "
            "cannot represent"
        )
    if stores:
        raise ValueError(
            f"cannot verify {label}: it contains a classical store, which "
            "state-vector comparison cannot represent"
        )
    return trimmed


def _terminating_measurements(circuit: QuantumCircuit) -> set[int]:
    """Positions of the measurements nothing later depends on.

    A measurement terminates its qubit when nothing but barriers and other
    terminating measurements acts on that qubit afterwards. The test is per
    qubit: a barrier across the whole register orders later gates on *other*
    qubits after the measurement, but reads nothing from it, so it does not
    make the measurement mid-circuit -- Qiskit's ``remove_final_measurements``
    treats it as if it did.
    """
    used_later: set[Any] = set()
    terminating: set[int] = set()
    for index in range(len(circuit.data) - 1, -1, -1):
        instruction = circuit.data[index]
        operation = instruction.operation
        if isinstance(operation, Barrier):
            continue
        if isinstance(operation, Measure) and used_later.isdisjoint(instruction.qubits):
            terminating.add(index)
            continue
        used_later.update(instruction.qubits)
    return terminating


def _measurement_writes(circuit: QuantumCircuit) -> dict[Any, int]:
    """How many measurements write each classical bit of ``circuit``."""
    writes: dict[Any, int] = {}
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            for clbit in instruction.clbits:
                writes[clbit] = writes.get(clbit, 0) + 1
    return writes


def _single_measurement_writes(circuit: QuantumCircuit) -> dict[Any, int]:
    """Return the bits ``circuit`` measures into, refusing repeated measurement.

    Verification moves each measured qubit into an ancilla for its bit, which
    reproduces a measurement's outcome only when every qubit is measured once
    and every bit is written once: a second measurement of the same qubit
    would read the state left behind rather than an outcome, and a second
    write to the same bit would bring the first value back into play. Either is refused rather than
    compared on a model that does not describe it.
    """
    measured: set[Any] = set()
    for instruction in circuit.data:
        if not isinstance(instruction.operation, Measure):
            continue
        for qubit in instruction.qubits:
            if qubit in measured:
                raise ValueError(
                    "cannot verify a circuit that measures qubit "
                    f"{circuit.find_bit(qubit).index} more than once"
                )
            measured.add(qubit)
    writes = _measurement_writes(circuit)
    for clbit, count in writes.items():
        if count > 1:
            raise ValueError(
                "cannot verify a circuit that writes classical bit "
                f"{circuit.find_bit(clbit).index} with more than one measurement"
            )
    return writes


def _acted_on_qubits(*circuits: QuantumCircuit) -> list[int]:
    """Indices of the qubits an instruction of any of ``circuits`` acts on.

    Barriers act on nothing. The circuits share their qubits by index, as a
    merged circuit and the one it was split from do.
    """
    active: set[int] = set()
    for circuit in circuits:
        position = {qubit: index for index, qubit in enumerate(circuit.qubits)}
        for instruction in circuit.data:
            if not isinstance(instruction.operation, Barrier):
                active.update(position[qubit] for qubit in instruction.qubits)
    return sorted(active)


def _on_qubits(circuit: QuantumCircuit, qubits: Sequence[int]) -> QuantumCircuit:
    """``circuit`` on the given qubits alone, renumbered in order.

    Every other qubit must be idle, and barriers are dropped. The classical
    bits are kept as they are, so two circuits restricted alike still share
    them.
    """
    out = QuantumCircuit(QuantumRegister(len(qubits), "q"))
    out.add_bits(circuit.clbits)
    renamed = {
        circuit.qubits[old]: out.qubits[new]
        for new, old in enumerate(qubits)
        if old < len(circuit.qubits)
    }
    for instruction in circuit.data:
        if isinstance(instruction.operation, Barrier):
            continue
        out.append(
            instruction.operation,
            [renamed[qubit] for qubit in instruction.qubits],
            instruction.clbits,
        )
    return out


def _random_product_state(n_qubits: int, seed: int) -> QuantumCircuit:
    """Deterministic single-qubit rotations giving every qubit a generic state.

    The angles give each qubit non-zero amplitude on both basis vectors, so the
    input overlaps every computational basis state. That makes agreement on it
    a strong check rather than a proof: one input is not a spanning set, and two
    circuits can agree on a particular state while differing elsewhere. A wrong
    protocol passes only if it happens to map this one generic input correctly.
    """
    if is_boolean(seed) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)
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
        if isinstance(instruction.operation, Measure)
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
    qubit receives the ancilla's starting state, which
    :func:`_prepare_left_behind_state` makes the same for every bit, so moving
    it afterwards -- the swaps undoing routing in a merged circuit, or a
    measurement Qiskit re-targeted past a ``swap`` -- cannot make two
    equivalent programs differ. The state is generic rather than ``|0>``, so
    an operation that *uses* the qubit afterwards, as a control or otherwise,
    is not hidden by acting trivially on it.
    """
    out = QuantumCircuit(QuantumRegister(total, "q"))
    position = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [position[qubit] for qubit in instruction.qubits]
        if isinstance(operation, Measure):
            out.swap(qubits[0], outcome_slots[instruction.clbits[0]])
            continue
        non_unitary = _describe_non_unitary(operation)
        if non_unitary is not None or instruction.clbits:
            raise ValueError(
                f"cannot verify a program containing {non_unitary or operation.name}, "
                "which state-vector comparison cannot represent"
            )
        out.append(operation, [out.qubits[qubit] for qubit in qubits])
    return out


#: Angles of the state every outcome ancilla starts in, and so the state a
#: measured qubit is left in: generic, so that no gate acting on it reduces to
#: the identity the way a control in ``|0>`` would.
_LEFT_BEHIND_ANGLES = (1.1, 0.7)


def _prepare_left_behind_state(circuit: QuantumCircuit, qubit: int) -> None:
    """Rotate ``qubit`` of ``circuit`` from ``|0>`` into the left-behind state."""
    theta, phi = _LEFT_BEHIND_ANGLES
    circuit.ry(theta, qubit)
    circuit.rz(phi, qubit)


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
