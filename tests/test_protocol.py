# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit, qasm3

from quport.aggregation import AggregationPlan, RemoteBlock, aggregate_remote_operations
from quport.architecture import MultiQPUArchitecture
from quport.compiler import compile_distributed
from quport.config import MultiQPUConfig
from quport.pipeline import random_benchmark_circuit
from quport.protocol import (
    MAX_VERIFIABLE_QUBITS,
    build_telegate_circuit,
    verify_telegate_equivalence,
)


def _arch(*, n_qpus: int = 2, compute: int = 3, comm: int = 1) -> MultiQPUArchitecture:
    return MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=n_qpus,
            compute_qubits_per_qpu=compute,
            comm_qubits_per_qpu=comm,
            intra_topology="clique",
            inter_topology="switch",
        )
    )


def test_expansion_of_a_single_block_has_the_expected_shape() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 4)
    qc.rz(0.4, 0)  # diagonal on the root: the copy survives
    qc.cx(0, 5)

    program = build_telegate_circuit(qc, arch)

    assert program.blocks == 1
    assert program.epr_pairs == 1
    assert program.n_data == arch.n_phys
    assert program.n_ancillas == 2  # one cat copy plus one recycled EPR helper
    assert program.measured is False
    assert program.unschedulable_gates == 0

    counts = program.circuit.count_ops()
    # entangler h(a) + reset h(a) + disentangler h(b) + reset h(b), plus the
    # circuit's own h on the root.
    assert counts["h"] == 5
    assert counts["cz"] == 1
    assert "measure" not in counts


@pytest.mark.parametrize("seed", range(6))
def test_expansion_reproduces_the_mapped_circuit(seed: int) -> None:
    """The emitted protocol must compute exactly what the mapped circuit does.

    This is the empirical check behind the whole entanglement stack: the data
    qubits must come out right *and* the ancillas must be left unentangled from
    them, which is what tracing them out and demanding unit fidelity tests.
    """
    cfg = MultiQPUConfig(
        n_qpus=2 + seed % 2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        optimization_level=0,
    )
    qc = random_benchmark_circuit(min(cfg.total_physical_qubits(), 5), 6, seed)
    result = compile_distributed(qc, cfg, seed=seed)
    arch = MultiQPUArchitecture(cfg)

    assert verify_telegate_equivalence(result.physical_circuit, arch, seed=seed)


def test_verification_fails_when_a_block_spans_a_non_diagonal_root_gate() -> None:
    """The diagonality rule is load-bearing, not a conservative guess.

    A hand-built plan that keeps one cat copy live across an ``X`` on its root
    is exactly what :mod:`quport.aggregation` refuses to emit. Feeding it in
    anyway drives the fidelity to zero, which is the evidence that the rule
    :mod:`quport.entanglement` enforces is the right one.
    """
    arch = _arch(compute=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 3)
    qc.x(0)
    qc.cx(0, 3)

    honest = aggregate_remote_operations(qc, arch)
    assert [block.gate_indices for block in honest.blocks] == [(1,), (3,)]
    assert verify_telegate_equivalence(qc, arch, honest)

    forced = AggregationPlan(
        blocks=(
            RemoteBlock(
                protocol="cat",
                root_phys=0,
                root_qpu=0,
                remote_qpu=1,
                gate_indices=(1, 3),
                epr_pairs=1,
            ),
        ),
        remote_gates=2,
        epr_pairs=1,
        baseline_epr_pairs=2,
        unschedulable_gates=0,
        evictions=0,
        peak_cat_copies=(0, 1),
    )
    assert not verify_telegate_equivalence(qc, arch, forced)


def test_diagonal_root_gates_inside_a_block_are_safe() -> None:
    """Rz/T/S/CZ-control traffic on the root must not disturb the copy."""
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 4)
    qc.rz(0.7, 0)
    qc.t(0)
    qc.cz(0, 1)  # the root acts as a control: still diagonal
    qc.barrier(0)
    qc.cx(0, 5)

    plan = aggregate_remote_operations(qc, arch)

    assert len(plan.blocks) == 1
    assert plan.blocks[0].gate_indices == (2, 7)
    assert verify_telegate_equivalence(qc, arch, plan)


def test_teleport_block_round_trips_the_operand() -> None:
    arch = _arch(compute=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.h(3)
    qc.swap(0, 3)  # no diagonal operand: served by a teleport round trip

    plan = aggregate_remote_operations(qc, arch)

    assert [block.protocol for block in plan.blocks] == ["teleport"]
    program = build_telegate_circuit(qc, arch, plan)
    assert program.epr_pairs == 2
    assert verify_telegate_equivalence(qc, arch, plan)


def test_multiple_concurrent_blocks_recycle_ancillas() -> None:
    arch = _arch(comm=2)
    qc = QuantumCircuit(arch.n_phys)
    for _ in range(3):
        qc.cx(0, 5)
        qc.cx(1, 6)
        qc.x(0)
        qc.x(1)

    plan = aggregate_remote_operations(qc, arch)
    program = build_telegate_circuit(qc, arch, plan)

    # Six blocks, but never more than two copies live at once, so the expansion
    # stays narrow instead of allocating one ancilla per block.
    assert program.blocks == 6
    assert program.n_ancillas <= 3
    assert verify_telegate_equivalence(qc, arch, plan)


def test_local_only_circuit_is_emitted_unchanged() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(4, 5)

    program = build_telegate_circuit(qc, arch)

    assert program.blocks == 0
    assert program.n_ancillas == 0
    assert program.circuit.num_qubits == arch.n_phys
    assert verify_telegate_equivalence(qc, arch)


def test_classical_bits_and_measurements_survive_the_expansion() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys, 2)
    qc.h(0)
    qc.cx(0, 4)
    qc.measure(0, 0)  # closes the block, and must land on the same clbit
    qc.measure(4, 1)

    program = build_telegate_circuit(qc, arch)

    assert program.circuit.num_clbits == 2
    counts = program.circuit.count_ops()
    assert counts["measure"] == 2


def test_measured_form_uses_feedforward_and_exports_to_qasm3() -> None:
    arch = _arch(compute=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 3)
    qc.x(0)
    qc.cx(0, 3)

    program = build_telegate_circuit(qc, arch, coherent=False)

    assert program.measured is True
    counts = program.circuit.count_ops()
    # Two blocks, each with an entangler and a disentangler measurement.
    assert counts["measure"] == 4
    assert counts["if_else"] == 4
    assert counts["reset"] == 4

    source = qasm3.dumps(program.circuit)
    assert "if (" in source


def test_measured_form_is_the_deferred_measurement_image_of_the_coherent_one() -> None:
    """Each feedforward branch must carry exactly the right Pauli correction.

    The measured form is the coherent form with ``cx(a, copy)`` replaced by
    ``measure a`` plus a conditional ``X``, and ``cz(copy, root)`` replaced by
    ``measure copy`` plus a conditional ``Z``. The coherent form is verified
    numerically elsewhere, so checking that substitution is what carries the
    result across; a conditional body with the wrong Pauli, the wrong target, or
    the wrong trigger value would break the protocol silently.
    """
    arch = _arch(compute=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 3)

    program = build_telegate_circuit(qc, arch, coherent=False)
    circuit = program.circuit
    root = circuit.qubits[0]
    copy = circuit.qubits[program.n_data]

    conditionals = [
        instruction
        for instruction in circuit.data
        if instruction.operation.name == "if_else"
    ]
    assert len(conditionals) == 2

    corrections = []
    for instruction in conditionals:
        body = instruction.operation.blocks[0]
        assert body.size() == 1
        inner = body.data[0]
        # The body acts on the operation's own qubits, so position 0 of the
        # body maps to instruction.qubits[0].
        assert len(instruction.qubits) == 1
        corrections.append((inner.operation.name, instruction.qubits[0]))

    # Entangler correction lands on the cat copy; disentangler correction on the
    # root. Every branch fires on outcome 1.
    assert corrections == [("x", copy), ("z", root)]
    for instruction in conditionals:
        assert instruction.operation.condition[1] == 1


def test_verification_refuses_a_plan_with_unschedulable_gates() -> None:
    arch = _arch(comm=0)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 3)

    with pytest.raises(ValueError, match="unschedulable gates"):
        verify_telegate_equivalence(qc, arch)


def test_verification_refuses_circuits_too_wide_to_simulate() -> None:
    arch = _arch(n_qpus=4, compute=MAX_VERIFIABLE_QUBITS, comm=1)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, arch.n_phys - 1)

    with pytest.raises(ValueError, match="state-vector verification limit"):
        verify_telegate_equivalence(qc, arch)


def test_build_telegate_circuit_validates_arguments() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)

    with pytest.raises(ValueError, match="mapped must be a QuantumCircuit"):
        build_telegate_circuit(object(), arch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arch must be a MultiQPUArchitecture"):
        build_telegate_circuit(qc, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coherent must be a boolean"):
        build_telegate_circuit(qc, arch, coherent="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plan must be an AggregationPlan"):
        build_telegate_circuit(qc, arch, object())  # type: ignore[arg-type]


def test_verification_seed_must_be_an_integer() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)

    with pytest.raises(ValueError, match="seed must be an integer"):
        verify_telegate_equivalence(qc, arch, seed=True)


# ---------------------------------------------------------------------------
# Distributed-program reassembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("intra", ["clique", "line", "ring", "grid2d"])
@pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
def test_compiled_artifacts_still_compute_their_circuit(
    intra: str, optimization_level: int
) -> None:
    """The central claim of distributed compilation, checked by simulation.

    The per-QPU programs and the remote-operation manifest are merged back into
    one circuit and compared against the mapped circuit they were split from.
    Every non-clique intra topology makes routing permute qubits inside a QPU,
    so this exercises the manifest remapping as well as the split itself.
    """
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology=intra,  # type: ignore[arg-type]
        optimization_level=optimization_level,
    )
    arch = MultiQPUArchitecture(cfg)
    result = compile_distributed(random_benchmark_circuit(5, 6, 2), cfg, seed=2)

    assert result.routed_remote_ops, "the fixture must produce remote operations"
    assert verify_distributed_program(
        result.physical_circuit, result.local_routed, result.routed_remote_ops, arch
    )


def test_reassembly_recovers_the_unrouted_split_too() -> None:
    """`split_into_qpus` alone must also be reversible, with its own manifest."""
    from quport.distributed import reassemble_distributed_program, split_into_qpus
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(n_qpus=3, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.h(0)
    mapped.cx(0, 3)
    mapped.cx(3, 6)
    mapped.cx(0, 1)
    mapped.cx(6, 0)

    program = split_into_qpus(mapped, arch)
    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch, restore_layout=False
    )

    assert merged.size() == mapped.size()
    assert verify_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch
    )


def test_reassembly_follows_qubit_dataflow_not_program_order() -> None:
    """A local gate may be listed either side of a marker on another qubit.

    Routing rebuilds each program from its DAG, so a gate that touches none of a
    remote operation's qubits can come out before or after that operation's
    marker. Both orders describe the same computation; merging by qubit dataflow
    accepts either, while reading the program linearly against the manifest
    would not.
    """
    from quport.distributed import reassemble_distributed_program, split_into_qpus
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)  # remote op 0, on qubits 0 and 3
    mapped.h(1)  # local to QPU 0, on a qubit the remote operation never touches

    program = split_into_qpus(mapped, arch)

    # Hand QPU 0 its local gate before the marker instead of after it.
    reordered = QuantumCircuit(arch.n_phys)
    instructions = list(program.local_circuits[0].data)
    for instruction in sorted(
        instructions, key=lambda item: item.operation.name != "h"
    ):
        reordered.append(instruction.operation, instruction.qubits, [])

    merged = reassemble_distributed_program(
        mapped,
        {0: reordered, 1: program.local_circuits[1]},
        program.remote_ops,
        arch,
        restore_layout=False,
    )

    assert sorted(instruction.operation.name for instruction in merged.data) == [
        "cx",
        "h",
    ]
    assert verify_distributed_program(
        mapped, {0: reordered, 1: program.local_circuits[1]}, program.remote_ops, arch
    )


@pytest.mark.parametrize("intra", ["clique", "line", "ring", "grid2d"])
def test_routing_cannot_reorder_a_qpus_remote_markers(intra: str) -> None:
    """Remote operations are synchronization points; their order is not routing's.

    A marker naming only its own operand does not order itself against the
    QPU's other markers, so the transpiler used to be free to emit two of them
    the other way round -- and the local gates it interleaves then tie them
    together in that order, leaving two QPUs demanding opposite orders and no
    execution satisfying both. Each marker therefore also names the previous
    marker's qubit.
    """
    from quport.distributed import _remote_barrier_ordinal
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology=intra,  # type: ignore[arg-type]
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    result = compile_distributed(random_benchmark_circuit(5, 8, 7), cfg, seed=7)
    assert len(result.routed_remote_ops) > 2, "the fixture must exercise ordering"

    for circuit in result.local_routed.values():
        ordinals = [
            ordinal
            for instruction in circuit.data
            if (ordinal := _remote_barrier_ordinal(instruction.operation)) is not None
        ]
        assert ordinals == sorted(ordinals)

    assert verify_distributed_program(
        result.physical_circuit, result.local_routed, result.routed_remote_ops, arch
    )


def test_reassembly_reports_contradictory_orderings() -> None:
    """A genuine ordering conflict must be reported, not silently reordered."""
    from quport.distributed import reassemble_distributed_program, split_into_qpus

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)  # remote op 0
    mapped.cx(0, 3)  # remote op 1, on the *same* qubits, so the order is fixed

    program = split_into_qpus(mapped, arch)

    # Reverse QPU 1's markers. Now QPU 0 insists on 0 then 1 and QPU 1 insists
    # on 1 then 0, on the same qubits: no execution order satisfies both.
    reversed_qpu1 = QuantumCircuit(arch.n_phys)
    markers = [
        instruction
        for instruction in program.local_circuits[1].data
        if instruction.operation.name == "barrier"
    ]
    for instruction in reversed(markers):
        reversed_qpu1.append(instruction.operation, instruction.qubits, [])

    with pytest.raises(ValueError, match="contradictory orders"):
        reassemble_distributed_program(
            mapped,
            {0: program.local_circuits[0], 1: reversed_qpu1},
            program.remote_ops,
            arch,
            restore_layout=False,
        )


def test_reassembly_validates_its_arguments() -> None:
    from quport.distributed import reassemble_distributed_program

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    qc = QuantumCircuit(arch.n_phys)

    with pytest.raises(ValueError, match="mapped must be a QuantumCircuit"):
        reassemble_distributed_program(object(), {}, [], arch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="local_routed must be a mapping"):
        reassemble_distributed_program(qc, [], [], arch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arch must be a MultiQPUArchitecture"):
        reassemble_distributed_program(qc, {}, [], object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a QuantumCircuit"):
        reassemble_distributed_program(qc, {0: object()}, [], arch)  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Circuit shapes beyond the random benchmark
# ---------------------------------------------------------------------------


def _ghz(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for target in range(1, n_qubits):
        circuit.cx(0, target)
    return circuit


def _qft(n_qubits: int) -> QuantumCircuit:
    import math

    circuit = QuantumCircuit(n_qubits)
    for control in range(n_qubits):
        circuit.h(control)
        for target in range(control + 1, n_qubits):
            circuit.cp(math.pi / 2 ** (target - control), target, control)
    return circuit


def _measured_ghz(n_qubits: int) -> QuantumCircuit:
    circuit = _ghz(n_qubits)
    measured = QuantumCircuit(n_qubits, n_qubits)
    measured.compose(circuit, inplace=True)
    measured.barrier()
    for qubit in range(n_qubits):
        measured.measure(qubit, qubit)
    return measured


def _cz_chain(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits)
    for qubit in range(n_qubits):
        circuit.h(qubit)
    for qubit in range(n_qubits - 1):
        circuit.cz(qubit, qubit + 1)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


@pytest.mark.parametrize(
    "builder",
    [_ghz, _qft, _measured_ghz, _cz_chain],
    ids=["ghz", "qft", "measured", "cz"],
)
@pytest.mark.parametrize("intra", ["clique", "line"])
@pytest.mark.parametrize(
    "basis", [("rz", "sx", "x", "cx"), ("rz", "sx", "x", "cz")], ids=["cx", "cz"]
)
def test_structured_circuits_survive_the_whole_pipeline(
    builder: object, intra: str, basis: tuple[str, ...]
) -> None:
    """Shapes the random benchmark never produces, end to end.

    A shared control (`ghz`), a controlled-phase ladder (`qft`), terminating
    measurement with a classical register, and symmetric two-qubit gates whose
    root the aggregator has to choose (`cz`). A CZ basis matters because it is
    the only way the symmetric-root path is reached at all.
    """
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology=intra,  # type: ignore[arg-type]
        basis_gates=basis,
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    result = compile_distributed(builder(5), cfg, seed=1, strategy="ebit")  # type: ignore[operator]

    assert verify_distributed_program(
        result.physical_circuit, result.local_routed, result.routed_remote_ops, arch
    )
    if result.aggregation.unschedulable_gates == 0:
        assert verify_telegate_equivalence(
            result.physical_circuit, arch, result.aggregation
        )


def test_terminating_measurements_and_classical_bits_round_trip() -> None:
    """Reassembly has to carry classical arguments, or `measure` cannot be rebuilt."""
    from quport.distributed import reassemble_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1, optimization_level=0
    )
    arch = MultiQPUArchitecture(cfg)
    result = compile_distributed(_measured_ghz(5), cfg, seed=1)

    merged = reassemble_distributed_program(
        result.physical_circuit,
        result.local_routed,
        result.routed_remote_ops,
        arch,
        restore_layout=False,
    )

    assert merged.num_clbits == result.physical_circuit.num_clbits
    assert merged.count_ops()["measure"] == 5


def test_verification_refuses_mid_circuit_measurement() -> None:
    """State-vector comparison cannot represent a measurement others depend on."""
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1, optimization_level=0
    )
    arch = MultiQPUArchitecture(cfg)
    circuit = QuantumCircuit(4, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.cx(0, 2)
    result = compile_distributed(circuit, cfg, seed=0)

    with pytest.raises(ValueError, match="mid-circuit measure"):
        verify_distributed_program(
            result.physical_circuit, result.local_routed, result.routed_remote_ops, arch
        )


def test_verification_refuses_classical_control_with_a_clear_error() -> None:
    """An ``if`` block has no state vector; say so instead of crashing in Qiskit.

    The condition here reads a bit nothing has measured yet, so the mid-circuit
    measurement check does not fire. Without a check of its own the control-flow
    block reached the simulator and surfaced as a Qiskit error that callers
    catching ``ValueError`` -- the command line among them -- do not expect.
    """
    from qiskit import ClassicalRegister, QuantumRegister

    from quport.distributed import split_into_qpus
    from quport.protocol import verify_distributed_program

    arch = _arch(compute=2)
    creg = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(QuantumRegister(arch.n_phys, "q"), creg)
    circuit.h(0)
    circuit.cx(0, 3)
    with circuit.if_test((creg, 0)):
        circuit.x(1)
    circuit.measure(0, 0)
    program = split_into_qpus(circuit, arch)

    with pytest.raises(ValueError, match="classically controlled if_else"):
        verify_distributed_program(
            circuit, program.local_circuits, program.remote_ops, arch
        )
    with pytest.raises(ValueError, match="classically controlled if_else"):
        verify_telegate_equivalence(circuit, arch)


def test_symmetric_cz_across_qpus_is_served_and_correct() -> None:
    """Both operands of a CZ are diagonal, so the aggregator picks a root."""
    arch = _arch(compute=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.h(3)
    qc.cz(0, 3)
    qc.cz(0, 3)

    plan = aggregate_remote_operations(qc, arch)

    # One cat copy serves both gates: a CZ leaves its root diagonal.
    assert len(plan.blocks) == 1
    assert plan.blocks[0].protocol == "cat"
    assert plan.epr_pairs == 1
    assert plan.baseline_epr_pairs == 2
    assert verify_telegate_equivalence(qc, arch, plan)


_ONE_QUBIT = ("h", "x", "y", "z", "s", "sdg", "t", "sx")
_ONE_QUBIT_ROTATIONS = ("rz", "rx", "ry", "p")
_TWO_QUBIT = (
    "cx",
    "cy",
    "cz",
    "ch",
    "csx",
    "cs",
    "csdg",
    "swap",
    "iswap",
    "ecr",
    "dcx",
)
_TWO_QUBIT_ROTATIONS = ("cp", "crz", "crx", "cry", "rzz", "rzx", "rxx", "ryy")
_THREE_QUBIT = ("ccx", "cswap", "ccz")


@pytest.mark.parametrize("seed", range(16))
def test_plans_over_the_full_standard_gate_set_compute_their_circuit(
    seed: int,
) -> None:
    """Aggregation is only as sound as the diagonality rule under it.

    Compiled circuits reach the aggregator in the ``cx`` basis, which never
    exercises a gate diagonal on one operand only (``rzx``, ``ch``), a
    symmetric one with a free root (``cp``, ``rzz``), a gate diagonal on
    neither (``iswap``, ``ecr``) or a three-qubit gate. Here random physical
    circuits drawn from all of them are aggregated under several port budgets
    and every plan is checked by simulation: a cat copy kept alive across an
    operation that disturbs its root sends the fidelity to zero.
    """
    import math
    import random

    rng = random.Random(seed)
    arch = _arch(
        n_qpus=rng.randint(2, 3), compute=rng.randint(1, 2), comm=rng.randint(2, 3)
    )
    n = arch.n_phys
    qc = QuantumCircuit(n)
    for _ in range(rng.randint(6, 20)):
        kind = rng.random()
        if kind < 0.3:
            if rng.random() < 0.5:
                getattr(qc, rng.choice(_ONE_QUBIT))(rng.randrange(n))
            else:
                angle = rng.uniform(0.0, 2 * math.pi)
                getattr(qc, rng.choice(_ONE_QUBIT_ROTATIONS))(angle, rng.randrange(n))
        elif kind < 0.9:
            a, b = rng.sample(range(n), 2)
            if rng.random() < 0.5:
                getattr(qc, rng.choice(_TWO_QUBIT))(a, b)
            else:
                angle = rng.uniform(0.0, 2 * math.pi)
                getattr(qc, rng.choice(_TWO_QUBIT_ROTATIONS))(angle, a, b)
        else:
            getattr(qc, rng.choice(_THREE_QUBIT))(*rng.sample(range(n), 3))

    verified = 0
    for ports in (None, 2, 1):
        plan = aggregate_remote_operations(qc, arch, ports_per_qpu=ports)
        if plan.unschedulable_gates:
            # Too few ports for this plan to run at all; nothing to check.
            continue
        assert verify_telegate_equivalence(qc, arch, plan, seed=seed), ports
        verified += 1
    assert verified, "the fixture must produce at least one runnable plan"


def test_verification_fits_in_memory_well_below_the_width_limit() -> None:
    """Sixteen data qubits must verify; the state vector is only 2**18 long.

    Tracing the ancillas out into a reduced density matrix would need 4**16
    entries -- 64 GiB -- for the data alone, which ran out of memory long
    before the advertised 24-qubit limit. The fidelity is summed from the
    state vector instead.
    """
    arch = _arch(n_qpus=2, compute=7, comm=1)
    assert arch.n_phys == 16
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 8)
    qc.cx(0, 9)

    plan = aggregate_remote_operations(qc, arch)
    assert plan.blocks, "the fixture must cross QPUs"
    assert verify_telegate_equivalence(qc, arch, plan)
