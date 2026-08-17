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
@pytest.mark.parametrize("optimization_level", [0, 3])
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
    """Two QPUs may list the same remote operations in opposite orders.

    Barriers on disjoint qubits commute, so a routed program can order its
    markers however its DAG rebuild happened to. Reading each program strictly
    linearly would deadlock here; merging by qubit dataflow does not.
    """
    from quport.distributed import reassemble_distributed_program, split_into_qpus

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)  # remote op 0, on qubits 0 and 3
    mapped.cx(1, 4)  # remote op 1, on qubits 1 and 4 -- disjoint from op 0

    program = split_into_qpus(mapped, arch)

    # Hand QPU 1 its markers in the opposite order. Both orders are legal: the
    # two operations touch disjoint qubits, so nothing constrains them.
    swapped = QuantumCircuit(arch.n_phys)
    order = [1, 0]
    for index in order:
        for instruction in program.local_circuits[1].data:
            label = getattr(instruction.operation, "label", None)
            if label == f"quport_remote_{index}":
                swapped.append(instruction.operation, instruction.qubits, [])

    merged = reassemble_distributed_program(
        mapped,
        {0: program.local_circuits[0], 1: swapped},
        program.remote_ops,
        arch,
        restore_layout=False,
    )

    assert [instruction.operation.name for instruction in merged.data] == ["cx", "cx"]


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
