# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.aggregation import aggregate_remote_operations
from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig
from quport.hypergraph import build_distributable_packets, ebit_cost
from quport.pipeline import random_benchmark_circuit

UNBOUNDED_PORTS = 1_000_000


def _arch(
    *, n_qpus: int = 2, compute: int = 3, comm: int = 1, inter: str = "switch"
) -> MultiQPUArchitecture:
    return MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=n_qpus,
            compute_qubits_per_qpu=compute,
            comm_qubits_per_qpu=comm,
            intra_topology="clique",
            inter_topology=inter,  # type: ignore[arg-type]
        )
    )


def test_run_of_gates_from_one_root_costs_a_single_epr_pair() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 4)
    qc.cx(0, 5)
    qc.cx(0, 4)

    plan = aggregate_remote_operations(qc, arch)

    assert len(plan.blocks) == 1
    block = plan.blocks[0]
    assert block.protocol == "cat"
    assert block.root_phys == 0
    assert block.root_qpu == 0
    assert block.remote_qpu == 1
    assert block.gate_indices == (1, 2, 3)
    assert block.start_index == 1
    assert block.end_index == 3
    assert plan.epr_pairs == 1
    assert plan.baseline_epr_pairs == 3
    assert plan.remote_gates == 3
    assert plan.reduction == pytest.approx(1.0 - 1.0 / 3.0)
    assert plan.peak_cat_copies == (0, 1)
    assert plan.evictions == 0
    assert plan.unschedulable_gates == 0


def test_breaking_the_root_forces_a_second_epr_pair() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.x(0)
    qc.cx(0, 4)

    plan = aggregate_remote_operations(qc, arch)

    assert [block.gate_indices for block in plan.blocks] == [(0,), (2,)]
    assert plan.epr_pairs == 2
    assert plan.baseline_epr_pairs == 2
    assert plan.reduction == 0.0


def test_local_gates_do_not_disturb_an_unrelated_cat_copy() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.x(1)  # a non-diagonal gate, but on a qubit that roots nothing
    qc.cx(4, 5)  # local on QPU 1, unrelated to the copy of qubit 0
    qc.cx(0, 4)

    plan = aggregate_remote_operations(qc, arch)

    assert len(plan.blocks) == 1
    assert plan.blocks[0].gate_indices == (0, 3)
    assert plan.epr_pairs == 1


def test_port_pressure_evicts_a_live_copy_and_costs_extra_pairs() -> None:
    arch = _arch(comm=1)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.cx(1, 5)
    qc.cx(0, 4)
    qc.cx(1, 5)

    scarce = aggregate_remote_operations(qc, arch)
    roomy = aggregate_remote_operations(qc, arch, ports_per_qpu=UNBOUNDED_PORTS)

    # One port on QPU 1 cannot hold both cat copies, so they thrash.
    assert scarce.epr_pairs == 4
    assert scarce.evictions > 0
    assert scarce.peak_cat_copies == (0, 1)
    # With ports to spare both copies stay live and each root pays once.
    assert roomy.epr_pairs == 2
    assert roomy.evictions == 0
    assert roomy.peak_cat_copies == (0, 2)


def test_max_block_gates_caps_how_long_a_port_stays_pinned() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    for _ in range(4):
        qc.cx(0, 4)

    uncapped = aggregate_remote_operations(qc, arch)
    capped = aggregate_remote_operations(qc, arch, max_block_gates=2)

    assert [block.size() for block in uncapped.blocks] == [4]
    assert [block.size() for block in capped.blocks] == [2, 2]
    assert capped.epr_pairs == 2


def test_non_diagonal_cross_qpu_gate_uses_a_teleport_round_trip() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.swap(0, 4)

    plan = aggregate_remote_operations(qc, arch)

    assert len(plan.blocks) == 1
    block = plan.blocks[0]
    assert block.protocol == "teleport"
    assert block.epr_pairs == 2
    # The host is the QPU of the first operand; the second operand travels.
    assert block.root_phys == 4
    assert block.root_qpu == 1
    assert block.remote_qpu == 0
    assert plan.epr_pairs == 2


def test_three_qubit_gate_spanning_three_qpus_gathers_both_operands() -> None:
    arch = _arch(n_qpus=3, comm=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.ccx(0, 5, 10)

    plan = aggregate_remote_operations(qc, arch)

    assert len(plan.blocks) == 2
    assert {block.protocol for block in plan.blocks} == {"teleport"}
    assert {block.remote_qpu for block in plan.blocks} == {0}
    assert plan.epr_pairs == 4
    assert plan.blocks_by_gate_index()[0] == plan.blocks


def test_zero_ports_makes_every_remote_gate_unschedulable() -> None:
    arch = _arch(comm=0)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 3)
    qc.cx(0, 4)

    plan = aggregate_remote_operations(qc, arch)

    assert plan.blocks == ()
    assert plan.epr_pairs == 0
    assert plan.baseline_epr_pairs == 0
    assert plan.unschedulable_gates == 2
    assert plan.reduction == 0.0


def test_local_only_circuit_needs_no_entanglement() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 1)
    qc.cx(4, 5)

    plan = aggregate_remote_operations(qc, arch)

    assert plan.blocks == ()
    assert plan.remote_gates == 0
    assert plan.epr_pairs == 0
    assert plan.reduction == 0.0


def test_blocks_are_sorted_and_cover_every_served_gate_once() -> None:
    arch = _arch(n_qpus=3, comm=2)
    qc = random_benchmark_circuit(8, 10, 5)
    from qiskit import transpile

    mapped = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cx"],
        initial_layout=list(range(8)),
        optimization_level=0,
        seed_transpiler=5,
    )

    plan = aggregate_remote_operations(mapped, arch)

    starts = [
        (block.start_index, block.remote_qpu, block.root_phys) for block in plan.blocks
    ]
    assert starts == sorted(starts)
    for block in plan.blocks:
        assert block.gate_indices == tuple(sorted(block.gate_indices))
        assert len(set(block.gate_indices)) == len(block.gate_indices)
    served = [index for block in plan.blocks for index in block.gate_indices]
    assert len(served) == len(set(served))


@pytest.mark.parametrize("seed", range(6))
def test_unbounded_ports_match_hypergraph_ebits(seed: int) -> None:
    """The compile-time planner and the partition-time model must agree.

    :mod:`quport.aggregation` walks a mapped circuit and allocates real ports;
    :mod:`quport.hypergraph` evaluates a closed-form lambda-1 metric over
    packets. With ports unconstrained they are two independent computations of
    the same quantity, so any divergence is a bug in one of them.
    """
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=3 + (seed % 2),
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1 + (seed % 2),
        inter_topology="switch",
    )
    qc = random_benchmark_circuit(9, 8, seed)
    result = compile_distributed(qc, cfg, seed=seed)
    arch = MultiQPUArchitecture(cfg)
    mapped = result.physical_circuit

    plan = aggregate_remote_operations(mapped, arch, ports_per_qpu=UNBOUNDED_PORTS)
    decomposition = build_distributable_packets(mapped)
    physical_part = [arch.qpu_of_phys(phys) for phys in range(mapped.num_qubits)]

    assert plan.epr_pairs == ebit_cost(decomposition, physical_part, cfg.n_qpus)
    assert plan.epr_pairs <= plan.baseline_epr_pairs


@pytest.mark.parametrize("seed", range(4))
def test_more_ports_never_increase_epr_demand(seed: int) -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=4, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1, inter_topology="ring"
    )
    qc = random_benchmark_circuit(10, 10, seed)
    result = compile_distributed(qc, cfg, seed=seed)
    arch = MultiQPUArchitecture(cfg)

    previous = None
    for ports in (1, 2, 4, UNBOUNDED_PORTS):
        plan = aggregate_remote_operations(
            result.physical_circuit, arch, ports_per_qpu=ports
        )
        assert all(peak <= ports for peak in plan.peak_cat_copies)
        if previous is not None:
            assert plan.epr_pairs <= previous
        previous = plan.epr_pairs


def test_plan_serializes_to_standards_compliant_json() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.cx(0, 5)

    payload = aggregate_remote_operations(qc, arch).to_dict()
    text = json.dumps(payload, allow_nan=False)

    assert json.loads(text)["epr_pairs"] == 1
    assert payload["blocks"][0]["gates"] == 2


def test_per_qpu_port_budget_is_honoured() -> None:
    arch = _arch(n_qpus=2, comm=2)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 5)
    qc.cx(1, 6)
    qc.cx(0, 5)
    qc.cx(1, 6)

    plan = aggregate_remote_operations(qc, arch, ports_per_qpu=[2, 1])

    # QPU 1 may host only one copy, so the two roots evict each other.
    assert plan.peak_cat_copies[1] == 1
    assert plan.epr_pairs == 4


@pytest.mark.parametrize(
    ("ports", "message"),
    [
        (-1, "ports_per_qpu must be non-negative"),
        ("2", "must be an integer or a sequence"),
        ([1], "length must match n_qpus"),
        ([1, True], r"ports_per_qpu\[1\] must be an integer"),
        ([1, -2], r"ports_per_qpu\[1\] must be non-negative"),
    ],
)
def test_aggregate_rejects_invalid_port_budgets(ports: object, message: str) -> None:
    arch = _arch()
    with pytest.raises(ValueError, match=message):
        aggregate_remote_operations(QuantumCircuit(arch.n_phys), arch, ports_per_qpu=ports)  # type: ignore[arg-type]


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_aggregate_rejects_invalid_block_caps(cap: object) -> None:
    arch = _arch()
    with pytest.raises(ValueError, match="max_block_gates must be"):
        aggregate_remote_operations(QuantumCircuit(arch.n_phys), arch, max_block_gates=cap)  # type: ignore[arg-type]


def test_aggregate_rejects_wrong_argument_types() -> None:
    arch = _arch()
    with pytest.raises(ValueError, match="mapped must be a QuantumCircuit"):
        aggregate_remote_operations(object(), arch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arch must be a MultiQPUArchitecture"):
        aggregate_remote_operations(QuantumCircuit(1), object())  # type: ignore[arg-type]
