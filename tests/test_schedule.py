# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.config import LatencyModel, MultiQPUConfig
from quport.network import UNREACHABLE_DISTANCE
from quport.schedule import (
    UNSCHEDULABLE_PENALTY,
    estimate_parallel_makespan,
    estimate_parallel_makespan_layered,
    estimate_parallel_makespan_topology,
    estimate_topology_schedule_plan,
)


def test_unschedulable_penalty_matches_unreachable_distance() -> None:
    assert UNSCHEDULABLE_PENALTY == float(UNREACHABLE_DISTANCE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("oneq", -1.0),
        ("twoq", -1.0),
        ("swap", -1.0),
        ("epr_gen", float("nan")),
        ("classical_rtt", -1.0),
        ("remote_gate_overhead", float("inf")),
        ("oneq", True),
        ("oneq", object()),
    ],
)
def test_schedule_estimators_reject_invalid_latency_model_values(
    field: str, value: object
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    model = LatencyModel()
    object.__setattr__(model, field, value)

    for estimator in (
        estimate_parallel_makespan,
        estimate_parallel_makespan_layered,
        estimate_parallel_makespan_topology,
    ):
        with pytest.raises(ValueError):
            estimator(qc, arch, model)


def test_topology_estimator_counts_parallel_remote_rounds_with_port_limits() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 3)
    qc.cx(1, 4)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 2
    assert summary.remote_rounds == 2
    assert summary.peak_qpu_ports_used == 1
    assert summary.makespan > 0.0


def test_topology_estimator_handles_disconnected_qpus_with_penalty_not_crash() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan >= UNSCHEDULABLE_PENALTY


def test_topology_estimator_disconnected_pairs_are_unschedulable_even_with_zero_latencies() -> (
    None
):
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(
        qc,
        arch,
        LatencyModel(epr_gen=0.0, classical_rtt=0.0, remote_gate_overhead=0.0),
    )

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan >= UNSCHEDULABLE_PENALTY


def test_topology_estimator_scales_penalty_with_multiple_unreachable_remote_ops() -> (
    None
):
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=1,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)
    qc.cx(1, 3)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 2
    assert summary.remote_rounds == 2
    assert summary.makespan >= 2e9


def test_topology_estimator_penalizes_unschedulable_switch_pair_budget() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        switch_parallel_links=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan >= UNSCHEDULABLE_PENALTY


def test_topology_estimator_counts_penalty_rounds_when_ports_unavailable() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)
    qc.cx(1, 3)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 2
    assert summary.remote_rounds == 2
    assert summary.makespan >= 2e9


def test_topology_estimator_scales_unschedulable_switch_pair_budget_penalty() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        switch_parallel_links=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 3)
    qc.cx(1, 3)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 2
    assert summary.remote_rounds == 2
    assert summary.makespan >= 2e9


def test_topology_estimator_allows_same_pair_when_switch_pair_budget_is_one() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        switch_parallel_links=1,
        link_capacity=2,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 4)
    qc.cx(1, 5)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 2
    assert summary.remote_rounds == 1
    assert summary.peak_qpu_ports_used == 2


def test_topology_estimator_scales_zero_switch_budget_penalty_for_many_ops() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        switch_parallel_links=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 4)
    qc.cx(1, 4)
    qc.cx(2, 4)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 3
    assert summary.remote_rounds == 3
    assert summary.makespan >= 3e9


def test_topology_estimator_rejects_invalid_switch_reconfig_delay() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        switch_reconfig_delay=0.0,
    )
    object.__setattr__(cfg, "switch_reconfig_delay", float("nan"))
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    with pytest.raises(ValueError, match="switch_reconfig_delay"):
        estimate_parallel_makespan_topology(qc, arch, LatencyModel())


def test_schedule_estimators_reject_boolean_n_qpus() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    object.__setattr__(cfg, "n_qpus", True)

    with pytest.raises(ValueError, match="n_qpus"):
        MultiQPUArchitecture(cfg)


def test_schedule_estimators_handle_zero_and_multi_qubit_ops_consistently() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits(), 1)
    qc.barrier()  # zero-qubit op: should be ignored by estimators
    qc.ccx(0, 1, 2)  # multi-qubit op: modeled conservatively on first-qpu only
    qc.measure(0, 0)

    model = LatencyModel(oneq=1.0, twoq=5.0, swap=7.0)
    summary_linear = estimate_parallel_makespan(qc, arch, model)
    summary_layered = estimate_parallel_makespan_layered(qc, arch, model)

    assert summary_linear.remote_ops == 0
    assert summary_layered.remote_ops == 0
    assert summary_linear.makespan > 0.0
    assert summary_linear.makespan == pytest.approx(summary_layered.makespan)


def test_topology_estimator_rejects_boolean_comm_ports() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
    )
    object.__setattr__(cfg, "comm_qubits_per_qpu", True)

    with pytest.raises(ValueError, match="comm_qubits_per_qpu"):
        MultiQPUArchitecture(cfg)


def test_schedule_estimators_accept_int_subclasses_for_n_qpus() -> None:
    class FancyInt(int):
        pass

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    object.__setattr__(cfg, "n_qpus", FancyInt(2))
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    model = LatencyModel()
    for estimator in (
        estimate_parallel_makespan,
        estimate_parallel_makespan_layered,
        estimate_parallel_makespan_topology,
    ):
        summary = estimator(qc, arch, model)
        assert summary.makespan > 0


def test_topology_estimator_penalizes_zero_link_capacity() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
        link_capacity=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan >= UNSCHEDULABLE_PENALTY


def test_topology_estimator_scales_zero_link_capacity_penalty_for_many_ops() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=2,
        inter_topology="ring",
        link_capacity=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 5)
    qc.cx(1, 6)
    qc.cx(2, 7)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 3
    assert summary.remote_rounds == 3
    assert summary.makespan >= 3e9


def test_topology_estimator_rejects_invalid_async_overlap() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_overlap=0.5,
    )
    object.__setattr__(cfg, "async_overlap", float("nan"))
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    with pytest.raises(ValueError, match="async_overlap"):
        estimate_parallel_makespan_topology(qc, arch, LatencyModel())


def test_topology_estimator_rejects_boolean_async_overlap() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_overlap=0.5,
    )
    object.__setattr__(cfg, "async_overlap", True)
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    with pytest.raises(ValueError, match="async_overlap"):
        estimate_parallel_makespan_topology(qc, arch, LatencyModel())


def test_topology_estimator_rejects_non_boolean_async_classical() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_classical=True,
    )
    object.__setattr__(cfg, "async_classical", "yes")
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    with pytest.raises(ValueError, match="async_classical"):
        estimate_parallel_makespan_topology(qc, arch, LatencyModel())


def test_topology_estimator_ignores_switch_reconfig_delay_on_non_switch_topology() -> (
    None
):
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
        switch_reconfig_delay=0.0,
    )
    object.__setattr__(cfg, "switch_reconfig_delay", float("nan"))
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())
    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1


def test_topology_estimator_ignores_switch_pair_budget_on_non_switch_topology() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
        switch_parallel_links=1,
    )
    object.__setattr__(cfg, "switch_parallel_links", float("nan"))
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())
    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1


def test_layered_estimator_rejects_boolean_comm_ports() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    object.__setattr__(cfg, "comm_qubits_per_qpu", True)

    with pytest.raises(ValueError, match="comm_qubits_per_qpu"):
        MultiQPUArchitecture(cfg)


def test_layered_estimator_scales_penalty_when_ports_unavailable() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)
    qc.cx(1, 3)

    summary = estimate_parallel_makespan_layered(qc, arch, LatencyModel())

    assert summary.remote_ops == 2
    assert summary.makespan >= 2e9


def test_layered_estimator_penalty_does_not_hide_large_local_work() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=0,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    # Independent local and remote two-qubit operations can co-appear in one layer.
    qc.cx(1, 2)
    qc.cx(0, 3)

    model = LatencyModel(twoq=2e9)
    summary = estimate_parallel_makespan_layered(qc, arch, model)

    assert summary.remote_ops == 1
    assert summary.makespan >= 2e9


def test_topology_estimator_no_port_penalty_keeps_large_local_layer_time() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=0,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    # Local and remote op can occur in the same layer (disjoint qubits).
    qc.cx(1, 2)
    qc.cx(0, 3)

    model = LatencyModel(twoq=2e9)
    summary = estimate_parallel_makespan_topology(qc, arch, model)

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan >= 2e9


def test_topology_estimator_zero_link_penalty_keeps_large_local_layer_time() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
        link_capacity=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(1, 2)
    qc.cx(0, 4)

    model = LatencyModel(twoq=2e9)
    summary = estimate_parallel_makespan_topology(qc, arch, model)

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan >= 2e9


def test_schedule_estimators_ignore_zero_qubit_instructions() -> None:
    from qiskit.circuit import Instruction

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)
    qc.append(Instruction("global_note", 0, 0, []), [], [])

    model = LatencyModel()
    base = estimate_parallel_makespan(qc, arch, model)
    layered = estimate_parallel_makespan_layered(qc, arch, model)
    topo = estimate_parallel_makespan_topology(qc, arch, model)

    assert base.remote_ops == 1
    assert layered.remote_ops == 1
    assert topo.remote_ops == 1
    assert base.makespan > 0.0
    assert layered.makespan > 0.0
    assert topo.makespan > 0.0


def test_schedule_estimators_handle_zero_qubit_only_circuits() -> None:
    from qiskit.circuit import Instruction

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.append(Instruction("global_note", 0, 0, []), [], [])

    model = LatencyModel()
    base = estimate_parallel_makespan(qc, arch, model)
    layered = estimate_parallel_makespan_layered(qc, arch, model)
    topo = estimate_parallel_makespan_topology(qc, arch, model)

    assert base.remote_ops == 0
    assert layered.remote_ops == 0
    assert topo.remote_ops == 0
    assert base.makespan == 0.0
    assert layered.makespan == 0.0
    assert topo.makespan == 0.0


def test_schedule_estimators_accept_non_builtin_integral_qpu_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    monkeypatch.setattr(arch, "qpu_of_phys", lambda i: np.int64(i // 2))

    for estimator in (
        estimate_parallel_makespan,
        estimate_parallel_makespan_layered,
        estimate_parallel_makespan_topology,
    ):
        summary = estimator(qc, arch, LatencyModel())
        assert summary.makespan > 0


@pytest.mark.parametrize(
    "mapping_value,error_pattern",
    [
        (2, r"out-of-range QPU"),
        (-1, r"out-of-range QPU"),
        (0.5, r"must return an integer"),
        (True, r"must return an integer"),
    ],
)
def test_schedule_estimators_reject_invalid_qpu_mapping_values(
    monkeypatch: pytest.MonkeyPatch,
    mapping_value: object,
    error_pattern: str,
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    monkeypatch.setattr(arch, "qpu_of_phys", lambda _i: mapping_value)

    for estimator in (
        estimate_parallel_makespan,
        estimate_parallel_makespan_layered,
        estimate_parallel_makespan_topology,
    ):
        with pytest.raises(ValueError, match=error_pattern):
            estimator(qc, arch, LatencyModel())


def test_topology_estimator_treats_single_port_clos_as_ring_not_switch() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="clos",
        switch_parallel_links=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    summary = estimate_parallel_makespan_topology(qc, arch, LatencyModel())

    assert summary.remote_ops == 1
    assert summary.remote_rounds == 1
    assert summary.makespan < UNSCHEDULABLE_PENALTY


def test_topology_schedule_plan_exposes_round_trace() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        link_capacity=2,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 4)
    qc.cx(1, 5)

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    assert plan.summary.remote_ops == 2
    assert plan.summary.remote_rounds == 1
    assert len(plan.layers) == plan.summary.layers
    remote_layers = [layer for layer in plan.layers if layer.remote_ops]
    assert len(remote_layers) == 1
    round_trace = remote_layers[0].remote_rounds[0]
    assert round_trace.qpu_pairs == ((0, 1), (0, 1))
    assert round_trace.qpu_ports_used == (2, 2)
    assert round_trace.unschedulable_ops == 0


def test_topology_schedule_plan_records_unschedulable_rounds() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=0,
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 1)

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    assert plan.summary.remote_ops == 1
    assert plan.summary.remote_rounds == 1
    remote_layers = [layer for layer in plan.layers if layer.remote_ops]
    assert remote_layers[0].remote_rounds[0].unschedulable_ops == 1
    assert remote_layers[0].remote_rounds[0].duration >= UNSCHEDULABLE_PENALTY


def test_topology_schedule_plan_summary_matches_public_summary_api() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.h(0)
    qc.cx(0, 3)
    qc.cx(1, 6)

    model = LatencyModel()
    plan = estimate_topology_schedule_plan(qc, arch, model)
    summary = estimate_parallel_makespan_topology(qc, arch, model)

    assert plan.summary == summary
    assert plan.summary.layers == len(plan.layers)
    assert sum(layer.remote_ops for layer in plan.layers) == summary.remote_ops
    assert (
        sum(len(layer.remote_rounds) for layer in plan.layers) == summary.remote_rounds
    )


def test_topology_schedule_plan_handles_circuits_without_dag_layers() -> None:
    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=0,
        comm_qubits_per_qpu=0,
    )
    arch = MultiQPUArchitecture(cfg)
    qc = QuantumCircuit(0)

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    assert plan.summary.makespan == 0.0
    assert plan.summary.layers == 0
    assert plan.summary.remote_ops == 0
    assert plan.summary.remote_rounds == 0
    assert plan.layers == ()


def test_topology_schedule_plan_records_multihop_link_utilization() -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    # Physical qubits 0 and 4 are on QPUs 0 and 2, which are two hops apart on
    # the ring.  The deterministic shortest path is 0 -> 1 -> 2.
    qc.cx(0, 4)

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    round_trace = next(
        layer.remote_rounds[0] for layer in plan.layers if layer.remote_ops
    )
    assert round_trace.qpu_pairs == ((0, 2),)
    assert round_trace.qpu_ports_used == (1, 0, 1, 0)
    assert round_trace.link_utilization == (((0, 1), 1), ((1, 2), 1))


def test_topology_schedule_plan_records_switch_reconfiguration_delay() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        switch_reconfig_delay=7.0,
    )
    arch = MultiQPUArchitecture(cfg)
    lat = LatencyModel(epr_gen=10.0, classical_rtt=4.0, remote_gate_overhead=3.0)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 2)

    plan = estimate_topology_schedule_plan(qc, arch, lat)

    remote_round = next(
        layer.remote_rounds[0] for layer in plan.layers if layer.remote_ops
    )
    # One hop: epr_gen + overlapped classical_rtt (50% by default) + overhead + reconfig.
    assert remote_round.duration == 10.0 + 2.0 + 3.0 + 7.0
    assert plan.summary.makespan == remote_round.duration


def test_topology_schedule_plan_records_zero_switch_pair_budget_penalties() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        switch_parallel_links=0,
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.cx(0, 4)
    qc.cx(1, 5)

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    remote_layers = [layer for layer in plan.layers if layer.remote_ops]
    assert len(remote_layers) == 1
    assert remote_layers[0].remote_ops == 2
    assert [round_.unschedulable_ops for round_ in remote_layers[0].remote_rounds] == [
        1,
        1,
    ]
    assert plan.summary.remote_rounds == 2
    assert plan.summary.makespan >= 2 * UNSCHEDULABLE_PENALTY
