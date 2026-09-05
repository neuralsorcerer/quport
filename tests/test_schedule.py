# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.config import LatencyModel, MultiQPUConfig
from quport.network import UNREACHABLE_DISTANCE
from quport.schedule import (
    UNSCHEDULABLE_PENALTY,
    LayerScheduleTrace,
    RemoteRoundTrace,
    TopologySchedulePlan,
    TopologyScheduleSummary,
    estimate_parallel_makespan,
    estimate_parallel_makespan_layered,
    estimate_parallel_makespan_topology,
    estimate_topology_schedule_plan,
)


def _assert_schedule_plan_timeline_is_consistent(plan: object) -> None:
    layers = plan.layers
    if not layers:
        assert plan.summary.makespan == 0.0
        return

    assert layers[0].start_time == 0.0
    for previous, current in zip(layers[:-1], layers[1:], strict=True):
        assert current.start_time == previous.end_time
    assert layers[-1].end_time == plan.summary.makespan

    for layer in layers:
        assert layer.end_time == layer.start_time + layer.duration
        previous_round_end = layer.start_time
        for round_trace in layer.remote_rounds:
            assert round_trace.start_time == previous_round_end
            assert round_trace.end_time == round_trace.start_time + round_trace.duration
            assert round_trace.end_time <= layer.end_time
            previous_round_end = round_trace.end_time


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
        ("oneq", 10**400),
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


def test_schedule_estimators_ignore_qubit_scoped_barrier_directives() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.barrier(0)  # single-qubit directive: no local gate time
    qc.barrier(0, 4)  # spans both QPUs but is a directive, not a remote op

    model = LatencyModel()
    summary_linear = estimate_parallel_makespan(qc, arch, model)
    summary_layered = estimate_parallel_makespan_layered(qc, arch, model)
    summary_topology = estimate_parallel_makespan_topology(qc, arch, model)

    assert summary_linear.remote_ops == 0
    assert summary_layered.remote_ops == 0
    assert summary_topology.remote_ops == 0
    assert summary_linear.makespan == 0.0
    assert summary_layered.makespan == 0.0
    assert summary_topology.makespan == 0.0
    assert summary_topology.remote_rounds == 0


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


def test_topology_schedule_plan_records_absolute_layer_and_round_times() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    latency = LatencyModel(
        oneq=1.0,
        twoq=10.0,
        epr_gen=100.0,
        classical_rtt=20.0,
        remote_gate_overhead=5.0,
    )

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.x(0)
    qc.cx(0, 3)
    qc.x(1)

    plan = estimate_topology_schedule_plan(qc, arch, latency)

    _assert_schedule_plan_timeline_is_consistent(plan)

    remote_layer = next(layer for layer in plan.layers if layer.remote_rounds)
    remote_round = remote_layer.remote_rounds[0]
    assert remote_round.start_time == remote_layer.start_time


def test_topology_schedule_plan_to_dict_is_json_ready_and_stable() -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    qc = QuantumCircuit(cfg.total_physical_qubits())
    qc.x(0)
    qc.cx(0, 4)

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())
    payload = plan.to_dict()

    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["summary"] == plan.summary.to_dict()
    remote_layer_payload = next(
        layer for layer in payload["layers"] if layer["remote_rounds"]
    )
    round_payload = remote_layer_payload["remote_rounds"][0]
    assert round_payload["qpu_pairs"] == [[0, 2]]
    assert round_payload["link_utilization"] == [
        {"edge": [0, 1], "count": 1},
        {"edge": [1, 2], "count": 1},
    ]
    assert round_payload["start_time"] == remote_layer_payload["start_time"]


@pytest.mark.parametrize(
    "summary,match",
    [
        (
            TopologyScheduleSummary(
                makespan=float("nan"),
                layers=0,
                remote_ops=0,
                remote_rounds=0,
                peak_link_util=0,
                peak_qpu_ports_used=0,
            ),
            "summary.makespan must be finite",
        ),
        (
            TopologyScheduleSummary(
                makespan=0.0,
                layers=True,
                remote_ops=0,
                remote_rounds=0,
                peak_link_util=0,
                peak_qpu_ports_used=0,
            ),
            "summary.layers must be an integer, not boolean",
        ),
    ],
)
def test_topology_schedule_summary_to_dict_rejects_invalid_fields(
    summary: TopologyScheduleSummary, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        summary.to_dict()


@pytest.mark.parametrize(
    "round_trace,match",
    [
        (
            RemoteRoundTrace(
                layer_index=0,
                round_index=0,
                qpu_pairs=((0, 0),),
                duration=0.0,
                qpu_ports_used=(0, 0),
                link_utilization=(),
            ),
            r"round.qpu_pairs\[0\] entries must be distinct",
        ),
        (
            RemoteRoundTrace(
                layer_index=0,
                round_index=0,
                qpu_pairs=((0, 1),),
                duration=0.0,
                qpu_ports_used=(0, -1),
                link_utilization=(),
            ),
            r"round.qpu_ports_used\[1\] must be non-negative",
        ),
        (
            RemoteRoundTrace(
                layer_index=0,
                round_index=0,
                qpu_pairs=((0, 1),),
                duration=0.0,
                qpu_ports_used=(0, 0),
                link_utilization=(((0, 1), True),),
            ),
            r"round.link_utilization\[0\].count must be an integer, not boolean",
        ),
    ],
)
def test_remote_round_trace_to_dict_rejects_invalid_fields(
    round_trace: RemoteRoundTrace, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        round_trace.to_dict()


def test_layer_schedule_trace_to_dict_rejects_invalid_round_payload() -> None:
    layer = LayerScheduleTrace(
        layer_index=0,
        local_duration=0.0,
        remote_ops=1,
        remote_rounds=(
            RemoteRoundTrace(
                layer_index=0,
                round_index=0,
                qpu_pairs=((0, 1),),
                duration=float("inf"),
                qpu_ports_used=(1, 1),
                link_utilization=(((0, 1), 1),),
            ),
        ),
        duration=0.0,
    )

    with pytest.raises(ValueError, match="round.duration must be finite"):
        layer.to_dict()


def test_topology_schedule_plan_times_unschedulable_no_port_rounds() -> None:
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

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    _assert_schedule_plan_timeline_is_consistent(plan)
    remote_layer = next(layer for layer in plan.layers if layer.remote_rounds)
    assert [round_.start_time for round_ in remote_layer.remote_rounds] == [
        remote_layer.start_time,
        remote_layer.start_time + UNSCHEDULABLE_PENALTY,
    ]
    assert all(round_.unschedulable_ops == 1 for round_ in remote_layer.remote_rounds)


def test_topology_schedule_plan_times_unreachable_rounds() -> None:
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

    plan = estimate_topology_schedule_plan(qc, arch, LatencyModel())

    _assert_schedule_plan_timeline_is_consistent(plan)
    remote_layer = next(layer for layer in plan.layers if layer.remote_rounds)
    assert len(remote_layer.remote_rounds) == 1
    assert remote_layer.remote_rounds[0].start_time == remote_layer.start_time
    assert remote_layer.remote_rounds[0].duration == UNSCHEDULABLE_PENALTY
    assert remote_layer.remote_rounds[0].unschedulable_ops == 1


def _single_layer_remote_plan(
    cfg: MultiQPUConfig, pairs: list[tuple[int, int]]
) -> TopologySchedulePlan:
    """Build a physical circuit whose cross-QPU gates all fall in one DAG layer."""
    arch = MultiQPUArchitecture(cfg)
    circuit = QuantumCircuit(arch.n_phys)
    for control, target in pairs:
        circuit.cx(control, target)
    return estimate_topology_schedule_plan(circuit, arch, LatencyModel())


@pytest.mark.parametrize(
    ("link_capacity", "expected_rounds"),
    [(1, 2), (2, 1)],
)
def test_link_capacity_limits_remote_ops_sharing_one_link(
    link_capacity: int, expected_rounds: int
) -> None:
    """Two remote ops crossing the same QPU link contend for that link's capacity."""
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=0,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        link_capacity=link_capacity,
        switch_parallel_links=1000,
    )

    # Disjoint qubit pairs, so both gates share a DAG layer; both cross QPU0<->QPU1.
    layer = _single_layer_remote_plan(cfg, [(0, 2), (1, 3)]).layers[0]

    assert layer.remote_ops == 2
    assert len(layer.remote_rounds) == expected_rounds
    for remote_round in layer.remote_rounds:
        for _edge, count in remote_round.link_utilization:
            assert count <= link_capacity


@pytest.mark.parametrize(
    ("switch_parallel_links", "expected_rounds"),
    [(1, 2), (2, 1)],
)
def test_switch_parallel_links_limits_distinct_pairs_per_round(
    switch_parallel_links: int, expected_rounds: int
) -> None:
    """A switch fabric can serve only `switch_parallel_links` QPU pairs per round."""
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=0,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        link_capacity=4,
        switch_parallel_links=switch_parallel_links,
    )

    # QPU0<->QPU1 and QPU2<->QPU3: two distinct pairs, no shared link, ports to spare.
    layer = _single_layer_remote_plan(cfg, [(0, 2), (4, 6)]).layers[0]

    assert layer.remote_ops == 2
    assert len(layer.remote_rounds) == expected_rounds
    for remote_round in layer.remote_rounds:
        assert len(set(remote_round.qpu_pairs)) <= switch_parallel_links


def _three_qpu_arch() -> MultiQPUArchitecture:
    # Blocks of two: QPU0 = {0,1}, QPU1 = {2,3}, QPU2 = {4,5}.
    return MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=3,
            compute_qubits_per_qpu=1,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )


def test_estimators_agree_on_remote_count_for_multi_qubit_cross_qpu_gates() -> None:
    """A 3-qubit gate spanning QPUs is one remote event in every view.

    ``split_into_qpus`` emits a single ``RemoteOp`` for such a gate, so the
    layered and topology estimators must not silently bill it as local work.
    """
    from quport.distributed import split_into_qpus

    arch = _three_qpu_arch()
    circuit = QuantumCircuit(arch.n_phys)
    circuit.ccx(0, 2, 4)  # spans QPU0, QPU1, QPU2
    circuit.ccx(0, 1, 2)  # spans QPU0 (twice) and QPU1
    latency = LatencyModel()

    expected = len(split_into_qpus(circuit, arch).remote_ops)
    assert expected == 2

    assert estimate_parallel_makespan(circuit, arch, latency).remote_ops == expected
    assert (
        estimate_parallel_makespan_layered(circuit, arch, latency).remote_ops
        == expected
    )
    summary = estimate_topology_schedule_plan(circuit, arch, latency).summary
    assert summary.remote_ops == expected
    assert summary.remote_rounds >= 1


def test_multi_qubit_gate_inside_one_qpu_is_still_billed_as_local() -> None:
    """Operands sharing a QPU stay local and cost one local two-qubit slot."""
    from quport.distributed import split_into_qpus

    arch = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=2,
            compute_qubits_per_qpu=3,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )
    circuit = QuantumCircuit(arch.n_phys)
    circuit.ccx(0, 1, 2)  # entirely inside QPU0
    latency = LatencyModel()

    assert split_into_qpus(circuit, arch).remote_ops == []
    assert estimate_parallel_makespan_layered(circuit, arch, latency).remote_ops == 0

    summary = estimate_topology_schedule_plan(circuit, arch, latency).summary
    assert summary.remote_ops == 0
    assert summary.remote_rounds == 0
    assert summary.makespan == latency.twoq


def test_multi_qubit_barrier_across_qpus_is_never_a_remote_event() -> None:
    """Barriers are directives: they separate layers but cost nothing."""
    from quport.distributed import split_into_qpus

    arch = _three_qpu_arch()
    circuit = QuantumCircuit(arch.n_phys)
    circuit.barrier(0, 2, 4)
    latency = LatencyModel()

    assert split_into_qpus(circuit, arch).remote_ops == []
    assert estimate_parallel_makespan_layered(circuit, arch, latency).remote_ops == 0

    summary = estimate_topology_schedule_plan(circuit, arch, latency).summary
    assert summary.remote_ops == 0
    assert summary.makespan == 0.0


def _one_layer_plan(
    cfg: MultiQPUConfig, pairs: list[tuple[int, int]]
) -> TopologySchedulePlan:
    arch = MultiQPUArchitecture(cfg)
    circuit = QuantumCircuit(arch.n_phys)
    for control, target in pairs:
        circuit.cx(control, target)
    return estimate_topology_schedule_plan(circuit, arch, LatencyModel())


def test_comm_port_budget_serializes_remote_ops_sharing_a_qpu() -> None:
    """A QPU with one port cannot serve two remote ops in the same round.

    Both ops touch QPU 0 but use different links, and link capacity is ample, so
    only the port budget can force the second round.
    """
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        link_capacity=4,
        switch_parallel_links=1000,
    )

    layer = _one_layer_plan(cfg, [(0, 2), (1, 4)]).layers[0]

    assert layer.remote_ops == 2
    assert len(layer.remote_rounds) == 2
    assert {tuple(r.qpu_pairs) for r in layer.remote_rounds} == {((0, 1),), ((0, 2),)}
    for remote_round in layer.remote_rounds:
        assert max(remote_round.qpu_ports_used) <= 1


def test_remote_cost_scales_with_qpu_hop_count() -> None:
    """`d(a,b) * epr_gen` means each extra hop costs exactly one more `epr_gen`."""
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )
    latency = LatencyModel()

    one_hop = _one_layer_plan(cfg, [(0, 2)]).summary.makespan  # QPU0 <-> QPU1
    two_hop = _one_layer_plan(cfg, [(0, 4)]).summary.makespan  # QPU0 <-> QPU2

    assert two_hop - one_hop == pytest.approx(latency.epr_gen)
    assert one_hop > latency.epr_gen


def test_layer_duration_overlaps_local_work_with_remote_rounds() -> None:
    """A layer takes `max(local, remote)`, not their sum."""
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
    )

    # Remote op between QPU0 and QPU1, plus an unrelated local 2Q gate on QPU2.
    layer = _one_layer_plan(cfg, [(0, 3), (6, 7)]).layers[0]

    rounds_time = sum(remote_round.duration for remote_round in layer.remote_rounds)
    assert layer.local_duration > 0.0
    assert rounds_time > 0.0
    assert layer.duration == pytest.approx(max(layer.local_duration, rounds_time))
    assert layer.duration < layer.local_duration + rounds_time


@pytest.mark.parametrize("async_overlap", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_classical_latency_hiding_scales_with_one_minus_overlap(
    async_overlap: float,
) -> None:
    """Effective RTT is `(1 - overlap) * classical_rtt`.

    At overlap 0.5 that is numerically identical to `overlap * classical_rtt`, so
    the default alone cannot distinguish the two forms.
    """
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_classical=True,
        async_overlap=async_overlap,
    )
    latency = LatencyModel()

    makespan = _one_layer_plan(cfg, [(0, 2)]).summary.makespan

    expected = (
        latency.epr_gen
        + latency.classical_rtt * (1.0 - async_overlap)
        + latency.remote_gate_overhead
    )
    assert makespan == pytest.approx(expected)


def test_unschedulable_layer_duration_also_overlaps_local_work() -> None:
    """The zero-capacity branch overlaps local work with penalty rounds too.

    With `link_capacity=0` every remote op becomes an unschedulable penalty
    round, but the layer still runs local gates alongside them, so its duration
    is the maximum rather than the sum.
    """
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        link_capacity=0,
    )

    # Remote op across QPU0/QPU1 plus an unrelated local 2Q gate on QPU2.
    layer = _one_layer_plan(cfg, [(0, 3), (6, 7)]).layers[0]

    rounds_time = sum(remote_round.duration for remote_round in layer.remote_rounds)
    assert layer.local_duration > 0.0
    assert rounds_time == pytest.approx(UNSCHEDULABLE_PENALTY)
    assert all(r.unschedulable_ops == 1 for r in layer.remote_rounds)
    assert layer.duration == pytest.approx(max(layer.local_duration, rounds_time))
    assert layer.duration < layer.local_duration + rounds_time


def test_simple_estimator_synchronizes_both_qpus_at_a_remote_op() -> None:
    """A remote op is a rendezvous: both timelines advance from the later one.

    QPU0 runs five local gates, then a remote op, then QPU1 runs one more. Only
    a real sync makes QPU1 inherit QPU0's elapsed time before its final gate.
    """
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)  # QPU0 = {0, 1}, QPU1 = {2, 3}
    latency = LatencyModel()

    circuit = QuantumCircuit(arch.n_phys)
    for _ in range(5):
        circuit.x(0)
    circuit.cx(0, 2)
    circuit.x(2)

    summary = estimate_parallel_makespan(circuit, arch, latency)

    remote_cost = latency.epr_gen + latency.classical_rtt + latency.remote_gate_overhead
    assert summary.remote_ops == 1
    assert summary.steps == 1
    assert summary.makespan == pytest.approx(
        5 * latency.oneq + remote_cost + latency.oneq
    )


def test_link_capacity_counts_both_traversal_directions_as_one_link() -> None:
    """An undirected link is one resource however it is traversed.

    The two remote ops cross QPU0<->QPU1 in opposite operand orders. Keying link
    usage by traversal direction instead of canonically would file them under
    separate links and let both through in a single round.
    """
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=0,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        link_capacity=1,
        switch_parallel_links=1000,
    )

    layer = _one_layer_plan(cfg, [(0, 2), (3, 1)]).layers[0]

    assert layer.remote_ops == 2
    assert len(layer.remote_rounds) == 2
    for remote_round in layer.remote_rounds:
        assert remote_round.link_utilization == (((0, 1), 1),)

    # A lone op whose operands run high-QPU-first must still report the link
    # canonically, otherwise consumers see (0, 1) and (1, 0) as separate links.
    reversed_only = _one_layer_plan(cfg, [(2, 0)]).layers[0]
    assert reversed_only.remote_rounds[0].link_utilization == (((0, 1), 1),)
    assert reversed_only.remote_rounds[0].qpu_pairs == ((0, 1),)


def test_round_packing_schedules_the_most_distant_pair_first() -> None:
    """Greedy packing is longest-first, so the costliest op leads the layer."""
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
    )

    # Both ops need QPU0's single port: one is 1 hop, the other 2 hops.
    layer = _one_layer_plan(cfg, [(0, 2), (1, 4)]).layers[0]

    assert len(layer.remote_rounds) == 2
    assert layer.remote_rounds[0].qpu_pairs == ((0, 2),)  # the two-hop pair
    assert layer.remote_rounds[1].qpu_pairs == ((0, 1),)
    assert layer.remote_rounds[0].duration > layer.remote_rounds[1].duration


def test_peak_link_utilisation_counts_the_link_in_use() -> None:
    """`peak_link_util` reports usage after placement, so one op means one link."""
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
    )

    summary = _one_layer_plan(cfg, [(0, 2)]).summary

    assert summary.remote_ops == 1
    assert summary.peak_link_util == 1
    assert summary.peak_qpu_ports_used == 1


def test_layered_estimator_always_charges_at_least_one_remote_round() -> None:
    """Rounds are `ceil(degree / ports)`; flooring would make an op free.

    With two ports and a single remote op, a floored division yields zero rounds
    and the communication cost vanishes from the layer.
    """
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)
    latency = LatencyModel()

    circuit = QuantumCircuit(arch.n_phys)
    circuit.cx(0, 3)

    summary = estimate_parallel_makespan_layered(circuit, arch, latency)

    assert summary.remote_ops == 1
    assert summary.makespan == pytest.approx(
        latency.epr_gen + latency.classical_rtt + latency.remote_gate_overhead
    )


def test_layered_estimator_runs_same_qpu_gates_in_a_layer_concurrently() -> None:
    """Within a DAG layer a QPU's local work is parallel, so durations max."""
    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=0,
        intra_topology="clique",
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)
    latency = LatencyModel()

    circuit = QuantumCircuit(arch.n_phys)
    for qubit in range(3):
        circuit.x(qubit)  # disjoint qubits: one DAG layer

    summary = estimate_parallel_makespan_layered(circuit, arch, latency)

    assert summary.steps == 1
    assert summary.makespan == pytest.approx(latency.oneq)


def _single_qpu_pair_arch() -> MultiQPUArchitecture:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    return MultiQPUArchitecture(cfg)


def test_linear_makespan_charges_local_two_qubit_gates_at_two_qubit_latency() -> None:
    """A same-QPU 2Q gate advances that QPU by `twoq` -- not `oneq`, and not a sync.

    `estimate_parallel_makespan` only reaches its two-qubit branch for gates that
    `split_into_qpus` left local: every cross-QPU pair is consumed by the remote-op
    path above it. Circuits built from the default basis on a partitioned layout
    happen to route every 2Q gate down one of those other paths, so without this
    test the branch never runs and a wrong latency term stays invisible.
    """
    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    block = arch.block_of_qpu(0)

    circuit = QuantumCircuit(arch.n_phys)
    for _ in range(3):
        circuit.cx(block.compute[0], block.compute[1])

    summary = estimate_parallel_makespan(circuit, arch, latency)

    assert summary.makespan == pytest.approx(3 * latency.twoq)
    assert summary.remote_ops == 0
    assert summary.steps == 0


def test_linear_makespan_charges_local_swaps_at_swap_latency() -> None:
    """A same-QPU SWAP costs `swap`, which is distinct from `twoq`."""
    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    block = arch.block_of_qpu(0)

    circuit = QuantumCircuit(arch.n_phys)
    circuit.swap(block.compute[0], block.compute[1])

    summary = estimate_parallel_makespan(circuit, arch, latency)

    assert latency.swap != latency.twoq
    assert summary.makespan == pytest.approx(latency.swap)
    assert summary.steps == 0


def test_linear_makespan_adds_local_gate_costs_along_one_timeline() -> None:
    """Local 1Q/2Q/SWAP gates on one QPU serialize; the makespan is their sum."""
    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    block = arch.block_of_qpu(0)

    circuit = QuantumCircuit(arch.n_phys)
    circuit.x(block.compute[0])
    circuit.cx(block.compute[0], block.compute[1])
    circuit.swap(block.compute[0], block.compute[1])

    summary = estimate_parallel_makespan(circuit, arch, latency)

    assert summary.makespan == pytest.approx(latency.oneq + latency.twoq + latency.swap)


def test_linear_makespan_syncs_both_timelines_only_on_remote_ops() -> None:
    """A local gate stays on its own timeline; the following remote op syncs both."""
    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    local_block = arch.block_of_qpu(0)
    remote_block = arch.block_of_qpu(1)
    remote_cost = latency.epr_gen + latency.classical_rtt + latency.remote_gate_overhead

    circuit = QuantumCircuit(arch.n_phys)
    circuit.cx(local_block.compute[0], local_block.compute[1])
    circuit.cx(local_block.compute[0], remote_block.compute[0])

    summary = estimate_parallel_makespan(circuit, arch, latency)

    assert summary.remote_ops == 1
    assert summary.steps == 1
    assert summary.makespan == pytest.approx(latency.twoq + remote_cost)


def test_layered_makespan_charges_local_swaps_at_swap_latency() -> None:
    """The layered estimator distinguishes SWAP from a generic 2Q gate too."""
    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    block = arch.block_of_qpu(0)

    swap_circuit = QuantumCircuit(arch.n_phys)
    swap_circuit.swap(block.compute[0], block.compute[1])
    twoq_circuit = QuantumCircuit(arch.n_phys)
    twoq_circuit.cx(block.compute[0], block.compute[1])

    assert estimate_parallel_makespan_layered(
        swap_circuit, arch, latency
    ).makespan == pytest.approx(latency.swap)
    assert estimate_parallel_makespan_layered(
        twoq_circuit, arch, latency
    ).makespan == pytest.approx(latency.twoq)


def test_topology_makespan_charges_local_swaps_at_swap_latency() -> None:
    """The topology-aware estimator keeps its own SWAP branch, so pin it separately.

    `_topology_schedule_plan` re-implements the per-layer local-duration walk rather
    than sharing the layered estimator's, so a wrong latency term in one is invisible
    to a test that only drives the other.
    """
    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    block = arch.block_of_qpu(0)

    swap_circuit = QuantumCircuit(arch.n_phys)
    swap_circuit.swap(block.compute[0], block.compute[1])
    twoq_circuit = QuantumCircuit(arch.n_phys)
    twoq_circuit.cx(block.compute[0], block.compute[1])

    swap_summary = estimate_parallel_makespan_topology(swap_circuit, arch, latency)
    twoq_summary = estimate_parallel_makespan_topology(twoq_circuit, arch, latency)

    assert swap_summary.makespan == pytest.approx(latency.swap)
    assert twoq_summary.makespan == pytest.approx(latency.twoq)
    assert swap_summary.remote_ops == 0


def test_estimators_ignore_operations_that_carry_no_qubits() -> None:
    """A zero-qubit op belongs to no QPU and must consume no time anywhere.

    The three estimators reach that conclusion by different routes, and the
    difference matters for what this test can claim. The linear estimator walks
    `circuit.data`, so it really does receive the op and skip it on its own
    `len == 0` guard. The layered and topology estimators walk `dag.layers()`,
    which groups nodes by the qubits they occupy and therefore omits qubit-less
    ops before any quport code runs -- their guards are unreachable today, and
    asserting they "ignore" the op would prove nothing about them.

    So pin the observable invariant for all three, and pin the DAG property the
    other two depend on. If a future Qiskit starts emitting qubit-less ops in
    layers, that second assertion fails and the guards become load-bearing.
    """
    from qiskit.circuit.library import GlobalPhaseGate
    from qiskit.converters import circuit_to_dag

    arch = _single_qpu_pair_arch()
    latency = LatencyModel()
    block = arch.block_of_qpu(0)

    plain = QuantumCircuit(arch.n_phys)
    plain.cx(block.compute[0], block.compute[1])

    with_phase = QuantumCircuit(arch.n_phys)
    with_phase.append(GlobalPhaseGate(0.25), [], [])
    with_phase.cx(block.compute[0], block.compute[1])

    # the op is in the circuit and in the DAG ...
    assert [inst.operation.name for inst in with_phase.data] == ["global_phase", "cx"]
    dag = circuit_to_dag(with_phase)
    assert [node.op.name for node in dag.op_nodes() if not node.qargs] == [
        "global_phase"
    ]
    # ... but layers() drops it, which is why only the linear estimator sees it
    assert [
        node.op.name
        for layer in dag.layers()
        for node in layer["graph"].op_nodes()
        if not node.qargs
    ] == []

    for estimator in (
        estimate_parallel_makespan,
        estimate_parallel_makespan_layered,
        estimate_parallel_makespan_topology,
    ):
        baseline = estimator(plain, arch, latency)
        with_zero_qubit_op = estimator(with_phase, arch, latency)
        assert with_zero_qubit_op.makespan == pytest.approx(baseline.makespan)
        assert with_zero_qubit_op.makespan == pytest.approx(latency.twoq)
        assert with_zero_qubit_op.remote_ops == baseline.remote_ops == 0


def test_zero_async_overlap_charges_the_full_classical_round_trip() -> None:
    """`async_overlap` hides part of the classical RTT; at 0.0 none of it is hidden."""
    overlapped_cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_overlap=1.0,
    )
    plain_cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_overlap=0.0,
    )
    latency = LatencyModel()

    def remote_makespan(cfg: MultiQPUConfig) -> float:
        arch = MultiQPUArchitecture(cfg)
        circuit = QuantumCircuit(arch.n_phys)
        circuit.cx(
            arch.block_of_qpu(0).compute[0],
            arch.block_of_qpu(1).compute[0],
        )
        return estimate_parallel_makespan_topology(circuit, arch, latency).makespan

    full_rtt = remote_makespan(plain_cfg)
    hidden_rtt = remote_makespan(overlapped_cfg)

    assert full_rtt == pytest.approx(
        latency.epr_gen + latency.classical_rtt + latency.remote_gate_overhead
    )
    assert hidden_rtt == pytest.approx(full_rtt - latency.classical_rtt)

    # `async_classical=False` skips the overlap arithmetic entirely: no part of
    # the round trip is hidden, whatever async_overlap says.
    synchronous_cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
        async_classical=False,
        async_overlap=1.0,
    )
    assert remote_makespan(synchronous_cfg) == pytest.approx(full_rtt)


# ---------------------------------------------------------------------------
# Auditing a finished schedule plan
# ---------------------------------------------------------------------------


def _audited_case(**overrides):
    """A compiled plan with remote rounds, plus what it takes to audit it."""
    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    settings = dict(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
        optimization_level=0,
    )
    settings.update(overrides)
    cfg = MultiQPUConfig(**settings)
    arch = MultiQPUArchitecture(cfg)
    model = LatencyModel()
    qc = random_benchmark_circuit(n_logical=9, depth=8, seed=0)
    res = compile_distributed(qc, cfg, model, seed=0, strategy="tpccap_sa")
    assert any(layer.remote_rounds for layer in res.schedule_plan.layers)
    return res, arch, model


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"inter_topology": "switch", "switch_reconfig_delay": 25.0},
        {"inter_topology": "mesh", "comm_qubits_per_qpu": 2, "link_capacity": 2},
        {"inter_topology": "clos", "n_qpus": 4, "comm_qubits_per_qpu": 2},
        {"inter_topology": "fat_tree", "n_qpus": 4, "compute_qubits_per_qpu": 4},
        {"async_classical": False},
    ],
    ids=["ring", "switch-reconfig", "mesh-2ports", "clos", "fat-tree", "sync-rtt"],
)
def test_estimator_output_survives_an_independent_re_derivation(overrides):
    """Every figure in the plan is rebuilt from the outside and must match."""
    from quport.schedule import audit_topology_schedule_plan

    res, arch, model = _audited_case(**overrides)

    assert (
        audit_topology_schedule_plan(
            res.schedule_plan, arch, model, res.physical_circuit
        )
        == ()
    )


def _first_round_layer(plan):
    return next(i for i, layer in enumerate(plan.layers) if layer.remote_rounds)


def _replace_layer(plan, index, **changes):
    import dataclasses

    layers = list(plan.layers)
    layers[index] = dataclasses.replace(layers[index], **changes)
    return dataclasses.replace(plan, layers=tuple(layers))


def _replace_first_round(plan, index, **changes):
    import dataclasses

    layer = plan.layers[index]
    rounds = list(layer.remote_rounds)
    rounds[0] = dataclasses.replace(rounds[0], **changes)
    return _replace_layer(plan, index, remote_rounds=tuple(rounds))


def test_audit_catches_a_plan_that_does_not_add_up():
    """An audit that never fails is not an audit.

    Each corruption is one a real bug could produce: a slipped interval, a
    miscounted aggregate, or a round claiming resources it did not use.
    """
    import dataclasses

    from quport.schedule import audit_topology_schedule_plan

    res, arch, model = _audited_case()
    plan = res.schedule_plan
    index = _first_round_layer(plan)
    layer = plan.layers[index]
    summary = plan.summary

    cases = {
        "starts at": _replace_layer(plan, index, start_time=layer.start_time + 1.0),
        "not start + duration": _replace_layer(
            plan, index, duration=layer.duration + 5.0
        ),
        "makespan": dataclasses.replace(
            plan, summary=dataclasses.replace(summary, makespan=summary.makespan + 1.0)
        ),
        "remote ops, the trace holds": dataclasses.replace(
            plan,
            summary=dataclasses.replace(summary, remote_ops=summary.remote_ops - 1),
        ),
        "rounds, the trace has": dataclasses.replace(
            plan,
            summary=dataclasses.replace(
                summary, remote_rounds=summary.remote_rounds + 3
            ),
        ),
        "budget is": _replace_first_round(
            plan,
            index,
            qpu_ports_used=tuple(99 for _ in layer.remote_rounds[0].qpu_ports_used),
        ),
        "pairs consume": _replace_first_round(plan, index, link_utilization=()),
        "peak_link_util": dataclasses.replace(
            plan,
            summary=dataclasses.replace(
                summary, peak_link_util=summary.peak_link_util + 7
            ),
        ),
        "peak_qpu_ports_used": dataclasses.replace(
            plan,
            summary=dataclasses.replace(
                summary, peak_qpu_ports_used=summary.peak_qpu_ports_used + 7
            ),
        ),
    }

    for expected, broken in cases.items():
        problems = audit_topology_schedule_plan(broken, arch, model)
        assert problems, f"audit missed: {expected}"
        assert any(
            expected in problem for problem in problems
        ), f"audit reported {problems} for {expected!r}"


def test_audit_catches_a_round_that_costs_nothing():
    from quport.schedule import audit_topology_schedule_plan

    res, arch, model = _audited_case()
    plan = res.schedule_plan
    index = _first_round_layer(plan)

    problems = audit_topology_schedule_plan(
        _replace_first_round(plan, index, duration=0.0), arch, model
    )
    assert problems


def test_audit_checks_the_plan_against_the_circuit_it_came_from():
    """Only the circuit can say whether the plan accounted for every remote op."""
    from qiskit import QuantumCircuit as QC

    from quport.schedule import audit_topology_schedule_plan

    res, arch, model = _audited_case()

    # Consistent with itself, but describing a different circuit.
    other = QC(res.physical_circuit.num_qubits)
    assert audit_topology_schedule_plan(res.schedule_plan, arch, model, other)
    assert (
        audit_topology_schedule_plan(res.schedule_plan, arch, model, other)[0].count(
            "spanning more than one QPU"
        )
        == 1
    )


def test_audit_accepts_a_plan_with_no_remote_operations():
    from quport.schedule import audit_topology_schedule_plan

    cfg = MultiQPUConfig(
        n_qpus=2, compute_qubits_per_qpu=4, comm_qubits_per_qpu=1, optimization_level=0
    )
    arch = MultiQPUArchitecture(cfg)
    model = LatencyModel()

    circuit = QuantumCircuit(cfg.total_physical_qubits())
    circuit.h(0)
    circuit.cx(0, 1)

    plan = estimate_topology_schedule_plan(circuit, arch, model)
    assert plan.summary.remote_ops == 0
    assert audit_topology_schedule_plan(plan, arch, model, circuit) == ()


def test_audit_accepts_unschedulable_penalty_rounds():
    """Zero comm ports makes every remote op a penalty round; that is still sound."""
    from quport.schedule import audit_topology_schedule_plan

    cfg = MultiQPUConfig(
        n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=0, optimization_level=0
    )
    arch = MultiQPUArchitecture(cfg)
    model = LatencyModel()

    circuit = QuantumCircuit(cfg.total_physical_qubits())
    circuit.cx(0, 2)

    plan = estimate_topology_schedule_plan(circuit, arch, model)
    assert plan.summary.remote_ops == 1
    assert any(
        rnd.unschedulable_ops for layer in plan.layers for rnd in layer.remote_rounds
    )
    assert audit_topology_schedule_plan(plan, arch, model, circuit) == ()


def test_a_round_reports_only_the_links_its_operations_use() -> None:
    """Probing a link for capacity must not enter it in the round's usage.

    Testing whether an operation fits reads the load on every link of its path.
    Reading through a ``defaultdict`` would insert each probed link at zero, and
    those phantom entries reach ``link_utilization`` -- which is meant to say
    what the round's *placed* operations consume, and which
    :func:`~quport.schedule.audit_topology_schedule_plan` re-derives from those
    operations alone.
    """
    from quport.schedule import audit_topology_schedule_plan

    # A ring long enough that paths overlap and capacity is genuinely contested,
    # which is what makes an operation get deferred after its links are probed.
    cfg = MultiQPUConfig(
        n_qpus=12,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=2,
        inter_topology="ring",
        link_capacity=1,
    )
    arch = MultiQPUArchitecture(cfg)
    model = LatencyModel()

    mapped = QuantumCircuit(arch.n_phys)
    block = arch.block
    for offset in range(6):
        mapped.cx(offset * block, ((offset + 6) % 12) * block)

    plan = estimate_topology_schedule_plan(mapped, arch, model)

    deferred = [
        rnd
        for layer in plan.layers
        for rnd in layer.remote_rounds
        if not rnd.unschedulable_ops
    ]
    assert deferred, "the fixture must produce real rounds"
    for rnd in deferred:
        assert all(count > 0 for _edge, count in rnd.link_utilization), (
            f"round {rnd.round_index} reports an unused link: "
            f"{[edge for edge, count in rnd.link_utilization if count == 0]}"
        )

    assert audit_topology_schedule_plan(plan, arch, model, mapped) == ()


def test_a_pair_routes_the_same_way_whichever_operand_leads() -> None:
    """An undirected pair's route cannot depend on the orientation it arrives in.

    A BFS next-hop table need not be reversal-symmetric: on a circulant graph
    ``path_edges(sp, a, b)`` and ``path_edges(sp, b, a)`` name different links of
    the same length. Routing whichever orientation happened to appear first
    would make an operation's links depend on incidental gate order, and put the
    trace at odds with every other reader of the same pair.
    """
    from quport.network import path_edges
    from quport.schedule import audit_topology_schedule_plan

    cfg = MultiQPUConfig(
        n_qpus=50,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=2,
        inter_topology="degree_d",
        inter_degree=4,
    )
    arch = MultiQPUArchitecture(cfg)
    shortest = arch.qpu_shortest_paths()

    asymmetric = [
        (a, b)
        for a in range(cfg.n_qpus)
        for b in range(a + 1, cfg.n_qpus)
        if set(path_edges(shortest, a, b)) != set(path_edges(shortest, b, a))
    ]
    assert asymmetric, "the fixture must have direction-dependent shortest paths"

    # Drive one such pair from the high-numbered operand, the orientation that
    # used to decide the cached route.
    high, low = asymmetric[0][1], asymmetric[0][0]
    block = arch.block
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(high * block, low * block)
    mapped.cx(low * block, high * block)

    model = LatencyModel()
    plan = estimate_topology_schedule_plan(mapped, arch, model)
    assert audit_topology_schedule_plan(plan, arch, model, mapped) == ()

    routed = {
        edge
        for layer in plan.layers
        for rnd in layer.remote_rounds
        for edge, _count in rnd.link_utilization
    }
    assert routed == set(path_edges(shortest, low, high))
