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
from quport.config import LatencyModel, MultiQPUConfig
from quport.pipeline import random_benchmark_circuit
from quport.schedule import estimate_entanglement_schedule

UNBOUNDED_PORTS = 1_000_000


def _arch(**overrides: object) -> MultiQPUArchitecture:
    settings: dict[str, object] = {
        "n_qpus": 2,
        "compute_qubits_per_qpu": 3,
        "comm_qubits_per_qpu": 1,
        "intra_topology": "clique",
        "inter_topology": "switch",
    }
    settings.update(overrides)
    return MultiQPUArchitecture(MultiQPUConfig(**settings))  # type: ignore[arg-type]


def test_timings_match_the_protocol_by_hand() -> None:
    """One cat block, two gates, default latencies -- every term checked.

    With ``async_overlap=0.5`` the effective classical round trip is 10. The
    block therefore runs: distribute (200) + entangler classical message (10) +
    protocol overhead (50) = 260 before the first gate; two local two-qubit
    gates at 10 each; then a disentangler message of 10.
    """
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.cx(0, 5)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    assert summary.blocks == 1
    assert summary.epr_pairs == 1
    assert summary.remote_gates == 2
    assert summary.unschedulable_gates == 0
    assert summary.makespan == pytest.approx(260.0 + 10.0 + 10.0 + 10.0)
    assert summary.entanglement_time == pytest.approx(200.0)
    assert summary.link_busy_time == (((0, 1), 200.0),)
    assert summary.peak_ports_in_use == (1, 1)
    assert summary.port_busy_time == pytest.approx((260.0, 290.0))
    assert summary.qpu_busy_time == pytest.approx((0.0, 20.0))


def test_aggregation_beats_one_transaction_per_gate() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.cx(0, 5)

    aggregated = estimate_entanglement_schedule(qc, arch, LatencyModel())
    per_gate = estimate_entanglement_schedule(
        qc,
        arch,
        LatencyModel(),
        plan=aggregate_remote_operations(qc, arch, max_block_gates=1),
    )

    assert per_gate.epr_pairs == 2
    assert aggregated.epr_pairs == 1
    assert aggregated.makespan < per_gate.makespan


def test_epr_success_probability_scales_distribution_time() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)

    deterministic = estimate_entanglement_schedule(qc, arch, LatencyModel())
    heralded = estimate_entanglement_schedule(
        qc, arch, LatencyModel(epr_success_prob=0.25)
    )

    assert heralded.entanglement_time == pytest.approx(
        deterministic.entanglement_time * 4.0
    )
    assert heralded.makespan == pytest.approx(
        deterministic.makespan + 3.0 * deterministic.entanglement_time
    )


@pytest.mark.parametrize("probability", [0.0, -0.5, 1.5])
def test_invalid_success_probability_is_rejected(probability: float) -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)

    with pytest.raises(ValueError, match="epr_success_prob"):
        estimate_entanglement_schedule(
            qc, arch, LatencyModel(epr_success_prob=probability)
        )


def test_distance_scales_entanglement_cost_on_a_ring() -> None:
    arch = _arch(n_qpus=4, inter_topology="ring")
    near = QuantumCircuit(arch.n_phys)
    near.cx(0, 4)  # QPU 0 -> QPU 1, one hop
    far = QuantumCircuit(arch.n_phys)
    far.cx(0, 8)  # QPU 0 -> QPU 2, two hops on a 4-QPU ring

    near_summary = estimate_entanglement_schedule(near, arch, LatencyModel())
    far_summary = estimate_entanglement_schedule(far, arch, LatencyModel())

    assert far_summary.makespan == pytest.approx(near_summary.makespan + 200.0)
    # Two hops occupy two links for the full distribution window.
    assert far_summary.entanglement_time == pytest.approx(800.0)
    assert len(far_summary.link_busy_time) == 2


def test_scarce_ports_serialise_independent_roots() -> None:
    arch = _arch(comm_qubits_per_qpu=1)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.cx(1, 5)

    scarce = estimate_entanglement_schedule(qc, arch, LatencyModel())
    roomy = estimate_entanglement_schedule(
        qc, arch, LatencyModel(), ports_per_qpu=UNBOUNDED_PORTS
    )

    assert scarce.peak_ports_in_use[1] == 1
    assert roomy.peak_ports_in_use[1] == 2
    assert roomy.makespan < scarce.makespan


def test_link_capacity_serialises_concurrent_entanglement() -> None:
    narrow = _arch(comm_qubits_per_qpu=2, link_capacity=1)
    wide = _arch(comm_qubits_per_qpu=2, link_capacity=2)
    # With two comm ports per QPU the block size is five, so QPU 1 starts at
    # physical qubit 5.
    qc = QuantumCircuit(narrow.n_phys)
    qc.cx(0, 5)
    qc.cx(1, 6)

    narrow_summary = estimate_entanglement_schedule(qc, narrow, LatencyModel())
    wide_summary = estimate_entanglement_schedule(qc, wide, LatencyModel())

    assert narrow_summary.blocks == wide_summary.blocks == 2
    # A single channel forces the second distribution to wait for the first.
    assert narrow_summary.makespan == pytest.approx(480.0)
    assert wide_summary.makespan == pytest.approx(280.0)
    assert narrow_summary.entanglement_time == wide_summary.entanglement_time


def test_unreachable_qpus_are_reported_as_unschedulable() -> None:
    arch = _arch(inter_topology="degree_d", inter_degree=0)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    assert summary.unschedulable_gates == 1
    assert summary.makespan > 1e8
    assert summary.link_busy_time == ()


def test_zero_link_capacity_is_reported_as_unschedulable() -> None:
    arch = _arch(link_capacity=0)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    assert summary.unschedulable_gates == 1
    assert summary.makespan > 1e8


def test_zero_ports_leaves_every_remote_gate_unserved() -> None:
    arch = _arch(comm_qubits_per_qpu=0)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 3)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    assert summary.blocks == 0
    assert summary.unschedulable_gates == 1
    assert summary.makespan > 1e8


def test_local_gates_on_separate_qpus_run_in_parallel() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 1)
    qc.cx(4, 5)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    assert summary.makespan == pytest.approx(10.0)
    assert summary.qpu_busy_time == pytest.approx((10.0, 10.0))
    assert summary.epr_pairs == 0


def test_barriers_synchronise_the_qubits_they_span() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 1)
    qc.barrier(0, 1, 4)
    qc.cx(4, 5)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    # The barrier drags qubit 4 to the end of QPU 0's first gate.
    assert summary.makespan == pytest.approx(20.0)


def test_teleport_block_pays_for_the_return_trip() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.swap(0, 4)

    summary = estimate_entanglement_schedule(qc, arch, LatencyModel())

    assert summary.epr_pairs == 2
    # Two distribution windows on the single link, one out and one back.
    assert summary.entanglement_time == pytest.approx(400.0)


@pytest.mark.parametrize("seed", range(5))
def test_port_budget_is_never_exceeded(seed: int) -> None:
    from quport.compiler import compile_distributed

    ports = 1 + (seed % 2)
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=ports,
        inter_topology="ring",
    )
    qc = random_benchmark_circuit(10, 10, seed)
    result = compile_distributed(qc, cfg, seed=seed)
    arch = MultiQPUArchitecture(cfg)

    summary = estimate_entanglement_schedule(
        result.physical_circuit, arch, LatencyModel()
    )

    assert all(peak <= ports for peak in summary.peak_ports_in_use)
    assert summary.unschedulable_gates == 0
    assert summary.epr_pairs == result.aggregation.epr_pairs
    # Each QPU has `ports` ports, so its total port occupancy is bounded by
    # ports * makespan. Gate time carries no such bound: a QPU runs gates on
    # disjoint qubits concurrently and each one is counted.
    assert all(
        busy <= ports * summary.makespan + 1e-9 for busy in summary.port_busy_time
    )


def test_summary_serializes_to_standards_compliant_json() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)

    payload = estimate_entanglement_schedule(qc, arch, LatencyModel()).to_dict()
    text = json.dumps(payload, allow_nan=False)

    restored = json.loads(text)
    assert restored["epr_pairs"] == 1
    assert restored["link_busy_time"][0]["edge"] == [0, 1]


def test_plan_built_for_a_larger_port_budget_is_rejected() -> None:
    """A mismatched plan is a clear error, not a silent unschedulable penalty."""
    arch = _arch(comm_qubits_per_qpu=1)
    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 4)
    qc.cx(1, 5)

    roomy_plan = aggregate_remote_operations(qc, arch, ports_per_qpu=UNBOUNDED_PORTS)

    with pytest.raises(ValueError, match="exceeds the schedule's comm-port budget"):
        estimate_entanglement_schedule(qc, arch, LatencyModel(), plan=roomy_plan)

    # Supplying the matching budget schedules the same plan without complaint.
    summary = estimate_entanglement_schedule(
        qc, arch, LatencyModel(), plan=roomy_plan, ports_per_qpu=UNBOUNDED_PORTS
    )
    assert summary.unschedulable_gates == 0


def test_schedule_rejects_invalid_arguments() -> None:
    arch = _arch()
    qc = QuantumCircuit(arch.n_phys)

    with pytest.raises(ValueError, match="plan must be an AggregationPlan"):
        estimate_entanglement_schedule(qc, arch, LatencyModel(), plan=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer or a sequence"):
        estimate_entanglement_schedule(qc, arch, LatencyModel(), ports_per_qpu="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length must match n_qpus"):
        estimate_entanglement_schedule(qc, arch, LatencyModel(), ports_per_qpu=[1])


# ---------------------------------------------------------------------------
# Auditing a finished entanglement schedule
# ---------------------------------------------------------------------------


def _scheduled_case(**overrides):
    """A compiled schedule with real EPR traffic, plus its inputs."""
    from quport.aggregation import aggregate_remote_operations
    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    settings = dict(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=2,
        inter_topology="ring",
        optimization_level=0,
    )
    settings.update(overrides)
    cfg = MultiQPUConfig(**settings)
    arch = MultiQPUArchitecture(cfg)
    model = LatencyModel()
    qc = random_benchmark_circuit(n_logical=9, depth=10, seed=0)
    res = compile_distributed(qc, cfg, model, seed=0, strategy="tpccap_sa")
    plan = aggregate_remote_operations(
        res.physical_circuit, arch, ports_per_qpu=cfg.comm_qubits_per_qpu
    )
    summary = estimate_entanglement_schedule(
        res.physical_circuit, arch, model, plan=plan
    )
    assert summary.link_busy_time, "the fixture must generate EPR traffic"
    return summary, res.physical_circuit, arch, model, plan


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"inter_topology": "switch", "comm_qubits_per_qpu": 1},
        {"inter_topology": "mesh", "n_qpus": 4, "link_capacity": 2},
        {"n_qpus": 4, "compute_qubits_per_qpu": 4, "comm_qubits_per_qpu": 1},
    ],
    ids=["ring", "switch-1port", "mesh-2channels", "wide"],
)
def test_schedule_satisfies_the_properties_it_must(overrides):
    from quport.schedule import audit_entanglement_schedule

    summary, mapped, arch, model, plan = _scheduled_case(**overrides)

    assert audit_entanglement_schedule(summary, mapped, arch, model, plan=plan) == ()


def test_audit_catches_a_schedule_that_cannot_be_true():
    """Each corruption breaks a property that holds by theorem, not convention."""
    import dataclasses

    from quport.schedule import audit_entanglement_schedule

    summary, mapped, arch, model, plan = _scheduled_case()
    edge, _busy = summary.link_busy_time[0]

    cases = {
        "cat copies at once": dataclasses.replace(
            summary, peak_ports_in_use=tuple(99 for _ in summary.peak_ports_in_use)
        ),
        "port-busy time": dataclasses.replace(
            summary,
            port_busy_time=tuple(
                summary.makespan * 1000.0 for _ in summary.port_busy_time
            ),
        ),
        "channels": dataclasses.replace(
            summary,
            link_busy_time=((edge, summary.makespan * 1000.0),)
            + summary.link_busy_time[1:],
        ),
        "per-link busy times sum": dataclasses.replace(
            summary, entanglement_time=summary.entanglement_time + 1.0
        ),
        "serialised on the busiest": dataclasses.replace(summary, makespan=1.0),
        "operations spanning more than one QPU": dataclasses.replace(
            summary, remote_gates=summary.remote_gates + 1
        ),
        "only": dataclasses.replace(
            summary, unschedulable_gates=summary.remote_gates + 1
        ),
        "the plan has": dataclasses.replace(summary, blocks=summary.blocks + 1),
        "the plan spends": dataclasses.replace(
            summary, epr_pairs=summary.epr_pairs + 1
        ),
        "entries for": dataclasses.replace(
            summary, qpu_busy_time=summary.qpu_busy_time[:-1]
        ),
    }

    for expected, broken in cases.items():
        problems = audit_entanglement_schedule(broken, mapped, arch, model, plan=plan)
        assert problems, f"audit missed: {expected}"
        assert any(
            expected in problem for problem in problems
        ), f"audit reported {problems} for {expected!r}"


def test_widening_a_resource_never_lengthens_one_fixed_plan():
    """Monotonicity is a property of the estimator, so it is checked directly.

    More ports and more link channels can only let an acquire return earlier, so
    every event time can only move earlier. A schedule that got *longer* when a
    resource was widened would mean the pools are not doing what they claim.
    """
    import dataclasses as dc

    from quport.aggregation import aggregate_remote_operations
    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        inter_topology="ring",
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    model = LatencyModel()
    qc = random_benchmark_circuit(n_logical=9, depth=10, seed=0)
    res = compile_distributed(qc, cfg, model, seed=0, strategy="tpccap_sa")
    mapped = res.physical_circuit
    plan = aggregate_remote_operations(mapped, arch, ports_per_qpu=1)

    previous = None
    for budget in (1, 2, 4, 8):
        makespan = estimate_entanglement_schedule(
            mapped, arch, model, plan=plan, ports_per_qpu=budget
        ).makespan
        if previous is not None:
            assert makespan <= previous + 1e-9
        previous = makespan

    previous = None
    for link_capacity in (1, 2, 4):
        wider = MultiQPUArchitecture(dc.replace(cfg, link_capacity=link_capacity))
        makespan = estimate_entanglement_schedule(
            mapped, wider, model, plan=plan, ports_per_qpu=1
        ).makespan
        if previous is not None:
            assert makespan <= previous + 1e-9
        previous = makespan


def test_slower_or_less_reliable_entanglement_never_shortens_a_schedule():
    from quport.aggregation import aggregate_remote_operations
    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=2,
        inter_topology="ring",
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    qc = random_benchmark_circuit(n_logical=9, depth=10, seed=0)
    res = compile_distributed(qc, cfg, LatencyModel(), seed=0, strategy="tpccap_sa")
    mapped = res.physical_circuit
    plan = aggregate_remote_operations(mapped, arch, ports_per_qpu=2)

    previous = None
    for epr_gen in (50.0, 200.0, 800.0):
        makespan = estimate_entanglement_schedule(
            mapped, arch, LatencyModel(epr_gen=epr_gen), plan=plan
        ).makespan
        if previous is not None:
            assert makespan >= previous - 1e-9
        previous = makespan

    previous = None
    for prob in (1.0, 0.5, 0.25):
        makespan = estimate_entanglement_schedule(
            mapped, arch, LatencyModel(epr_success_prob=prob), plan=plan
        ).makespan
        if previous is not None:
            assert makespan >= previous - 1e-9
        previous = makespan


def test_a_wide_gate_the_links_cannot_serve_counts_once() -> None:
    """Blocks are the plan's unit of work; ``unschedulable_gates`` counts gates.

    A three-qubit gate spanning three QPUs is gathered by two teleport blocks.
    With no link capacity neither can run, but that is one gate the schedule
    could not place, and the audit's ``unschedulable <= remote`` check says so.
    """
    from quport.schedule import audit_entanglement_schedule

    arch = _arch(n_qpus=3, compute_qubits_per_qpu=1, comm_qubits_per_qpu=2)
    starved = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=3,
            compute_qubits_per_qpu=1,
            comm_qubits_per_qpu=2,
            intra_topology="clique",
            inter_topology="switch",
            link_capacity=0,
        )
    )
    mapped = QuantumCircuit(arch.n_phys)
    mapped.ccx(0, 3, 6)

    plan = aggregate_remote_operations(mapped, starved)
    assert len(plan.blocks) == 2

    model = LatencyModel()
    summary = estimate_entanglement_schedule(mapped, starved, model, plan=plan)

    assert summary.remote_gates == 1
    assert summary.unschedulable_gates == 1
    assert audit_entanglement_schedule(summary, mapped, starved, model, plan=plan) == ()


def test_an_infeasible_cat_block_still_charges_every_gate_it_served() -> None:
    """One block, many gates: each gate it cannot serve is one gate lost."""
    arch = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=2,
            compute_qubits_per_qpu=2,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
            link_capacity=0,
        )
    )
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)
    mapped.cx(0, 4)

    plan = aggregate_remote_operations(mapped, arch)
    assert len(plan.blocks) == 1
    assert plan.blocks[0].size() == 2

    summary = estimate_entanglement_schedule(mapped, arch, LatencyModel(), plan=plan)
    assert summary.remote_gates == 2
    assert summary.unschedulable_gates == 2
