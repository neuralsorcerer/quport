# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end coverage of the entanglement-aware objective and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from typer.testing import CliRunner

from quport.architecture import MultiQPUArchitecture
from quport.cli import app
from quport.compiler import compile_distributed
from quport.config import LatencyModel, MultiQPUConfig, validate_epr_success_prob
from quport.hypergraph import build_distributable_packets, ebit_cost
from quport.partition import tpccap_partition, tpccap_sa_partition
from quport.pipeline import (
    benchmark_method_labels,
    benchmark_random_circuits,
    map_and_transpile,
    random_benchmark_circuit,
)


def _cfg(**overrides: object) -> MultiQPUConfig:
    settings: dict[str, object] = {
        "n_qpus": 3,
        "compute_qubits_per_qpu": 3,
        "comm_qubits_per_qpu": 1,
        "optimization_level": 0,
    }
    settings.update(overrides)
    return MultiQPUConfig(**settings)  # type: ignore[arg-type]


def _star_circuit(n_qubits: int) -> QuantumCircuit:
    """A control shared by every target: the aggregation-friendly extreme."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for target in range(1, n_qubits):
        qc.cx(0, target)
    return qc


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------


def test_expected_epr_time_scales_with_hops_and_attempts() -> None:
    model = LatencyModel(epr_gen=100.0)

    assert model.expected_epr_time(0) == pytest.approx(0.0)
    assert model.expected_epr_time(1) == pytest.approx(100.0)
    assert model.expected_epr_time(3) == pytest.approx(300.0)
    assert LatencyModel(epr_gen=100.0, epr_success_prob=0.5).expected_epr_time(
        2
    ) == pytest.approx(400.0)


def test_expected_epr_time_rejects_invalid_hops() -> None:
    model = LatencyModel()
    with pytest.raises(ValueError, match="hops must be non-negative"):
        model.expected_epr_time(-1)
    with pytest.raises(ValueError, match="hops must be a non-negative integer"):
        model.expected_epr_time(1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, float("inf")])
def test_validate_epr_success_prob_rejects_out_of_range(value: float) -> None:
    with pytest.raises(ValueError, match="epr_success_prob"):
        validate_epr_success_prob(value)


def test_validate_epr_success_prob_accepts_the_closed_upper_bound() -> None:
    assert validate_epr_success_prob(1.0) == 1.0
    assert validate_epr_success_prob(0.25) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Partitioner objective
# ---------------------------------------------------------------------------


def _sp(cfg: MultiQPUConfig) -> object:
    return MultiQPUArchitecture(cfg).qpu_shortest_paths()


def test_w_ebit_defaults_leave_the_historical_objective_untouched() -> None:
    cfg = _cfg()
    qc = _star_circuit(7)
    weights = {(0, target): 1.0 for target in range(1, 7)}
    packets = build_distributable_packets(qc)

    without, diag_without = tpccap_partition(
        n=7,
        weights=weights,
        n_qpus=cfg.n_qpus,
        capacity=cfg.capacity_per_qpu(),
        comm_ports_per_qpu=cfg.comm_qubits_per_qpu,
        sp=_sp(cfg),  # type: ignore[arg-type]
        seed=3,
    )
    diagnostic_only, diag_diagnostic = tpccap_partition(
        n=7,
        weights=weights,
        n_qpus=cfg.n_qpus,
        capacity=cfg.capacity_per_qpu(),
        comm_ports_per_qpu=cfg.comm_qubits_per_qpu,
        sp=_sp(cfg),  # type: ignore[arg-type]
        seed=3,
        packets=packets,
    )

    # Supplying packets with w_ebit == 0 reports e-bits without steering search.
    assert without.part == diagnostic_only.part
    assert diag_without.ebits == 0
    assert diag_without.weighted_ebit_distance == 0.0
    assert diag_diagnostic.ebits == ebit_cost(packets, without.part, cfg.n_qpus)


def test_ebit_objective_beats_cut_weight_on_a_shared_control() -> None:
    """Cut weight cannot see that one cat copy serves a whole fan-out.

    Six targets of one control are split across three QPUs of capacity four.
    Every placement cuts the same number of gates, but grouping the targets onto
    as few remote QPUs as possible is what actually saves EPR pairs.
    """
    cfg = _cfg(n_qpus=3, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1)
    qc = _star_circuit(12)
    weights = {(0, target): 1.0 for target in range(1, 12)}
    packets = build_distributable_packets(qc)

    cut_driven, _ = tpccap_sa_partition(
        n=12,
        weights=weights,
        n_qpus=cfg.n_qpus,
        capacity=cfg.capacity_per_qpu(),
        comm_ports_per_qpu=cfg.comm_qubits_per_qpu,
        sp=_sp(cfg),  # type: ignore[arg-type]
        seed=1,
        steps=200,
    )[:2]
    ebit_driven, ebit_diag, _ = tpccap_sa_partition(
        n=12,
        weights=weights,
        n_qpus=cfg.n_qpus,
        capacity=cfg.capacity_per_qpu(),
        comm_ports_per_qpu=cfg.comm_qubits_per_qpu,
        sp=_sp(cfg),  # type: ignore[arg-type]
        seed=1,
        steps=200,
        w_dist=0.0,
        packets=packets,
        w_ebit=1.0,
    )

    cut_ebits = ebit_cost(packets, cut_driven.part, cfg.n_qpus)
    ebit_ebits = ebit_cost(packets, ebit_driven.part, cfg.n_qpus)

    assert ebit_diag.ebits == ebit_ebits
    assert ebit_ebits <= cut_ebits
    # Twelve qubits over three QPUs of capacity four: at best the control shares
    # a QPU with three targets and the rest need one copy per remote QPU.
    assert ebit_ebits == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"w_ebit": -1.0}, "w_ebit must be non-negative"),
        ({"w_ebit": float("nan")}, "w_ebit must be finite"),
        ({"packets": object()}, "packets must be a PacketDecomposition"),
    ],
)
def test_partitioners_validate_entanglement_arguments(
    kwargs: dict[str, object], message: str
) -> None:
    cfg = _cfg()
    common = {
        "n": 3,
        "weights": {(0, 1): 1.0},
        "n_qpus": cfg.n_qpus,
        "capacity": cfg.capacity_per_qpu(),
        "comm_ports_per_qpu": cfg.comm_qubits_per_qpu,
        "sp": _sp(cfg),
    }

    with pytest.raises(ValueError, match=message):
        tpccap_partition(**common, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        tpccap_sa_partition(**common, steps=1, **kwargs)  # type: ignore[arg-type]


def test_packets_must_describe_the_same_qubit_count() -> None:
    cfg = _cfg()
    packets = build_distributable_packets(QuantumCircuit(5))

    with pytest.raises(ValueError, match="same number of logical qubits"):
        tpccap_partition(
            n=3,
            weights={(0, 1): 1.0},
            n_qpus=cfg.n_qpus,
            capacity=cfg.capacity_per_qpu(),
            comm_ports_per_qpu=cfg.comm_qubits_per_qpu,
            sp=_sp(cfg),  # type: ignore[arg-type]
            packets=packets,
        )


# ---------------------------------------------------------------------------
# Pipeline and compiler integration
# ---------------------------------------------------------------------------


def test_ebit_strategy_is_available_end_to_end() -> None:
    cfg = _cfg()
    qc = random_benchmark_circuit(8, 8, 4)

    result = map_and_transpile(qc, cfg, seed=4, strategy="ebit")

    assert result.strategy == "ebit"
    assert result.partition_diagnostics is not None
    assert result.partition_diagnostics.ebits >= 0
    assert len(result.partition) == 8


def test_unknown_strategy_message_lists_ebit() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="'ebit'"):
        map_and_transpile(random_benchmark_circuit(2, 1, 0), cfg, strategy="nope")


def test_benchmark_method_ids_include_ebit() -> None:
    labels = benchmark_method_labels()

    assert labels[5.0] == "ebit"
    rows = benchmark_random_circuits(
        _cfg(), n_logical=5, depth=3, trials=1, seed=0, strategies=("ebit",)
    )
    assert [row["method"] for row in rows] == [5.0]
    assert [row["strategy"] for row in rows] == ["ebit"]


def test_compile_distributed_reports_consistent_entanglement_accounting() -> None:
    cfg = _cfg(n_qpus=4, compute_qubits_per_qpu=3, comm_qubits_per_qpu=2)
    qc = random_benchmark_circuit(10, 10, 6)

    result = compile_distributed(qc, cfg, seed=6, strategy="ebit")

    plan = result.aggregation
    assert plan.epr_pairs == sum(block.epr_pairs for block in plan.blocks)
    assert plan.epr_pairs <= plan.baseline_epr_pairs
    assert result.entanglement_schedule.epr_pairs == plan.epr_pairs
    assert result.ebits.ebits == ebit_cost(result.packets, result.partition, cfg.n_qpus)
    # The partition-time model bounds the compile-time plan from below: ports can
    # only force extra transactions, never remove them.
    assert result.ebits.ebits <= plan.epr_pairs
    assert all(peak <= cfg.comm_qubits_per_qpu for peak in plan.peak_cat_copies)


def test_compile_distributed_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        compile_distributed(
            random_benchmark_circuit(2, 1, 0), _cfg(), strategy="not-a-strategy"
        )


def test_ebit_strategy_beats_cut_minimisation_at_its_own_objective() -> None:
    """The e-bit strategy has to actually spend fewer EPR pairs.

    It did not, before the penalty weights were rescaled: ``w_port`` measures
    squared boundary-qubit overflow, which on these instances is one to two
    orders of magnitude larger than an e-bit count, so removing the cut-distance
    term left the port penalty steering the entire search and the strategy lost
    to plain cut minimisation at the objective it was named for.
    """
    cfg = _cfg(n_qpus=4, compute_qubits_per_qpu=4, comm_qubits_per_qpu=2)
    qc = random_benchmark_circuit(n_logical=16, depth=12, seed=0)

    baseline = compile_distributed(qc, cfg, seed=0, strategy="tpccap_sa")
    entangled = compile_distributed(qc, cfg, seed=0, strategy="ebit")

    assert entangled.ebits.ebits < baseline.ebits.ebits
    assert entangled.aggregation.epr_pairs < baseline.aggregation.epr_pairs


def test_ebit_strategy_lands_near_the_proved_optimum() -> None:
    """Calibrated against exhaustive-search-verified branch and bound.

    A generous ceiling: the point is to catch a regression that puts the search
    back into the wrong basin, not to pin a tuning result. `partition_gap` also
    raises outright if the heuristic scores below the optimum.
    """
    from quport.exact import partition_gap

    cfg = _cfg(n_qpus=3, compute_qubits_per_qpu=3, comm_qubits_per_qpu=2)
    capacity = cfg.capacity_per_qpu()

    for seed in range(3):
        qc = random_benchmark_circuit(n_logical=9, depth=10, seed=seed)
        result = compile_distributed(qc, cfg, seed=seed, strategy="ebit")
        gap = partition_gap(
            result.partition,
            cfg.n_qpus,
            capacity,
            objective="ebits",
            packets=result.packets,
        )
        assert gap.proved_optimal
        assert gap.relative <= 0.35


def test_aggregation_saves_pairs_on_a_shared_control_circuit() -> None:
    cfg = _cfg(n_qpus=2, compute_qubits_per_qpu=4, comm_qubits_per_qpu=1)

    result = compile_distributed(_star_circuit(8), cfg, seed=0, strategy="ebit")

    assert result.aggregation.baseline_epr_pairs > result.aggregation.epr_pairs
    assert result.aggregation.reduction > 0.0
    assert result.ebits.reduction > 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "n_qpus": 2,
                "compute_qubits_per_qpu": 2,
                "comm_qubits_per_qpu": 1,
                "optimization_level": 0,
            }
        ),
        encoding="utf-8",
    )


def test_ebits_command_reports_and_writes_a_plan(tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    _write_config(config)
    out = tmp_path / "plan.json"

    _run(
        [
            "ebits",
            "--n-logical",
            "4",
            "--depth",
            "3",
            "--config",
            str(config),
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"aggregation", "ebits", "schedule"}
    assert payload["schedule"]["epr_pairs"] == payload["aggregation"]["epr_pairs"]
    assert payload["aggregation"]["epr_pairs"] <= (
        payload["aggregation"]["baseline_epr_pairs"]
    )


def test_ebits_command_emits_and_verifies_the_protocol(tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    _write_config(config)
    qasm = tmp_path / "telegate.qasm"

    result = CliRunner().invoke(
        app,
        [
            "ebits",
            "--n-logical",
            "4",
            "--depth",
            "4",
            "--seed",
            "2",
            "--config",
            str(config),
            "--verify",
            "--emit-qasm",
            str(qasm),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Verified" in result.output
    source = qasm.read_text(encoding="utf-8")
    assert source.startswith("OPENQASM 3.0;")
    # The emitted program carries real feedforward, not just gate names.
    assert "if (" in source


def test_ebits_command_reports_a_plan_it_cannot_verify(tmp_path: Path) -> None:
    """A port-less architecture has no runnable protocol; say so, do not crash."""
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "n_qpus": 2,
                "compute_qubits_per_qpu": 3,
                "comm_qubits_per_qpu": 0,
                "optimization_level": 0,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ebits",
            "--n-logical",
            "6",
            "--depth",
            "8",
            "--config",
            str(config),
            "--verify",
        ],
    )

    assert result.exit_code != 0
    assert "Cannot verify" in result.output


def test_ebits_command_requires_a_circuit_source() -> None:
    result = CliRunner().invoke(app, ["ebits"])

    assert result.exit_code != 0
    assert "--n-logical is required" in result.output
