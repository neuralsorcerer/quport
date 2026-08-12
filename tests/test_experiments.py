# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from quport.config import MultiQPUConfig
from quport.experiments.bench import run_bench
from quport.experiments.sweep import run_sweep
from quport.pipeline import benchmark_method_labels


def _small_config() -> MultiQPUConfig:
    return MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        optimization_level=0,
    )


def test_run_bench_forwards_its_arguments_and_writes_the_csv(tmp_path: Path) -> None:
    """`quport.experiments.bench` ships in the package but nothing imported it.

    The wrapper's only job is forwarding, so a renamed keyword or a dropped
    argument would surface as a TypeError the first time a user called it.
    """
    out_csv = tmp_path / "bench.csv"

    # depth and trials differ so transposing them is visible in the row count.
    rows = run_bench(
        _small_config(),
        n_logical=4,
        depth=5,
        trials=2,
        seed=5,
        out_csv=str(out_csv),
    )

    labels = benchmark_method_labels()
    strategies = {str(row["strategy"]) for row in rows}
    assert strategies == {"baseline", "balanced", "tpccap"}
    assert len(rows) == 2 * len(strategies)
    assert {float(row["trial"]) for row in rows} == {0.0, 1.0}
    assert {float(row["seed"]) for row in rows} == {5.0, 6.0}
    for row in rows:
        assert labels[float(row["method"])] == row["strategy"]

    assert out_csv.exists()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    assert [r["strategy"] for r in written] == [str(r["strategy"]) for r in rows]


def test_run_bench_uses_the_default_latency_model(tmp_path: Path) -> None:
    """The wrapper supplies `LatencyModel()`; costs must be populated, not zeroed."""
    rows = run_bench(
        _small_config(),
        n_logical=4,
        depth=3,
        trials=1,
        seed=0,
        out_csv=str(tmp_path / "bench.csv"),
    )

    assert rows
    for row in rows:
        total = float(row["cost_total"])
        local = float(row["cost_local"])
        remote = float(row["cost_remote"])
        assert total > 0.0
        # total = local + remote + a non-negative depth penalty
        assert total >= local + remote
        assert local > 0.0


def test_run_sweep_writes_a_row_per_topology_and_strategy(tmp_path: Path) -> None:
    """`quport.experiments.sweep` was likewise never imported by the suite."""
    out_csv = tmp_path / "sweep.csv"

    run_sweep(n_logical=2, depth=1, trials=1, seed=0, out_csv=str(out_csv))

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = benchmark_method_labels()
    intra = {row["intra"] for row in rows}
    inter = {row["inter"] for row in rows}
    ports = {float(row["ports"]) for row in rows}
    methods = {labels[float(row["method"])] for row in rows}

    assert intra == {"clique", "line", "ring"}
    assert inter == {"switch", "ring", "degree_d", "clos"}
    assert ports == {1.0, 2.0}
    assert methods == {"baseline", "balanced", "tpccap"}
    assert len(rows) == len(intra) * len(inter) * len(ports) * len(methods)
    assert len(rows) == len(
        {(row["intra"], row["inter"], row["ports"], row["method"]) for row in rows}
    )


def test_wrappers_forward_every_argument_they_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both wrappers exist only to forward, and the sweep aggregates its trials away.

    Averaging hides a dropped `trials`, so assert on the delegated call itself
    rather than on the summary CSV.
    """
    import quport.experiments.bench as bench_module
    import quport.experiments.sweep as sweep_module

    seen: dict[str, object] = {}

    def fake_benchmark(cfg, n_logical, depth, trials, **kwargs):  # type: ignore[no-untyped-def]
        seen["bench"] = (cfg, n_logical, depth, trials, kwargs)
        return []

    def fake_sweep(**kwargs):  # type: ignore[no-untyped-def]
        seen["sweep"] = kwargs

    monkeypatch.setattr(bench_module, "benchmark_random_circuits", fake_benchmark)
    monkeypatch.setattr(sweep_module, "sweep_topologies", fake_sweep)

    cfg = _small_config()
    bench_module.run_bench(cfg, n_logical=4, depth=5, trials=2, seed=7, out_csv="b.csv")
    sweep_module.run_sweep(
        n_logical=3, depth=6, trials=4, seed=9, out_csv=str(tmp_path / "s.csv")
    )

    bench_cfg, n_logical, depth, trials, bench_kwargs = seen["bench"]
    assert bench_cfg is cfg
    assert (n_logical, depth, trials) == (4, 5, 2)
    assert bench_kwargs["seed"] == 7
    assert bench_kwargs["out_csv"] == "b.csv"
    assert bench_kwargs["latency"] is not None

    assert seen["sweep"] == {
        "n_logical": 3,
        "depth": 6,
        "trials": 4,
        "seed": 9,
        "out_csv": str(tmp_path / "s.csv"),
    }
