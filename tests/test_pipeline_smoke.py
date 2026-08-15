# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from quport.config import LatencyModel, MultiQPUConfig
from quport.pipeline import map_and_transpile, random_benchmark_circuit


def test_map_and_transpile_smoke() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=4,
        comm_qubits_per_qpu=1,
        intra_topology="ring",
        inter_topology="ring",
    )
    qc = random_benchmark_circuit(n_logical=6, depth=5, seed=1)
    res = map_and_transpile(
        qc, cfg, latency=LatencyModel(), seed=1, strategy="balanced"
    )
    assert res.mapped_circuit.num_qubits == cfg.total_physical_qubits()
    assert res.metrics.depth > 0


def test_baseline_partition_reports_qpu_assignments() -> None:
    from quport.pipeline import transpile_baseline

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    qc = random_benchmark_circuit(n_logical=4, depth=1, seed=7)

    res = transpile_baseline(qc, cfg, latency=LatencyModel(), seed=7)

    assert res.partition == [0, 0, 0, 1]
    assert all(0 <= qpu < cfg.n_qpus for qpu in res.partition)


def test_tpccap_sa_layout_uses_idle_comm_ports_for_capacity() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    qc = random_benchmark_circuit(n_logical=2, depth=0, seed=11)

    res = map_and_transpile(
        qc, cfg, latency=LatencyModel(), seed=11, strategy="tpccap_sa"
    )

    assert len(res.partition) == qc.num_qubits
    assert res.mapped_circuit.num_qubits == cfg.total_physical_qubits()


def test_benchmark_writes_header_for_zero_trials(tmp_path: Path) -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=2, comm_qubits_per_qpu=0)
    out = tmp_path / "empty.csv"

    rows = benchmark_random_circuits(
        cfg,
        n_logical=1,
        depth=0,
        trials=0,
        out_csv=str(out),
        strategies=("baseline",),
    )

    assert rows == []
    assert out.read_text(encoding="utf-8").startswith("trial,seed,method,strategy")


def test_sweep_writes_reproducible_topology_labels(tmp_path: Path) -> None:
    from quport.pipeline import sweep_topologies

    out = tmp_path / "sweep.csv"

    sweep_topologies(
        n_logical=1,
        depth=0,
        trials=0,
        seed=5,
        out_csv=str(out),
        intra_topologies=("clique",),
        inter_topologies=("switch",),
        comm_ports=(0,),
        compute_per_qpu=1,
        n_qpus=1,
    )

    csv_text = out.read_text(encoding="utf-8")
    assert "intra,inter,ports,method" in csv_text
    assert "clique,switch" in csv_text


def test_benchmark_rejects_negative_trials() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)

    with pytest.raises(ValueError, match="trials must be non-negative"):
        benchmark_random_circuits(cfg, n_logical=1, depth=0, trials=-1)


def test_benchmark_rejects_unknown_strategy() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)

    with pytest.raises(ValueError, match="Unknown benchmark strategies"):
        benchmark_random_circuits(
            cfg,
            n_logical=1,
            depth=0,
            trials=1,
            strategies=("baseline", "not-a-strategy"),
        )


def test_sweep_writes_header_when_all_configs_are_skipped(tmp_path: Path) -> None:
    from quport.pipeline import sweep_topologies

    out = tmp_path / "skipped.csv"

    sweep_topologies(
        n_logical=2,
        depth=0,
        trials=0,
        seed=5,
        out_csv=str(out),
        intra_topologies=("clique",),
        inter_topologies=("switch",),
        comm_ports=(0,),
        compute_per_qpu=1,
        n_qpus=1,
    )

    assert out.read_text(encoding="utf-8") == (
        "intra,inter,ports,method,swaps_mean,remote_2q_mean,depth_mean,"
        "cost_mean,cost_median,transpile_time_mean\n"
    )


def test_load_config_rejects_non_mapping_json(tmp_path: Path) -> None:
    from quport.config import load_config

    config_path = tmp_path / "bad.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping/object"):
        load_config(str(config_path))


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    from quport.config import load_config

    config_path = tmp_path / "bad.json"
    config_path.write_text('{"n_qpus": 2, "unknown": 3}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown field"):
        load_config(str(config_path))


def test_optional_module_available_handles_missing_parent() -> None:
    from quport.config import optional_module_available

    assert optional_module_available("quport_missing_dependency.child") is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_logical": -1, "depth": 0, "seed": 1}, "n_logical must be non-negative"),
        (
            {"n_logical": True, "depth": 0, "seed": 1},
            "n_logical must be a non-negative integer",
        ),
        ({"n_logical": 1, "depth": -1, "seed": 1}, "depth must be non-negative"),
        ({"n_logical": 1, "depth": 0, "seed": -1}, "seed must be non-negative"),
    ],
)
def test_random_benchmark_circuit_rejects_invalid_integer_inputs(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        random_benchmark_circuit(**kwargs)  # type: ignore[arg-type]


def test_benchmark_rejects_string_strategy_sequence() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)

    with pytest.raises(ValueError, match="strategies must be a sequence of strings"):
        benchmark_random_circuits(
            cfg,
            n_logical=1,
            depth=0,
            trials=0,
            strategies="baseline",
        )


def test_benchmark_rejects_non_sequence_strategies() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)

    with pytest.raises(ValueError, match="strategies must be a sequence of strings"):
        benchmark_random_circuits(
            cfg,
            n_logical=1,
            depth=0,
            trials=0,
            strategies=None,  # type: ignore[arg-type]
        )


def test_benchmark_rejects_non_string_strategy_entry() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)

    with pytest.raises(ValueError, match=r"strategies\[1\] must be a string"):
        benchmark_random_circuits(
            cfg,
            n_logical=1,
            depth=0,
            trials=0,
            strategies=("baseline", 1),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_qpus": 0}, "n_qpus must be positive"),
        ({"compute_per_qpu": -1}, "compute_per_qpu must be non-negative"),
        ({"inter_degree": -1}, "inter_degree must be non-negative"),
        ({"comm_ports": (True,)}, r"comm_ports\[0\] must be a non-negative integer"),
        ({"comm_ports": 1}, "comm_ports must be a sequence"),
        ({"intra_topologies": "clique"}, "intra_topologies must be a sequence"),
        (
            {"inter_topologies": ("switch", 3)},
            r"inter_topologies\[1\] must be a string",
        ),
    ],
)
def test_sweep_rejects_invalid_api_inputs(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    from quport.pipeline import sweep_topologies

    base: dict[str, object] = {
        "n_logical": 1,
        "depth": 0,
        "trials": 0,
        "seed": 1,
        "out_csv": str(tmp_path / "out.csv"),
        "intra_topologies": ("clique",),
        "inter_topologies": ("switch",),
        "comm_ports": (0,),
        "compute_per_qpu": 1,
        "n_qpus": 1,
        "inter_degree": 0,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        sweep_topologies(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, True])
def test_mapping_entrypoints_reject_invalid_optional_seeds(seed: object) -> None:
    from quport.pipeline import transpile_baseline

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)
    qc = random_benchmark_circuit(n_logical=1, depth=0, seed=0)

    with pytest.raises(ValueError, match="seed"):
        map_and_transpile(qc, cfg, seed=seed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        transpile_baseline(qc, cfg, seed=seed)  # type: ignore[arg-type]


def test_benchmark_supports_tpccap_sa_strategy() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )

    rows = benchmark_random_circuits(
        cfg,
        n_logical=3,
        depth=1,
        trials=1,
        seed=2,
        strategies=("tpccap_sa",),
    )

    assert len(rows) == 1
    assert rows[0]["strategy"] == "tpccap_sa"
    assert rows[0]["method"] == 3.0


def test_sweep_can_include_tpccap_sa_strategy(tmp_path: Path) -> None:
    from quport.pipeline import sweep_topologies

    out = tmp_path / "sweep_tpccap_sa.csv"

    sweep_topologies(
        n_logical=1,
        depth=0,
        trials=0,
        seed=5,
        out_csv=str(out),
        intra_topologies=("clique",),
        inter_topologies=("switch",),
        comm_ports=(0,),
        compute_per_qpu=1,
        n_qpus=1,
        strategies=("tpccap_sa",),
    )

    assert "3.0" in out.read_text(encoding="utf-8")


def test_benchmark_preserves_requested_strategy_order() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )

    rows = benchmark_random_circuits(
        cfg,
        n_logical=3,
        depth=0,
        trials=1,
        seed=4,
        strategies=("tpccap_sa", "baseline"),
    )

    assert [row["strategy"] for row in rows] == ["tpccap_sa", "baseline"]
    assert [row["method"] for row in rows] == [3.0, 0.0]


@pytest.mark.parametrize(
    ("strategies", "match"),
    [
        ((), "at least one strategy"),
        (("baseline", "baseline"), "duplicate strategies: baseline"),
        (("not-a-strategy",), "Unknown benchmark strategies: not-a-strategy"),
    ],
)
def test_benchmark_rejects_invalid_strategy_sequences(
    strategies: tuple[str, ...], match: str
) -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)

    with pytest.raises(ValueError, match=match):
        benchmark_random_circuits(
            cfg,
            n_logical=1,
            depth=0,
            trials=0,
            strategies=strategies,
        )


@pytest.mark.parametrize(
    ("strategies", "match"),
    [
        ((), "at least one strategy"),
        (("tpccap", "tpccap"), "duplicate strategies: tpccap"),
        (("bad",), "Unknown sweep strategies: bad"),
    ],
)
def test_sweep_rejects_invalid_strategy_sequences(
    tmp_path: Path, strategies: tuple[str, ...], match: str
) -> None:
    from quport.pipeline import sweep_topologies

    with pytest.raises(ValueError, match=match):
        sweep_topologies(
            n_logical=1,
            depth=0,
            trials=0,
            seed=5,
            out_csv=str(tmp_path / "sweep.csv"),
            intra_topologies=("clique",),
            inter_topologies=("switch",),
            comm_ports=(0,),
            compute_per_qpu=1,
            n_qpus=1,
            strategies=strategies,
        )


def test_benchmark_supports_cluster_strategy(tmp_path: Path) -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=2, comm_qubits_per_qpu=0)
    out = tmp_path / "cluster.csv"

    rows = benchmark_random_circuits(
        cfg,
        n_logical=1,
        depth=0,
        trials=1,
        out_csv=str(out),
        strategies=("cluster",),
    )

    assert len(rows) == 1
    assert rows[0]["strategy"] == "cluster"
    assert rows[0]["method"] == 4.0
    assert "cluster" in out.read_text(encoding="utf-8")


def test_benchmark_method_labels_are_stable_copy() -> None:
    from quport.pipeline import benchmark_method_labels

    labels = benchmark_method_labels()
    labels[4.0] = "mutated"

    assert benchmark_method_labels()[4.0] == "cluster"


def test_sweep_supports_cluster_strategy_with_zero_trials(tmp_path: Path) -> None:
    from quport.pipeline import sweep_topologies

    out = tmp_path / "sweep_cluster.csv"

    sweep_topologies(
        n_logical=1,
        depth=0,
        trials=0,
        seed=5,
        out_csv=str(out),
        intra_topologies=("clique",),
        inter_topologies=("switch",),
        comm_ports=(0,),
        compute_per_qpu=1,
        n_qpus=1,
        strategies=("cluster",),
    )

    assert out.read_text(encoding="utf-8") == (
        "intra,inter,ports,method,swaps_mean,remote_2q_mean,depth_mean,"
        "cost_mean,cost_median,transpile_time_mean\n"
        "clique,switch,0.0,4.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
    )


def test_benchmark_preserves_cluster_strategy_order() -> None:
    from quport.pipeline import benchmark_random_circuits

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=2, comm_qubits_per_qpu=0)

    rows = benchmark_random_circuits(
        cfg,
        n_logical=1,
        depth=0,
        trials=1,
        seed=4,
        strategies=("cluster", "baseline"),
    )

    assert [row["strategy"] for row in rows] == ["cluster", "baseline"]
    assert [row["method"] for row in rows] == [4.0, 0.0]


@pytest.mark.parametrize("strategy", ["balanced", "cluster", "tpccap", "tpccap_sa"])
@pytest.mark.parametrize("intra_topology", ["line", "ring", "grid2d"])
def test_mapped_circuit_respects_coupling_map_and_preserves_the_unitary(
    strategy: str, intra_topology: str
) -> None:
    """Global mapping must be legal on the device *and* semantics-preserving.

    Every two-qubit operation has to land on a real directed coupling-map edge,
    and undoing the transpiler's layout must recover the input circuit's unitary
    (padded with identity on the ancilla qubits).
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    from quport.architecture import MultiQPUArchitecture

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        intra_topology=intra_topology,
        inter_topology="switch",
        optimization_level=1,
    )
    arch = MultiQPUArchitecture(cfg)
    circuit = random_benchmark_circuit(n_logical=3, depth=3, seed=4)

    mapped = map_and_transpile(circuit, cfg, seed=4, strategy=strategy).mapped_circuit

    edges = set(arch.build_coupling_map().get_edges())
    position = {qubit: index for index, qubit in enumerate(mapped.qubits)}
    for instruction in mapped.data:
        if getattr(instruction.operation, "_directive", False):
            continue
        if len(instruction.qubits) == 2:
            pair = tuple(position[qubit] for qubit in instruction.qubits)
            assert pair in edges, f"{instruction.operation.name} on non-edge {pair}"

    padded = QuantumCircuit(mapped.num_qubits)
    padded.compose(circuit, qubits=list(range(circuit.num_qubits)), inplace=True)
    assert Operator.from_circuit(mapped).equiv(Operator(padded))


def test_transpile_baseline_reports_partition_cut_as_not_applicable() -> None:
    """The baseline does no partitioning, so its cut is the -1 sentinel.

    Reporting 0.0 instead would read as a genuinely perfect partition in the
    benchmark CSV.
    """
    from quport.pipeline import transpile_baseline

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
        optimization_level=0,
    )
    result = transpile_baseline(random_benchmark_circuit(4, 3, 1), cfg, seed=1)

    assert result.partition_cut == -1.0
    assert result.strategy == "baseline"
    assert result.partition_diagnostics is None


def test_sweep_means_are_averaged_over_that_strategy_only(tmp_path: Path) -> None:
    """Each summary row averages its own strategy's trials, not every row.

    Dividing by the whole snapshot would scale every mean down by the number of
    strategies in the sweep.
    """
    import csv
    import statistics

    from quport.config import LatencyModel
    from quport.pipeline import (
        _BENCHMARK_METHOD_IDS,
        benchmark_random_circuits,
        sweep_topologies,
    )

    out = tmp_path / "sweep.csv"
    strategies = ("balanced", "cluster")
    kwargs = dict(n_logical=4, depth=2, trials=3, seed=5)

    sweep_topologies(
        out_csv=str(out),
        intra_topologies=("clique",),
        inter_topologies=("switch",),
        comm_ports=(1,),
        compute_per_qpu=3,
        n_qpus=4,
        strategies=strategies,
        **kwargs,
    )

    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
        inter_degree=2,
    )
    trials = benchmark_random_circuits(
        cfg=cfg, latency=LatencyModel(), out_csv=None, strategies=strategies, **kwargs
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == len(strategies)
    for strategy in strategies:
        own = [r for r in trials if r["strategy"] == strategy]
        assert len(own) == kwargs["trials"]
        row = next(
            r for r in rows if float(r["method"]) == _BENCHMARK_METHOD_IDS[strategy]
        )
        for csv_key, trial_key in (
            ("swaps_mean", "swaps"),
            ("remote_2q_mean", "remote_2q"),
            ("depth_mean", "depth"),
            ("cost_mean", "cost_total"),
        ):
            expected = statistics.mean(float(r[trial_key]) for r in own)
            assert float(row[csv_key]) == pytest.approx(expected)

        expected_median = statistics.median(float(r["cost_total"]) for r in own)
        assert float(row["cost_median"]) == pytest.approx(expected_median)


def _tpccap_partition_for(strategy: str, temporal_decay: float | None) -> list[int]:
    from quport.pipeline import map_and_transpile, random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="ring",
        optimization_level=0,
    )
    circuit = random_benchmark_circuit(n_logical=10, depth=8, seed=4)
    return map_and_transpile(
        circuit, cfg, seed=4, strategy=strategy, temporal_decay=temporal_decay
    ).partition


def test_temporal_decay_default_preserves_the_historical_weighting() -> None:
    """`None` must reproduce the shipped per-strategy behaviour exactly.

    Historically `tpccap` used uniform interaction counts while `tpccap_sa` used
    a decay of 0.98. That asymmetry is preserved by default so existing
    benchmark numbers do not move.
    """
    assert _tpccap_partition_for("tpccap", None) == _tpccap_partition_for("tpccap", 1.0)
    assert _tpccap_partition_for("tpccap_sa", None) == _tpccap_partition_for(
        "tpccap_sa", 0.98
    )


def test_temporal_decay_applies_to_both_topology_aware_strategies() -> None:
    """An explicit value weights `tpccap` and `tpccap_sa` the same way.

    That is what makes a comparison between them an ablation of the annealing
    rather than of the annealing plus a different objective input.
    """
    # tpccap_sa defaults to 0.98, so asking for uniform weights must change it.
    assert _tpccap_partition_for("tpccap_sa", 1.0) != _tpccap_partition_for(
        "tpccap_sa", None
    )
    # tpccap defaults to uniform, so asking for decay must change it.
    assert _tpccap_partition_for("tpccap", 0.5) != _tpccap_partition_for("tpccap", None)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan"), float("inf"), True])
def test_temporal_decay_rejects_values_outside_the_valid_range(bad: object) -> None:
    from quport.pipeline import map_and_transpile, random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        optimization_level=0,
    )
    circuit = random_benchmark_circuit(n_logical=3, depth=2, seed=0)

    with pytest.raises(ValueError, match="temporal_decay"):
        map_and_transpile(
            circuit, cfg, seed=0, strategy="tpccap", temporal_decay=bad  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("strategy", ["balanced", "cluster"])
def test_temporal_decay_does_not_affect_non_topology_strategies(strategy: str) -> None:
    assert _tpccap_partition_for(strategy, None) == _tpccap_partition_for(strategy, 0.5)


@pytest.mark.parametrize("strategy", ["balanced", "cluster"])
def test_temporal_decay_is_ignored_not_rejected_for_strategies_that_skip_it(
    strategy: str,
) -> None:
    """Validation timing must match `compile_distributed`.

    That function deliberately ignores an out-of-range decay for strategies
    which never consult it; the same argument on the same kind of call should
    not behave differently here.
    """
    from quport.compiler import compile_distributed
    from quport.pipeline import map_and_transpile, random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        optimization_level=0,
    )
    circuit = random_benchmark_circuit(n_logical=3, depth=2, seed=0)

    mapped = map_and_transpile(
        circuit, cfg, seed=0, strategy=strategy, temporal_decay=1.5
    )
    assert mapped.strategy == strategy

    distributed = compile_distributed(
        circuit, cfg, seed=0, strategy=strategy, temporal_decay=1.5
    )
    assert distributed.strategy == strategy


def test_temporal_decay_of_one_matches_uniform_interaction_counts() -> None:
    """`decay = 1.0` gives `1.0 ** t == 1.0`, i.e. exactly the plain counts."""
    from quport.interaction import extract_temporal_twoq_weights, extract_twoq_weights
    from quport.pipeline import _translate_to_basis, random_benchmark_circuit

    cfg = MultiQPUConfig()
    circuit = _translate_to_basis(
        random_benchmark_circuit(n_logical=8, depth=6, seed=2), cfg.basis_gates, 2
    )

    uniform = {
        key: float(value) for key, value in extract_twoq_weights(circuit).items()
    }
    decayed = {
        key: float(value)
        for key, value in extract_temporal_twoq_weights(circuit, decay=1.0).items()
    }

    assert uniform == decayed


@pytest.mark.parametrize("trials", [4, 5])
def test_sweep_reports_a_real_cost_median(tmp_path: Path, trials: int) -> None:
    """The median column earns its place only if it can disagree with the mean.

    Estimated cost is heavily skewed across random circuits, which is why both are
    reported: a few hard instances can pull the mean past the median far enough to
    reverse which strategy looks better.

    Both parities are covered because they take different branches -- an even count
    averages the two middle values -- and the comparison is against
    `statistics.median`, which also sorts, so an unsorted implementation fails here
    rather than only on inputs that happen to arrive ordered.
    """
    import csv
    import statistics

    from quport.config import LatencyModel
    from quport.pipeline import (
        _BENCHMARK_METHOD_IDS,
        benchmark_random_circuits,
        sweep_topologies,
    )

    out = tmp_path / "sweep.csv"
    strategies = ("baseline", "tpccap")
    kwargs = dict(n_logical=6, depth=6, trials=trials, seed=0)

    sweep_topologies(
        out_csv=str(out),
        intra_topologies=("clique",),
        inter_topologies=("switch",),
        comm_ports=(1,),
        compute_per_qpu=3,
        n_qpus=3,
        strategies=strategies,
        **kwargs,
    )

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    rows = benchmark_random_circuits(
        cfg, latency=LatencyModel(), strategies=strategies, **kwargs
    )

    with out.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == len(strategies)

    disagreed = False
    for strategy in strategies:
        row = next(
            r
            for r in summary
            if float(r["method"]) == _BENCHMARK_METHOD_IDS[strategy]
        )
        own = [float(r["cost_total"]) for r in rows if r["strategy"] == strategy]
        assert len(own) == trials
        assert float(row["cost_median"]) == pytest.approx(statistics.median(own))
        assert float(row["cost_mean"]) == pytest.approx(statistics.mean(own))
        if float(row["cost_median"]) != pytest.approx(float(row["cost_mean"])):
            disagreed = True

    assert disagreed, "expected the median and mean to differ on at least one row"
