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
        "cost_mean,transpile_time_mean\n"
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
