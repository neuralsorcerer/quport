# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from quport._validation import validate_nonnegative_integral
from quport.architecture import MultiQPUArchitecture
from quport.config import InterTopology, IntraTopology, LatencyModel, MultiQPUConfig
from quport.cost import CostBreakdown, estimate_cost
from quport.interaction import extract_temporal_twoq_weights, extract_twoq_weights
from quport.layout import compute_layout_hints, naive_layout
from quport.metrics import CircuitMetrics, compute_metrics
from quport.partition import (
    PartitionDiagnostics,
    PartitionResult,
    balanced_greedy_partition,
    heavy_edge_clustering_partition,
    tpccap_partition,
    tpccap_sa_partition,
)

BenchmarkRow: TypeAlias = dict[str, float | str]
SweepSummaryRow: TypeAlias = dict[str, float | str]
_StringT = TypeVar("_StringT", bound=str)
_BENCHMARK_METHOD_IDS = {
    "baseline": 0.0,
    "balanced": 1.0,
    "tpccap": 2.0,
    "tpccap_sa": 3.0,
    "cluster": 4.0,
}
_BENCHMARK_METHOD_LABELS = {
    method_id: strategy for strategy, method_id in _BENCHMARK_METHOD_IDS.items()
}


def benchmark_method_labels() -> dict[float, str]:
    """Return a copy of the stable numeric benchmark-method label map."""
    return dict(_BENCHMARK_METHOD_LABELS)


def _validate_positive_int(value: object, *, label: str) -> int:
    """Return a positive integer, rejecting bools and non-integral values."""
    out = validate_nonnegative_integral(value, label=label)
    if out <= 0:
        raise ValueError(f"{label} must be positive")
    return out


def _validate_string_sequence(
    values: Sequence[_StringT],
    *,
    label: str,
) -> tuple[_StringT, ...]:
    """Validate an API sequence of strings without treating a bare string as a sequence."""
    if isinstance(values, str | bytes | bytearray):
        raise ValueError(f"{label} must be a sequence of strings, not a string")
    if not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of strings")
    out = tuple(values)
    for idx, value in enumerate(out):
        if not isinstance(value, str):
            raise ValueError(f"{label}[{idx}] must be a string")
    return out


def _validate_strategy_sequence(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """Validate benchmark/sweep strategy names, preserving caller order."""
    selected = _validate_string_sequence(values, label=label)
    if not selected:
        raise ValueError(f"{label} must contain at least one strategy")

    seen: set[str] = set()
    duplicates: list[str] = []
    for strategy in selected:
        if strategy in seen and strategy not in duplicates:
            duplicates.append(strategy)
        seen.add(strategy)
    if duplicates:
        duplicate_list = ", ".join(duplicates)
        raise ValueError(f"{label} contains duplicate strategies: {duplicate_list}")

    unknown = sorted(set(selected) - set(_BENCHMARK_METHOD_IDS))
    if unknown:
        allowed = ", ".join(_BENCHMARK_METHOD_IDS)
        raise ValueError(
            f"Unknown {label}: {', '.join(unknown)}. Use any of: {allowed}."
        )
    return selected


def _validate_nonnegative_int_sequence(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    """Validate a non-string sequence of non-negative integers."""
    if isinstance(values, str | bytes | bytearray) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of non-negative integers")
    return tuple(
        validate_nonnegative_integral(value, label=f"{label}[{idx}]")
        for idx, value in enumerate(values)
    )


@dataclass(frozen=True)
class MapResult:
    """Outputs of a mapping+transpilation run."""

    mapped_circuit: QuantumCircuit
    cfg: MultiQPUConfig
    partition: list[int]
    partition_cut: float
    strategy: str
    partition_diagnostics: PartitionDiagnostics | None
    mapping_time_s: float
    transpile_time_s: float
    metrics: CircuitMetrics
    cost: CostBreakdown


def _translate_to_basis(
    qc: QuantumCircuit, basis_gates: Sequence[str], seed: int | None
) -> QuantumCircuit:
    """Translate the circuit into a safe basis before routing.

    This avoids direction-fixing errors for gates that cannot be auto-flipped by the transpiler.
    """
    return transpile(
        qc, basis_gates=list(basis_gates), optimization_level=0, seed_transpiler=seed
    )


def map_and_transpile(
    qc: QuantumCircuit,
    cfg: MultiQPUConfig,
    latency: LatencyModel | None = None,
    seed: int | None = None,
    strategy: str = "balanced",
) -> MapResult:
    """End-to-end pipeline: partition -> initial layout -> transpile -> metrics -> cost.

    Parameters
    ----------
    qc:
        Input circuit (logical qubits).
    cfg:
        Multi-QPU architecture + transpiler configuration.
    latency:
        Cost model for latency estimation.
    seed:
        Random seed for reproducibility (partition refinement, SABRE).
    strategy:
        Partitioning strategy:
        - "balanced": balanced greedy partition + local refinement (recommended baseline)
        - "cluster" : heavy-edge clustering baseline
        - "tpccap"    : topology+port+congestion aware partitioner (novel)
        - "tpccap_sa" : TPCCAP + simulated annealing refinement (best)

    Returns
    -------
    MapResult
    """
    latency = latency or LatencyModel()
    if seed is not None:
        seed = validate_nonnegative_integral(seed, label="seed")

    if qc.num_qubits > cfg.total_physical_qubits():
        raise ValueError(
            f"Logical qubits={qc.num_qubits} exceed physical qubits={cfg.total_physical_qubits()} in config."
        )

    arch = MultiQPUArchitecture(cfg)
    coupling = arch.build_coupling_map()

    # Step 0: translate into basis (removes CY, etc.)
    qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)

    # Step 1: extract interaction weights for partitioning
    t0 = time.perf_counter()
    weights = extract_twoq_weights(qc_basis)
    capacity = cfg.capacity_per_qpu()

    partition_diag: PartitionDiagnostics | None = None
    comm_mode = "topk"

    if strategy == "balanced":
        pres: PartitionResult = balanced_greedy_partition(
            n=qc_basis.num_qubits,
            weights=weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        comm_mode = "topk"

    elif strategy == "tpccap":
        sp = arch.qpu_shortest_paths()
        pres, diag = tpccap_partition(
            n=qc_basis.num_qubits,
            weights=weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        partition_diag = diag
        comm_mode = "diverse"

    elif strategy == "tpccap_sa":
        sp = arch.qpu_shortest_paths()
        # time-decayed weights often improve early remote behavior on random circuits
        weights_td = extract_temporal_twoq_weights(qc_basis, decay=0.98)
        pres, diag, _anneal = tpccap_sa_partition(
            n=qc_basis.num_qubits,
            weights=weights_td,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        partition_diag = diag
        comm_mode = "diverse"

    elif strategy == "cluster":
        part = heavy_edge_clustering_partition(
            n=qc_basis.num_qubits,
            weights=weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
        )
        cut = 0
        for (i, j), w in weights.items():
            if part[i] != part[j]:
                cut += w
        comm_mode = "topk"

    else:
        raise ValueError(
            "Unknown strategy. Use 'balanced', 'cluster', 'tpccap', or 'tpccap_sa'."
        )

    hints = compute_layout_hints(qc_basis, arch, part, comm_mode=comm_mode)
    mapping_time = time.perf_counter() - t0

    # Step 2: transpile on the global coupling map using SABRE
    t1 = time.perf_counter()
    mapped = transpile(
        qc_basis,
        coupling_map=coupling,
        initial_layout=hints.initial_layout,
        basis_gates=list(cfg.basis_gates),
        optimization_level=cfg.optimization_level,
        layout_method=cfg.layout_method,
        routing_method=cfg.routing_method,
        seed_transpiler=seed,
    )
    transpile_time = time.perf_counter() - t1

    metrics = compute_metrics(mapped, arch)
    cost = estimate_cost(metrics, latency)

    return MapResult(
        mapped_circuit=mapped,
        cfg=cfg,
        partition=part,
        partition_cut=cut,
        strategy=strategy,
        partition_diagnostics=partition_diag,
        mapping_time_s=mapping_time,
        transpile_time_s=transpile_time,
        metrics=metrics,
        cost=cost,
    )


def transpile_baseline(
    qc: QuantumCircuit,
    cfg: MultiQPUConfig,
    latency: LatencyModel | None = None,
    seed: int | None = None,
) -> MapResult:
    """Baseline: translate to basis then let transpiler pick layout/routing (no partition hints)."""
    latency = latency or LatencyModel()
    if seed is not None:
        seed = validate_nonnegative_integral(seed, label="seed")

    if qc.num_qubits > cfg.total_physical_qubits():
        raise ValueError(
            f"Logical qubits={qc.num_qubits} exceed physical qubits={cfg.total_physical_qubits()} in config."
        )

    arch = MultiQPUArchitecture(cfg)
    coupling = arch.build_coupling_map()
    qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)

    # Optional: initial layout = identity (logical i -> physical i)
    # This baseline is deterministic and exposes inter-QPU penalties.
    init_layout = naive_layout(qc_basis.num_qubits)

    t0 = time.perf_counter()
    mapped = transpile(
        qc_basis,
        coupling_map=coupling,
        initial_layout=init_layout,
        basis_gates=list(cfg.basis_gates),
        optimization_level=cfg.optimization_level,
        layout_method=cfg.layout_method,
        routing_method=cfg.routing_method,
        seed_transpiler=seed,
    )
    t1 = time.perf_counter()

    metrics = compute_metrics(mapped, arch)
    cost = estimate_cost(metrics, latency)

    return MapResult(
        mapped_circuit=mapped,
        cfg=cfg,
        partition=[arch.qpu_of_phys(physical) for physical in init_layout],
        partition_cut=-1.0,
        strategy="baseline",
        partition_diagnostics=None,
        mapping_time_s=0.0,
        transpile_time_s=t1 - t0,
        metrics=metrics,
        cost=cost,
    )


def random_benchmark_circuit(
    n_logical: int,
    depth: int,
    seed: int,
) -> QuantumCircuit:
    """Random circuit generator suitable for mapping benchmarks."""
    n_logical_value = validate_nonnegative_integral(n_logical, label="n_logical")
    depth_value = validate_nonnegative_integral(depth, label="depth")
    seed_value = validate_nonnegative_integral(seed, label="seed")
    return random_circuit(
        num_qubits=n_logical_value,
        depth=depth_value,
        max_operands=2,
        measure=False,
        seed=seed_value,
    )


def benchmark_random_circuits(
    cfg: MultiQPUConfig,
    n_logical: int,
    depth: int,
    trials: int,
    seed: int = 0,
    latency: LatencyModel | None = None,
    out_csv: str | None = None,
    strategies: Sequence[str] = ("baseline", "balanced", "tpccap"),
) -> list[BenchmarkRow]:
    """Run a benchmark over multiple random circuits and write results to CSV.

    Parameters
    ----------
    strategies:
        Any subset of:
        - "baseline": identity-ish initial layout + SABRE routing
        - "balanced": QuPort baseline partitioner
        - "cluster": heavy-edge clustering partitioner
        - "tpccap": QuPort novel partitioner (topology+port+congestion aware)
        - "tpccap_sa": TPCCAP plus simulated-annealing refinement

    Notes
    -----
    The CSV is deliberately numeric-friendly. The column `method` encodes:
        baseline=0, balanced=1, tpccap=2, tpccap_sa=3, cluster=4
    and the column `strategy` stores the string name for readability.
    """
    latency = latency or LatencyModel()
    rows: list[BenchmarkRow] = []

    n_logical_value = validate_nonnegative_integral(n_logical, label="n_logical")
    depth_value = validate_nonnegative_integral(depth, label="depth")
    trials_value = validate_nonnegative_integral(trials, label="trials")
    seed_value = validate_nonnegative_integral(seed, label="seed")

    selected_strategies = _validate_strategy_sequence(
        strategies, label="benchmark strategies"
    )

    fieldnames = [
        "trial",
        "seed",
        "method",
        "strategy",
        "swaps",
        "remote_2q",
        "depth",
        "size",
        "cost_total",
        "cost_local",
        "cost_remote",
        "mapping_time_s",
        "transpile_time_s",
    ]

    for t in range(trials_value):
        s = seed_value + t
        qc = random_benchmark_circuit(n_logical_value, depth_value, s)

        results: list[tuple[str, MapResult]] = []
        for strategy_name in selected_strategies:
            if strategy_name == "baseline":
                result = transpile_baseline(qc, cfg, latency=latency, seed=s)
            else:
                result = map_and_transpile(
                    qc, cfg, latency=latency, seed=s, strategy=strategy_name
                )
            results.append((strategy_name, result))

        for strat, r in results:
            rows.append(
                {
                    "trial": float(t),
                    "seed": float(s),
                    "method": _BENCHMARK_METHOD_IDS[strat],
                    "strategy": strat,
                    "swaps": float(r.metrics.swaps),
                    "remote_2q": float(r.metrics.remote_2q),
                    "depth": float(r.metrics.depth),
                    "size": float(r.metrics.size),
                    "cost_total": float(r.cost.total),
                    "cost_local": float(r.cost.local),
                    "cost_remote": float(r.cost.remote),
                    "mapping_time_s": float(r.mapping_time_s),
                    "transpile_time_s": float(r.transpile_time_s),
                }
            )

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def sweep_topologies(
    n_logical: int,
    depth: int,
    trials: int,
    seed: int,
    out_csv: str,
    intra_topologies: Sequence[IntraTopology] = ("clique", "line", "ring"),
    inter_topologies: Sequence[InterTopology] = ("switch", "ring", "degree_d", "clos"),
    comm_ports: Sequence[int] = (1, 2),
    compute_per_qpu: int = 8,
    n_qpus: int = 10,
    inter_degree: int = 2,
    strategies: Sequence[str] = ("baseline", "balanced", "tpccap"),
) -> None:
    """Sweep multiple topology settings; write summary CSV."""
    n_logical_value = validate_nonnegative_integral(n_logical, label="n_logical")
    depth_value = validate_nonnegative_integral(depth, label="depth")
    trials_value = validate_nonnegative_integral(trials, label="trials")
    seed_value = validate_nonnegative_integral(seed, label="seed")
    compute_per_qpu_value = validate_nonnegative_integral(
        compute_per_qpu, label="compute_per_qpu"
    )
    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    inter_degree_value = validate_nonnegative_integral(
        inter_degree, label="inter_degree"
    )
    intra_values = _validate_string_sequence(intra_topologies, label="intra_topologies")
    inter_values = _validate_string_sequence(inter_topologies, label="inter_topologies")
    comm_port_values = _validate_nonnegative_int_sequence(
        comm_ports, label="comm_ports"
    )
    selected_strategies = _validate_strategy_sequence(
        strategies, label="sweep strategies"
    )

    latency = LatencyModel()
    summary: list[SweepSummaryRow] = []

    for intra in intra_values:
        for inter in inter_values:
            for ports in comm_port_values:
                cfg = MultiQPUConfig(
                    n_qpus=n_qpus_value,
                    compute_qubits_per_qpu=compute_per_qpu_value,
                    comm_qubits_per_qpu=ports,
                    intra_topology=intra,
                    inter_topology=inter,
                    inter_degree=inter_degree_value,
                )
                if n_logical_value > cfg.total_physical_qubits():
                    continue

                rows = benchmark_random_circuits(
                    cfg=cfg,
                    n_logical=n_logical_value,
                    depth=depth_value,
                    trials=trials_value,
                    seed=seed_value,
                    latency=latency,
                    out_csv=None,
                    strategies=selected_strategies,
                )

                rows_snapshot = list(rows)
                intra_id = intra
                inter_id = inter
                ports_count = ports

                def agg(
                    strategy_name: str,
                    *,
                    rows_snapshot: list[BenchmarkRow] = rows_snapshot,
                    intra_id: str = intra_id,
                    inter_id: str = inter_id,
                    ports_count: int = ports_count,
                ) -> SweepSummaryRow:
                    method = _BENCHMARK_METHOD_IDS[strategy_name]
                    rs = [r for r in rows_snapshot if r["strategy"] == strategy_name]

                    def mean(k: str) -> float:
                        return sum(float(r[k]) for r in rs) / max(1, len(rs))

                    return {
                        "intra": intra_id,
                        "inter": inter_id,
                        "ports": float(ports_count),
                        "method": method,
                        "swaps_mean": mean("swaps"),
                        "remote_2q_mean": mean("remote_2q"),
                        "depth_mean": mean("depth"),
                        "cost_mean": mean("cost_total"),
                        "transpile_time_mean": mean("transpile_time_s"),
                    }

                for strategy_name in selected_strategies:
                    summary.append(agg(strategy_name))

    sweep_fieldnames = [
        "intra",
        "inter",
        "ports",
        "method",
        "swaps_mean",
        "remote_2q_mean",
        "depth_mean",
        "cost_mean",
        "transpile_time_mean",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_fieldnames)
        writer.writeheader()
        writer.writerows(summary)
