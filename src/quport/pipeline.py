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
from typing import TypeAlias

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

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
    return random_circuit(
        num_qubits=n_logical,
        depth=depth,
        max_operands=2,
        measure=False,
        seed=seed,
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
        - "tpccap": QuPort novel partitioner (topology+port+congestion aware)

    Notes
    -----
    The CSV is deliberately numeric-friendly. The column `method` encodes:
        baseline=0, balanced=1, tpccap=2
    and the column `strategy` stores the string name for readability.
    """
    latency = latency or LatencyModel()
    rows: list[BenchmarkRow] = []

    if trials < 0:
        raise ValueError("trials must be non-negative")

    method_id = {"baseline": 0.0, "balanced": 1.0, "tpccap": 2.0}
    selected_strategies = tuple(strategies)
    unknown = sorted(set(selected_strategies) - set(method_id))
    if unknown:
        raise ValueError(
            "Unknown benchmark strategies: "
            + ", ".join(unknown)
            + ". Use any of: baseline, balanced, tpccap."
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

    for t in range(trials):
        s = seed + t
        qc = random_benchmark_circuit(n_logical, depth, s)

        results: dict[str, MapResult] = {}
        if "baseline" in selected_strategies:
            results["baseline"] = transpile_baseline(qc, cfg, latency=latency, seed=s)
        if "balanced" in selected_strategies:
            results["balanced"] = map_and_transpile(
                qc, cfg, latency=latency, seed=s, strategy="balanced"
            )
        if "tpccap" in selected_strategies:
            results["tpccap"] = map_and_transpile(
                qc, cfg, latency=latency, seed=s, strategy="tpccap"
            )

        for strat, r in results.items():
            rows.append(
                {
                    "trial": float(t),
                    "seed": float(s),
                    "method": method_id[strat],
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
) -> None:
    """Sweep multiple topology settings; write summary CSV."""
    latency = LatencyModel()
    summary: list[SweepSummaryRow] = []

    for intra in intra_topologies:
        for inter in inter_topologies:
            for ports in comm_ports:
                cfg = MultiQPUConfig(
                    n_qpus=n_qpus,
                    compute_qubits_per_qpu=compute_per_qpu,
                    comm_qubits_per_qpu=ports,
                    intra_topology=intra,
                    inter_topology=inter,
                    inter_degree=inter_degree,
                )
                if n_logical > cfg.total_physical_qubits():
                    continue

                rows = benchmark_random_circuits(
                    cfg=cfg,
                    n_logical=n_logical,
                    depth=depth,
                    trials=trials,
                    seed=seed,
                    latency=latency,
                    out_csv=None,
                )

                rows_snapshot = list(rows)
                intra_id = intra
                inter_id = inter
                ports_count = ports

                # aggregate for baseline (0.0), balanced (1.0), and tpccap (2.0)
                def agg(
                    method: float,
                    *,
                    rows_snapshot: list[BenchmarkRow] = rows_snapshot,
                    intra_id: str = intra_id,
                    inter_id: str = inter_id,
                    ports_count: int = ports_count,
                ) -> SweepSummaryRow:
                    rs = [r for r in rows_snapshot if r["method"] == method]

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

                summary.append(agg(0.0))
                summary.append(agg(1.0))
                summary.append(agg(2.0))

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
