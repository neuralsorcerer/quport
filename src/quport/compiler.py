# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed compilation pipeline.

This module implements the **paper-oriented** compilation flow for a controller that
maps a circuit onto *multiple lower-layer QPUs* under NISQ qubit scarcity.

Key idea
--------
Instead of allowing the single-QPU transpiler to insert SWAPs across inter-QPU links
(which would imply moving unknown quantum state between QPUs), we:

1) **Partition** logical qubits across QPUs.
2) **Assign** logical qubits to physical qubits (compute + comm ports).
3) **Split** the circuit into per-QPU local circuits + an ordered list of *remote ops*.
4) **Route locally** within each QPU using Qiskit SABRE (minimizing SWAPs).
5) **Schedule remote ops** with comm-port and link-capacity constraints.

This matches how many distributed quantum computing (DQC) proposals treat inter-QPU
operations: as remote gates implemented with entanglement generation + local operations,
not literal SWAPs between devices.

The output is suitable for:
- benchmarking random circuits
- producing plots for a research paper (remote ops vs ports vs topology)
- generating per-QPU QASM programs and a remote-op schedule trace
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile

from quport._validation import validate_nonnegative_integral
from quport.architecture import MultiQPUArchitecture
from quport.config import LatencyModel, MultiQPUConfig
from quport.distributed import DistributedProgram, split_into_qpus
from quport.interaction import (
    extract_temporal_twoq_weights,
    extract_twoq_weights,
    validate_temporal_decay,
)
from quport.layout import compute_layout_hints
from quport.metrics import CircuitMetrics, compute_metrics, count_ops
from quport.partition import (
    AnnealDiagnostics,
    PartitionDiagnostics,
    balanced_greedy_partition,
    heavy_edge_clustering_partition,
    tpccap_partition,
    tpccap_sa_partition,
)
from quport.schedule import (
    TopologySchedulePlan,
    TopologyScheduleSummary,
    estimate_topology_schedule_plan,
)


def _translate_to_basis(
    qc: QuantumCircuit, basis_gates: Sequence[str], seed: int | None
) -> QuantumCircuit:
    """Translate the circuit into a safe basis before any mapping.

    Why: Qiskit can only automatically flip directions for a subset of 2Q gates.
    Translating early ensures we only see basis operations (e.g., CX, RZ, SX, X)
    and avoids errors like "cy cannot be flipped".
    """
    return transpile(
        qc, basis_gates=list(basis_gates), optimization_level=0, seed_transpiler=seed
    )


@dataclass(frozen=True)
class DistributedCompileResult:
    physical_circuit: QuantumCircuit
    cfg: MultiQPUConfig
    strategy: str
    partition: list[int]
    partition_cut: float
    partition_diagnostics: PartitionDiagnostics | None
    anneal_diagnostics: AnnealDiagnostics | None

    # Decomposed program
    program: DistributedProgram

    # Routed local programs (one per QPU)
    local_routed: dict[int, QuantumCircuit]

    # Metrics
    global_metrics: CircuitMetrics
    local_metrics: dict[int, dict[str, int]]  # op counts per QPU (including swaps)
    schedule: TopologyScheduleSummary
    schedule_plan: TopologySchedulePlan

    # Times
    mapping_time_s: float
    local_transpile_time_s: float


def compile_distributed(
    qc: QuantumCircuit,
    cfg: MultiQPUConfig,
    latency: LatencyModel | None = None,
    seed: int | None = None,
    strategy: str = "tpccap_sa",
    temporal_decay: float = 0.98,
) -> DistributedCompileResult:
    """Compile a circuit for multi-QPU execution.

    Strategies
    ----------
    - "balanced"   : balanced greedy partition
    - "cluster"    : heavy-edge clustering baseline
    - "tpccap"     : TPCCAP (topology+port+congestion aware)
    - "tpccap_sa"  : TPCCAP + simulated annealing refinement (recommended)

    temporal_decay
    --------------
    If < 1, uses time-decayed weights to bias the partitioner toward reducing
    *early* remote interactions.
    """
    latency = latency or LatencyModel()
    if seed is not None:
        seed = validate_nonnegative_integral(seed, label="seed")

    if qc.num_qubits > cfg.total_physical_qubits():
        raise ValueError(
            f"Logical qubits={qc.num_qubits} exceed physical qubits={cfg.total_physical_qubits()} in config."
        )

    supported_strategies = {"balanced", "cluster", "tpccap", "tpccap_sa"}
    if strategy not in supported_strategies:
        raise ValueError("Unknown strategy.")

    is_tpccap = strategy in ("tpccap", "tpccap_sa")
    temporal_decay_value = (
        validate_temporal_decay(temporal_decay, label="temporal_decay")
        if is_tpccap
        else 1.0
    )

    arch = MultiQPUArchitecture(cfg)

    # 0) Translate into safe basis (prevents direction-fixing errors)
    qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)

    # 1) Extract interaction graph
    t0 = time.perf_counter()

    partition_weights: Mapping[tuple[int, int], float]
    if is_tpccap and temporal_decay_value < 1.0:
        partition_weights = extract_temporal_twoq_weights(
            qc_basis, decay=temporal_decay_value
        )  # float weights
    else:
        partition_weights = extract_twoq_weights(qc_basis)  # int weights

    capacity = cfg.capacity_per_qpu()
    part: list[int]
    cut: float
    part_diag: PartitionDiagnostics | None = None
    anneal_diag: AnnealDiagnostics | None = None

    if strategy == "balanced":
        pres = balanced_greedy_partition(
            n=qc_basis.num_qubits,
            weights=partition_weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut

    elif strategy == "cluster":
        static_weights = partition_weights
        part = heavy_edge_clustering_partition(
            n=qc_basis.num_qubits,
            weights=static_weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
        )
        cut = 0
        for (i, j), w in static_weights.items():
            if part[i] != part[j]:
                cut += w

    elif strategy == "tpccap":
        sp = arch.qpu_shortest_paths()
        pres, diag = tpccap_partition(
            n=qc_basis.num_qubits,
            weights=partition_weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        part_diag = diag

    elif strategy == "tpccap_sa":
        sp = arch.qpu_shortest_paths()
        pres, diag, ad = tpccap_sa_partition(
            n=qc_basis.num_qubits,
            weights=partition_weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        part_diag = diag
        anneal_diag = ad

    else:
        raise ValueError("Unknown strategy.")

    # 2) Compute layout hints (comm qubit selection is diversity-aware for tpccap*)
    comm_mode = "diverse" if is_tpccap else "topk"
    hints = compute_layout_hints(qc_basis, arch, part, comm_mode=comm_mode)
    mapping_time = time.perf_counter() - t0

    # 3) Apply the layout WITHOUT global routing (keeps inter-QPU ops as remote events).
    physical = transpile(
        qc_basis,
        initial_layout=hints.initial_layout,
        basis_gates=list(cfg.basis_gates),
        optimization_level=0,
        seed_transpiler=seed,
    )

    # 4) Split into per-QPU local circuits + remote ops list
    program = split_into_qpus(physical, arch)

    # 5) Route locally within each QPU block (minimize SWAPs, no cross-QPU swaps)
    t1 = time.perf_counter()
    local_routed: dict[int, QuantumCircuit] = {}
    local_counts: dict[int, dict[str, int]] = {}

    identity_layout = list(range(arch.n_phys))
    for qpu_id, local_qc in program.local_circuits.items():
        cm_intra = arch.build_intra_coupling_map(qpu_id)
        routed = transpile(
            local_qc,
            coupling_map=cm_intra,
            initial_layout=identity_layout,
            basis_gates=list(cfg.basis_gates),
            optimization_level=cfg.optimization_level,
            layout_method="trivial",
            routing_method=cfg.routing_method,
            seed_transpiler=seed,
        )
        local_routed[qpu_id] = routed
        local_counts[qpu_id] = dict(count_ops(routed))

    local_time = time.perf_counter() - t1

    # 6) Global metrics + topology-aware schedule estimate (remote rounds)
    global_metrics = compute_metrics(physical, arch)
    sched_plan = estimate_topology_schedule_plan(physical, arch, latency)

    return DistributedCompileResult(
        physical_circuit=physical,
        cfg=cfg,
        strategy=strategy,
        partition=part,
        partition_cut=cut,
        partition_diagnostics=part_diag,
        anneal_diagnostics=anneal_diag,
        program=program,
        local_routed=local_routed,
        global_metrics=global_metrics,
        local_metrics=local_counts,
        schedule=sched_plan.summary,
        schedule_plan=sched_plan,
        mapping_time_s=mapping_time,
        local_transpile_time_s=local_time,
    )
