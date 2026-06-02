# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.config import LatencyModel, MultiQPUConfig
from quport.distributed import RemoteOp, split_into_qpus
from quport.network import UNREACHABLE_DISTANCE

UNSCHEDULABLE_PENALTY: float = float(UNREACHABLE_DISTANCE)


@dataclass(frozen=True)
class ScheduleSummary:
    """A coarse schedule summary (research metric)."""

    makespan: float
    steps: int
    remote_ops: int


def _validated_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _validated_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _validated_nonnegative_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        out = float(cast(SupportsFloat | SupportsIndex | str, value))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    if out < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return out


@dataclass(frozen=True)
class _ValidatedLatencyValues:
    oneq: float
    twoq: float
    swap: float
    epr_gen: float
    classical_rtt: float
    remote_gate_overhead: float

    @property
    def remote_cost(self) -> float:
        return self.epr_gen + self.classical_rtt + self.remote_gate_overhead


def _validated_latency_values(model: LatencyModel) -> _ValidatedLatencyValues:
    return _ValidatedLatencyValues(
        oneq=_validated_nonnegative_finite(model.oneq, label="oneq"),
        twoq=_validated_nonnegative_finite(model.twoq, label="twoq"),
        swap=_validated_nonnegative_finite(model.swap, label="swap"),
        epr_gen=_validated_nonnegative_finite(model.epr_gen, label="epr_gen"),
        classical_rtt=_validated_nonnegative_finite(
            model.classical_rtt, label="classical_rtt"
        ),
        remote_gate_overhead=_validated_nonnegative_finite(
            model.remote_gate_overhead, label="remote_gate_overhead"
        ),
    )


def _validate_schedule_inputs(
    arch: MultiQPUArchitecture, model: LatencyModel
) -> _ValidatedLatencyValues:
    n_qpus = _validated_nonnegative_int(arch.cfg.n_qpus, label="n_qpus")
    if n_qpus == 0:
        raise ValueError("n_qpus must be positive")
    return _validated_latency_values(model)


def _qubit_qpu_indices(
    mapped: QuantumCircuit, arch: MultiQPUArchitecture
) -> tuple[dict[object, int], list[int]]:
    """Build logical-qubit and physical-to-QPU lookup tables with validation."""
    qubits = mapped.qubits
    qindex = {q: i for i, q in enumerate(qubits)}
    phys_to_qpu: list[int] = []
    n_qpus = arch.cfg.n_qpus
    for phys in range(len(qubits)):
        qpu = arch.qpu_of_phys(phys)
        if isinstance(qpu, bool) or not isinstance(qpu, Integral):
            raise ValueError(
                f"qpu_of_phys({phys}) must return an integer QPU index, got {qpu!r}"
            )
        if qpu < 0 or qpu >= n_qpus:
            raise ValueError(f"qpu_of_phys({phys}) returned out-of-range QPU {qpu}")
        phys_to_qpu.append(int(qpu))
    return qindex, phys_to_qpu


def _instruction_qpus(
    qargs: tuple[object, ...] | list[object],
    qindex: dict[object, int],
    phys_to_qpu: list[int],
) -> tuple[int, ...]:
    """Map operation qargs to QPU ids with a small fast-path for 0/1/2 qubits."""
    argc = len(qargs)
    if argc == 0:
        return ()
    if argc == 1:
        return (phys_to_qpu[qindex[qargs[0]]],)
    if argc == 2:
        return (phys_to_qpu[qindex[qargs[0]]], phys_to_qpu[qindex[qargs[1]]])
    return tuple(phys_to_qpu[qindex[q]] for q in qargs)


def estimate_parallel_makespan(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
) -> ScheduleSummary:
    """Estimate execution makespan assuming QPUs run in parallel with sync on remote ops.

    Model (coarse):
    - Each QPU has its own timeline.
    - Local gates add (oneq/twoq/swap) to that QPU's time.
    - Remote ops require:
        * both involved QPUs to reach a synchronization point
        * add remote cost to both timelines (EPR + RTT + overhead)
    - Barriers produced by `split_into_qpus()` are used only implicitly (remote ops).

    This is intended for *comparative* studies across mappings/topologies.
    """
    lat = _validate_schedule_inputs(arch, model)
    # Validate QPU mappings before downstream splitting/scheduling logic.
    qindex, phys_to_qpu = _qubit_qpu_indices(mapped, arch)

    program = split_into_qpus(mapped, arch)
    t = [0.0] * arch.cfg.n_qpus
    remote_cost = lat.remote_cost

    # Build a simplified linear scan over original circuit instructions, applying costs.

    remote_by_index: dict[int, RemoteOp] = {op.index: op for op in program.remote_ops}
    steps = 0

    for idx, inst in enumerate(mapped.data):
        qpus = _instruction_qpus(inst.qubits, qindex, phys_to_qpu)
        name = inst.operation.name
        if idx in remote_by_index:
            rop = remote_by_index[idx]
            q0, q1 = rop.qpu0, rop.qpu1
            # sync
            sync_time = max(t[q0], t[q1])
            t[q0] = sync_time + remote_cost
            t[q1] = sync_time + remote_cost
            steps += 1
        else:
            if len(qpus) == 0:
                # Ignore 0-qubit directives/metadata operations.
                continue
            if len(qpus) == 1:
                qpu = qpus[0]
                t[qpu] += lat.oneq
            elif len(qpus) == 2:
                qpu0, qpu1 = qpus
                if qpu0 == qpu1:
                    if name == "swap":
                        t[qpu0] += lat.swap
                    else:
                        t[qpu0] += lat.twoq
                else:
                    # should have been remote op; be safe
                    sync_time = max(t[qpu0], t[qpu1])
                    t[qpu0] = sync_time + remote_cost
                    t[qpu1] = sync_time + remote_cost
                    steps += 1
            else:
                # conservative: serialize on first qpu
                t[qpus[0]] += lat.twoq

    return ScheduleSummary(
        makespan=max(t), steps=steps, remote_ops=len(program.remote_ops)
    )


def estimate_parallel_makespan_layered(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
) -> ScheduleSummary:
    """Estimate a parallel makespan using a DAG-layer scheduler with comm-port constraints.

    This is a more *paper-friendly* estimator than :func:`estimate_parallel_makespan`.

    Key differences
    ---------------
    - Uses the circuit DAG layers (i.e., an approximate parallel schedule under gate dependencies).
    - Within a layer, local ops on different QPUs are assumed to proceed in parallel.
    - Remote ops in the same layer are executed in parallel **up to comm port capacity**.

    Remote ops are grouped into "rounds" so that each QPU participates in at most
    `arch.cfg.comm_qubits_per_qpu` remote ops per round.

    The per-layer duration is computed as:
        max(local_layer_duration, remote_rounds * remote_cost)

    This corresponds to a best-case overlap model where remote communication can be pipelined
    alongside local compute when it uses distinct comm resources.
    """
    from qiskit.converters import circuit_to_dag

    lat = _validate_schedule_inputs(arch, model)

    n_qpus = arch.cfg.n_qpus
    comm_ports = _validated_nonnegative_int(
        arch.cfg.comm_qubits_per_qpu, label="comm_qubits_per_qpu"
    )

    dag = circuit_to_dag(mapped)
    qindex, phys_to_qpu = _qubit_qpu_indices(mapped, arch)

    remote_cost = lat.remote_cost

    total_time = 0.0
    total_remote = 0
    steps = 0

    for layer in dag.layers():
        steps += 1
        # Map: qpu -> local duration needed in this layer
        local_dur = [0.0] * n_qpus
        # Remote edges in this layer (qpu0,qpu1)
        remote_pairs: list[tuple[int, int]] = []

        for node in layer["graph"].op_nodes():
            name = node.op.name
            qpus = _instruction_qpus(node.qargs, qindex, phys_to_qpu)
            if len(qpus) == 0:
                continue
            if len(qpus) == 1:
                qpu = qpus[0]
                local_dur[qpu] = max(local_dur[qpu], lat.oneq)
            elif len(qpus) == 2:
                q0, q1 = qpus
                if q0 == q1:
                    if name == "swap":
                        local_dur[q0] = max(local_dur[q0], lat.swap)
                    else:
                        local_dur[q0] = max(local_dur[q0], lat.twoq)
                else:
                    remote_pairs.append((q0, q1))
            else:
                # conservative: treat as local 2q time on the first qpu
                local_dur[qpus[0]] = max(local_dur[qpus[0]], lat.twoq)

        layer_local = max(local_dur)

        # Compute number of remote rounds needed given comm port capacity.
        if not remote_pairs:
            layer_time = layer_local
        else:
            total_remote += len(remote_pairs)
            if comm_ports <= 0:
                # Remote ops infeasible: penalize each unschedulable remote op.
                layer_time = max(layer_local, UNSCHEDULABLE_PENALTY * len(remote_pairs))
            else:
                # Lower bound via per-QPU degree/port
                deg = [0] * n_qpus
                for a, b in remote_pairs:
                    deg[a] += 1
                    deg[b] += 1
                max_deg = max(deg, default=0)
                rounds = (max_deg + comm_ports - 1) // comm_ports
                layer_time = max(layer_local, float(rounds) * remote_cost)

        total_time += layer_time

    return ScheduleSummary(makespan=total_time, steps=steps, remote_ops=total_remote)


@dataclass(frozen=True)
class TopologyScheduleSummary:
    """Topology- and resource-aware schedule summary (paper-friendly)."""

    makespan: float
    layers: int
    remote_ops: int
    remote_rounds: int
    peak_link_util: int
    peak_qpu_ports_used: int


@dataclass(frozen=True)
class RemoteRoundTrace:
    """Resource usage for one packed remote-operation communication round."""

    layer_index: int
    round_index: int
    qpu_pairs: tuple[tuple[int, int], ...]
    duration: float
    qpu_ports_used: tuple[int, ...]
    link_utilization: tuple[tuple[tuple[int, int], int], ...]
    unschedulable_ops: int = 0


@dataclass(frozen=True)
class LayerScheduleTrace:
    """Detailed schedule trace for one circuit DAG layer."""

    layer_index: int
    local_duration: float
    remote_ops: int
    remote_rounds: tuple[RemoteRoundTrace, ...]
    duration: float


@dataclass(frozen=True)
class TopologySchedulePlan:
    """Topology schedule summary plus per-layer/per-round trace details."""

    summary: TopologyScheduleSummary
    layers: tuple[LayerScheduleTrace, ...]


def _effective_classical_rtt(
    cfg: MultiQPUConfig, lat: _ValidatedLatencyValues
) -> float:
    """Compute effective classical RTT under optional overlap (latency hiding)."""
    async_classical = _validated_bool(
        getattr(cfg, "async_classical", False), label="async_classical"
    )
    if async_classical:
        overlap = _validated_nonnegative_finite(
            getattr(cfg, "async_overlap", 0.0), label="async_overlap"
        )
        overlap = min(1.0, overlap)
        return lat.classical_rtt * (1.0 - overlap)
    return lat.classical_rtt


def _topology_schedule_plan(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
) -> TopologySchedulePlan:
    """Build a topology-aware schedule summary and detailed trace.

    Compared to :func:`estimate_parallel_makespan_layered`, this estimator:

    - respects comm port capacity (comm qubits per QPU)
    - respects per-link capacity on the inter-QPU topology
    - charges remote ops proportional to QPU distance (hop count)
    - optionally includes a per-round switch reconfiguration delay

    This is intended for paper plots comparing different topologies/port budgets.

    Notes
    -----
    - We use Qiskit's DAG layers as a dependency-aware parallelization heuristic.
    - Within each layer, local ops are parallel across QPUs.
    - Remote ops are packed into rounds using a greedy algorithm.

    The public summary function projects this plan down to the historical
    :class:`TopologyScheduleSummary` return type.
    """
    from collections import defaultdict

    from qiskit.converters import circuit_to_dag

    from .network import path_edges

    lat = _validate_schedule_inputs(arch, model)

    cfg = arch.cfg
    n_qpus = cfg.n_qpus
    dag = circuit_to_dag(mapped)
    qindex, phys_to_qpu = _qubit_qpu_indices(mapped, arch)

    ports = _validated_nonnegative_int(
        cfg.comm_qubits_per_qpu, label="comm_qubits_per_qpu"
    )
    link_cap = _validated_nonnegative_int(
        getattr(cfg, "link_capacity", 1), label="link_capacity"
    )
    # Clos behaves like an all-to-all switched fabric only when there are enough
    # ports for the 2-level approximation; with one port it falls back to a ring.
    is_switch_like = cfg.inter_topology in ("switch", "mesh") or (
        cfg.inter_topology == "clos" and cfg.comm_qubits_per_qpu >= 2
    )
    sw_pairs_cap = 1_000_000
    sw_reconf = 0.0
    if is_switch_like:
        sw_pairs_cap = _validated_nonnegative_int(
            getattr(cfg, "switch_parallel_links", 1_000_000),
            label="switch_parallel_links",
        )
        sw_reconf = _validated_nonnegative_finite(
            getattr(cfg, "switch_reconfig_delay", 0.0), label="switch_reconfig_delay"
        )

    sp = arch.qpu_shortest_paths()
    classical_eff = _effective_classical_rtt(cfg, lat)

    total = 0.0
    layers = 0
    total_remote = 0
    total_rounds = 0
    peak_link = 0
    peak_ports = 0
    layer_traces: list[LayerScheduleTrace] = []

    edge_cache: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    hop_cache: dict[tuple[int, int], float] = {}
    cost_cache: dict[tuple[int, int], float] = {}

    def pair_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def hops_for(a: int, b: int) -> float:
        key = pair_key(a, b)
        if key not in hop_cache:
            hop_cache[key] = sp.dist[a][b]
        return hop_cache[key]

    def edges_for(a: int, b: int) -> tuple[tuple[int, int], ...]:
        key = pair_key(a, b)
        if key not in edge_cache:
            edge_cache[key] = tuple(path_edges(sp, a, b))
        return edge_cache[key]

    def is_reachable(a: int, b: int) -> bool:
        return hops_for(a, b) < UNREACHABLE_DISTANCE

    def remote_cost(a: int, b: int) -> float:
        key = pair_key(a, b)
        if key not in cost_cache:
            hops = hop_cache.get(key)
            if hops is None:
                hops = sp.dist[a][b]
                hop_cache[key] = hops
            if hops >= UNREACHABLE_DISTANCE:
                cost_cache[key] = UNSCHEDULABLE_PENALTY
            else:
                # EPR generation cost grows with hops (entanglement swapping / path loss proxy)
                cost_cache[key] = (
                    hops * lat.epr_gen + classical_eff + lat.remote_gate_overhead
                )
        return cost_cache[key]

    def unschedulable_round_trace(
        a: int,
        b: int,
        round_traces: list[RemoteRoundTrace],
        *,
        qpu_ports_used: tuple[int, ...] | None = None,
    ) -> RemoteRoundTrace:
        return RemoteRoundTrace(
            layer_index=layers - 1,
            round_index=len(round_traces),
            qpu_pairs=(pair_key(a, b),),
            duration=UNSCHEDULABLE_PENALTY,
            qpu_ports_used=qpu_ports_used or (0,) * n_qpus,
            link_utilization=(),
            unschedulable_ops=1,
        )

    def append_layer_trace(
        *,
        local_duration: float,
        remote_ops: int,
        remote_rounds: list[RemoteRoundTrace],
        duration: float,
    ) -> None:
        layer_traces.append(
            LayerScheduleTrace(
                layer_index=layers - 1,
                local_duration=local_duration,
                remote_ops=remote_ops,
                remote_rounds=tuple(remote_rounds),
                duration=duration,
            )
        )

    for layer in dag.layers():
        layers += 1
        local_dur = [0.0] * n_qpus
        remote_pairs: list[tuple[int, int]] = []

        for node in layer["graph"].op_nodes():
            name = node.op.name
            qs = [qindex[q] for q in node.qargs]
            if len(qs) == 0:
                continue
            if len(qs) == 1:
                qpu = phys_to_qpu[qs[0]]
                local_dur[qpu] = max(local_dur[qpu], lat.oneq)
            elif len(qs) == 2:
                q0 = phys_to_qpu[qs[0]]
                q1 = phys_to_qpu[qs[1]]
                if q0 == q1:
                    if name == "swap":
                        local_dur[q0] = max(local_dur[q0], lat.swap)
                    else:
                        local_dur[q0] = max(local_dur[q0], lat.twoq)
                else:
                    remote_pairs.append((q0, q1))
            else:
                # conservative
                qpu = phys_to_qpu[qs[0]]
                local_dur[qpu] = max(local_dur[qpu], lat.twoq)

        layer_local = max(local_dur) if local_dur else 0.0

        if not remote_pairs:
            total += layer_local
            append_layer_trace(
                local_duration=layer_local,
                remote_ops=0,
                remote_rounds=[],
                duration=layer_local,
            )
            continue

        total_remote += len(remote_pairs)
        round_traces: list[RemoteRoundTrace] = []

        if ports <= 0 or link_cap == 0:
            # Remote ops impossible: either no comm ports or zero link capacity.
            unschedulable_ops = len(remote_pairs)
            rounds_time = UNSCHEDULABLE_PENALTY * unschedulable_ops
            layer_time = max(layer_local, rounds_time)
            total += layer_time
            total_rounds += unschedulable_ops
            for a, b in remote_pairs:
                round_traces.append(unschedulable_round_trace(a, b, round_traces))
            append_layer_trace(
                local_duration=layer_local,
                remote_ops=len(remote_pairs),
                remote_rounds=round_traces,
                duration=layer_time,
            )
            continue

        reachable_pairs: list[tuple[int, int]] = []
        unreachable_pairs = 0
        for a, b in remote_pairs:
            if is_reachable(a, b):
                reachable_pairs.append((a, b))
            else:
                unreachable_pairs += 1

        rounds_time = UNSCHEDULABLE_PENALTY * unreachable_pairs
        rounds_here = unreachable_pairs
        for a, b in remote_pairs:
            if not is_reachable(a, b):
                round_traces.append(unschedulable_round_trace(a, b, round_traces))

        # Greedy round packing with port + link constraints for reachable pairs.
        remaining = sorted(
            reachable_pairs,
            key=lambda ab: hops_for(ab[0], ab[1]),
            reverse=True,
        )

        # Fast path: zero switch pair budget makes every remaining reachable op unschedulable.
        if is_switch_like and sw_pairs_cap == 0 and remaining:
            for a, b in remaining:
                round_traces.append(unschedulable_round_trace(a, b, round_traces))
            rounds_time += UNSCHEDULABLE_PENALTY * len(remaining)
            rounds_here += len(remaining)
            remaining = []

        while remaining:
            used_ports = [0] * n_qpus
            used_link: defaultdict[tuple[int, int], int] = defaultdict(
                int
            )  # edge->count
            used_pairs: set[tuple[int, int]] = set()
            placed_pairs: list[tuple[int, int]] = []
            placed_any = False
            round_max_cost = 0.0

            next_remaining: list[tuple[int, int]] = []
            for a, b in remaining:
                if used_ports[a] >= ports or used_ports[b] >= ports:
                    next_remaining.append((a, b))
                    continue

                # switch network optional cap on distinct pairs
                key = pair_key(a, b)
                if (
                    is_switch_like
                    and len(used_pairs) >= sw_pairs_cap
                    and key not in used_pairs
                ):
                    next_remaining.append((a, b))
                    continue

                edges = edges_for(a, b)
                feasible = True
                for e in edges:
                    if used_link[e] >= link_cap:
                        feasible = False
                        break
                if not feasible:
                    next_remaining.append((a, b))
                    continue

                # place op
                placed_pairs.append(key)
                used_ports[a] += 1
                used_ports[b] += 1
                peak_ports = max(peak_ports, used_ports[a], used_ports[b])
                used_pairs.add(key)
                for e in edges:
                    used_link[e] += 1
                    peak_link = max(peak_link, used_link[e])

                placed_any = True
                round_max_cost = max(round_max_cost, remote_cost(a, b))

            if not placed_any:
                # Constraints can make a "reachable" pair unschedulable (e.g., switch_parallel_links=0).
                # Charge one penalty round and defer the rest.
                skipped = remaining[0]
                next_remaining = remaining[1:]
                rounds_time += UNSCHEDULABLE_PENALTY
                rounds_here += 1
                round_traces.append(
                    unschedulable_round_trace(
                        skipped[0],
                        skipped[1],
                        round_traces,
                        qpu_ports_used=tuple(used_ports),
                    )
                )
                remaining = next_remaining
                continue

            remaining = next_remaining

            # Round duration is the max remote cost in this round + optional reconfig
            if is_switch_like and sw_reconf > 0.0:
                round_max_cost += sw_reconf
            rounds_time += round_max_cost
            rounds_here += 1
            round_traces.append(
                RemoteRoundTrace(
                    layer_index=layers - 1,
                    round_index=len(round_traces),
                    qpu_pairs=tuple(placed_pairs),
                    duration=round_max_cost,
                    qpu_ports_used=tuple(used_ports),
                    link_utilization=tuple(sorted(used_link.items())),
                )
            )

        total_rounds += rounds_here
        layer_time = max(layer_local, rounds_time)
        total += layer_time
        append_layer_trace(
            local_duration=layer_local,
            remote_ops=len(remote_pairs),
            remote_rounds=round_traces,
            duration=layer_time,
        )

    summary = TopologyScheduleSummary(
        makespan=total,
        layers=layers,
        remote_ops=total_remote,
        remote_rounds=total_rounds,
        peak_link_util=peak_link,
        peak_qpu_ports_used=peak_ports,
    )
    return TopologySchedulePlan(summary=summary, layers=tuple(layer_traces))


def estimate_parallel_makespan_topology(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
) -> TopologyScheduleSummary:
    """Estimate makespan with **comm-port + link-capacity** constraints."""
    return _topology_schedule_plan(mapped, arch, model).summary


def estimate_topology_schedule_plan(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
) -> TopologySchedulePlan:
    """Return a topology-aware schedule summary plus per-layer/per-round trace.

    The trace exposes which QPU pairs were packed into each communication round,
    per-QPU port usage, per-link utilization, and unschedulable penalty rounds.
    """
    return _topology_schedule_plan(mapped, arch, model)
