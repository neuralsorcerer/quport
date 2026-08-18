# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit

from quport.aggregation import AggregationPlan, aggregate_remote_operations
from quport.architecture import MultiQPUArchitecture
from quport.config import LatencyModel, MultiQPUConfig, validate_epr_success_prob
from quport.distributed import RemoteOp, split_into_qpus
from quport.entanglement import is_directive
from quport.network import UNREACHABLE_DISTANCE, QpuEdge, path_edges

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
    except (TypeError, ValueError, OverflowError):
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


def _first_remote_partner(qpus: Sequence[int]) -> int | None:
    """Return the first QPU differing from the operation's leading QPU.

    :func:`quport.distributed.split_into_qpus` turns an operation on three or
    more qubits that spans several QPUs into one remote event, between the QPU
    of its first operand and the first operand sitting on a different QPU. The
    layered and topology estimators mirror that choice so that every view of a
    circuit agrees on how many remote events it contains: the split program, the
    remote-op manifest, and all three schedule estimators.

    Returns ``None`` when every operand shares one QPU, i.e. the operation is
    local and costs local two-qubit time.
    """
    first = qpus[0]
    return next((qpu for qpu in qpus[1:] if qpu != first), None)


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
        # Compiler directives (barriers) synchronize but consume no time; they
        # are never RemoteOps (split_into_qpus handles them separately).
        if getattr(inst.operation, "_directive", False):
            continue
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
            # Compiler directives (barriers) act as layer separators in the DAG
            # but consume no time and are never remote operations.
            if getattr(node.op, "_directive", False):
                continue
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
                partner = _first_remote_partner(qpus)
                if partner is None:
                    local_dur[qpus[0]] = max(local_dur[qpus[0]], lat.twoq)
                else:
                    remote_pairs.append((qpus[0], partner))

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


def _json_ready_nonnegative_int(value: object, *, label: str) -> int:
    """Validate an integer schedule-manifest field before JSON export."""
    return _validated_nonnegative_int(value, label=label)


def _json_ready_nonnegative_float(value: object, *, label: str) -> float:
    """Validate a finite non-negative timing field before JSON export."""
    return _validated_nonnegative_finite(value, label=label)


def _json_ready_pair(value: object, *, label: str) -> list[int]:
    """Validate an unordered non-self QPU/link pair and return a JSON list."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-tuple")
    a = _json_ready_nonnegative_int(value[0], label=f"{label}[0]")
    b = _json_ready_nonnegative_int(value[1], label=f"{label}[1]")
    if a == b:
        raise ValueError(f"{label} entries must be distinct")
    return [a, b]


@dataclass(frozen=True)
class TopologyScheduleSummary:
    """Topology- and resource-aware schedule summary (paper-friendly)."""

    makespan: float
    layers: int
    remote_ops: int
    remote_rounds: int
    peak_link_util: int
    peak_qpu_ports_used: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a stable JSON-ready representation of the summary."""
        return {
            "makespan": _json_ready_nonnegative_float(
                self.makespan, label="summary.makespan"
            ),
            "layers": _json_ready_nonnegative_int(self.layers, label="summary.layers"),
            "remote_ops": _json_ready_nonnegative_int(
                self.remote_ops, label="summary.remote_ops"
            ),
            "remote_rounds": _json_ready_nonnegative_int(
                self.remote_rounds, label="summary.remote_rounds"
            ),
            "peak_link_util": _json_ready_nonnegative_int(
                self.peak_link_util, label="summary.peak_link_util"
            ),
            "peak_qpu_ports_used": _json_ready_nonnegative_int(
                self.peak_qpu_ports_used, label="summary.peak_qpu_ports_used"
            ),
        }


@dataclass(frozen=True)
class RemoteRoundTrace:
    """Resource usage for one packed remote-operation communication round.

    ``start_time`` and ``end_time`` are absolute offsets in the schedule plan.
    They make the trace directly consumable by simulators and visualization tools
    without re-integrating layer and round durations from the summary.

    A round is one of two kinds, and ``unschedulable_ops`` tells them apart:

    ``unschedulable_ops == 0``
        A real round. ``qpu_pairs`` lists the operations placed in it, one entry
        each, and ``qpu_ports_used`` and ``link_utilization`` are exactly what
        those operations consume.
    ``unschedulable_ops > 0``
        A penalty round for operations no port, link, or route could serve.
        ``qpu_pairs`` still names them, one entry each, so the count of
        operations a round accounts for is ``unschedulable_ops or
        len(qpu_pairs)`` -- never their sum, which would count a penalty
        operation twice.
    """

    layer_index: int
    round_index: int
    qpu_pairs: tuple[tuple[int, int], ...]
    duration: float
    qpu_ports_used: tuple[int, ...]
    link_utilization: tuple[tuple[tuple[int, int], int], ...]
    unschedulable_ops: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready representation of this communication round."""
        return {
            "layer_index": _json_ready_nonnegative_int(
                self.layer_index, label="round.layer_index"
            ),
            "round_index": _json_ready_nonnegative_int(
                self.round_index, label="round.round_index"
            ),
            "qpu_pairs": [
                _json_ready_pair(pair, label=f"round.qpu_pairs[{idx}]")
                for idx, pair in enumerate(self.qpu_pairs)
            ],
            "duration": _json_ready_nonnegative_float(
                self.duration, label="round.duration"
            ),
            "qpu_ports_used": [
                _json_ready_nonnegative_int(
                    port_count, label=f"round.qpu_ports_used[{idx}]"
                )
                for idx, port_count in enumerate(self.qpu_ports_used)
            ],
            "link_utilization": [
                {
                    "edge": _json_ready_pair(
                        edge, label=f"round.link_utilization[{idx}].edge"
                    ),
                    "count": _json_ready_nonnegative_int(
                        count, label=f"round.link_utilization[{idx}].count"
                    ),
                }
                for idx, (edge, count) in enumerate(self.link_utilization)
            ],
            "unschedulable_ops": _json_ready_nonnegative_int(
                self.unschedulable_ops, label="round.unschedulable_ops"
            ),
            "start_time": _json_ready_nonnegative_float(
                self.start_time, label="round.start_time"
            ),
            "end_time": _json_ready_nonnegative_float(
                self.end_time, label="round.end_time"
            ),
        }


@dataclass(frozen=True)
class LayerScheduleTrace:
    """Detailed schedule trace for one circuit DAG layer.

    ``start_time`` and ``end_time`` are absolute offsets in the schedule plan.
    Local work is assumed to occupy the layer interval while remote rounds are
    serialized from ``start_time`` until their cumulative duration is complete.
    """

    layer_index: int
    local_duration: float
    remote_ops: int
    remote_rounds: tuple[RemoteRoundTrace, ...]
    duration: float
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready representation of this DAG-layer schedule."""
        return {
            "layer_index": _json_ready_nonnegative_int(
                self.layer_index, label="layer.layer_index"
            ),
            "local_duration": _json_ready_nonnegative_float(
                self.local_duration, label="layer.local_duration"
            ),
            "remote_ops": _json_ready_nonnegative_int(
                self.remote_ops, label="layer.remote_ops"
            ),
            "remote_rounds": [
                round_trace.to_dict() for round_trace in self.remote_rounds
            ],
            "duration": _json_ready_nonnegative_float(
                self.duration, label="layer.duration"
            ),
            "start_time": _json_ready_nonnegative_float(
                self.start_time, label="layer.start_time"
            ),
            "end_time": _json_ready_nonnegative_float(
                self.end_time, label="layer.end_time"
            ),
        }


@dataclass(frozen=True)
class TopologySchedulePlan:
    """Topology schedule summary plus per-layer/per-round trace details."""

    summary: TopologyScheduleSummary
    layers: tuple[LayerScheduleTrace, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready schedule manifest.

        The manifest preserves absolute layer and round timing, resource usage,
        and unschedulable penalty rounds, making compiled schedules easier to
        feed into visualization, simulation, or artifact-export workflows.
        """
        return {
            "summary": self.summary.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
        }


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

    def append_unschedulable_round_trace(
        a: int,
        b: int,
        round_traces: list[RemoteRoundTrace],
        *,
        layer_start: float,
        elapsed_rounds_time: float,
        qpu_ports_used: tuple[int, ...] | None = None,
    ) -> float:
        """Append one penalty round and return updated remote-round elapsed time."""
        round_start = layer_start + elapsed_rounds_time
        round_traces.append(
            RemoteRoundTrace(
                layer_index=layers - 1,
                round_index=len(round_traces),
                qpu_pairs=(pair_key(a, b),),
                duration=UNSCHEDULABLE_PENALTY,
                qpu_ports_used=qpu_ports_used or (0,) * n_qpus,
                link_utilization=(),
                unschedulable_ops=1,
                start_time=round_start,
                end_time=round_start + UNSCHEDULABLE_PENALTY,
            )
        )
        return elapsed_rounds_time + UNSCHEDULABLE_PENALTY

    def append_layer_trace(
        *,
        local_duration: float,
        remote_ops: int,
        remote_rounds: list[RemoteRoundTrace],
        duration: float,
        layer_start: float,
    ) -> None:
        layer_traces.append(
            LayerScheduleTrace(
                layer_index=layers - 1,
                local_duration=local_duration,
                remote_ops=remote_ops,
                remote_rounds=tuple(remote_rounds),
                duration=duration,
                start_time=layer_start,
                end_time=layer_start + duration,
            )
        )

    for layer in dag.layers():
        layer_start = total
        layers += 1
        local_dur = [0.0] * n_qpus
        remote_pairs: list[tuple[int, int]] = []

        for node in layer["graph"].op_nodes():
            # Compiler directives (barriers) act as layer separators in the DAG
            # but consume no time and are never remote operations.
            if getattr(node.op, "_directive", False):
                continue
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
                op_qpus = [phys_to_qpu[q] for q in qs]
                partner = _first_remote_partner(op_qpus)
                if partner is None:
                    local_dur[op_qpus[0]] = max(local_dur[op_qpus[0]], lat.twoq)
                else:
                    remote_pairs.append((op_qpus[0], partner))

        layer_local = max(local_dur) if local_dur else 0.0

        if not remote_pairs:
            total += layer_local
            append_layer_trace(
                local_duration=layer_local,
                remote_ops=0,
                remote_rounds=[],
                duration=layer_local,
                layer_start=layer_start,
            )
            continue

        total_remote += len(remote_pairs)
        round_traces: list[RemoteRoundTrace] = []

        if ports <= 0 or link_cap == 0:
            # Remote ops impossible: either no comm ports or zero link capacity.
            unschedulable_ops = len(remote_pairs)
            rounds_time = 0.0
            for a, b in remote_pairs:
                rounds_time = append_unschedulable_round_trace(
                    a,
                    b,
                    round_traces,
                    layer_start=layer_start,
                    elapsed_rounds_time=rounds_time,
                )
            layer_time = max(layer_local, rounds_time)
            total += layer_time
            total_rounds += unschedulable_ops
            append_layer_trace(
                local_duration=layer_local,
                remote_ops=len(remote_pairs),
                remote_rounds=round_traces,
                duration=layer_time,
                layer_start=layer_start,
            )
            continue

        reachable_pairs: list[tuple[int, int]] = []
        unreachable_pairs = 0
        for a, b in remote_pairs:
            if is_reachable(a, b):
                reachable_pairs.append((a, b))
            else:
                unreachable_pairs += 1

        rounds_time = 0.0
        rounds_here = unreachable_pairs
        for a, b in remote_pairs:
            if not is_reachable(a, b):
                rounds_time = append_unschedulable_round_trace(
                    a,
                    b,
                    round_traces,
                    layer_start=layer_start,
                    elapsed_rounds_time=rounds_time,
                )

        # Greedy round packing with port + link constraints for reachable pairs.
        remaining = sorted(
            reachable_pairs,
            key=lambda ab: hops_for(ab[0], ab[1]),
            reverse=True,
        )

        # Fast path: zero switch pair budget makes every remaining reachable op unschedulable.
        if is_switch_like and sw_pairs_cap == 0 and remaining:
            for a, b in remaining:
                rounds_time = append_unschedulable_round_trace(
                    a,
                    b,
                    round_traces,
                    layer_start=layer_start,
                    elapsed_rounds_time=rounds_time,
                )
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
                # Termination guard, not a costing path. Every round starts with
                # empty port/link/pair usage, so the three deferral tests above
                # reduce to ports <= 0, link_capacity == 0 and switch_parallel_links
                # == 0 -- each already diverted by a fast path before this loop.
                # A round that places nothing would leave `remaining` unchanged and
                # spin forever, so charge one penalty round and defer the rest.
                skipped = remaining[0]
                next_remaining = remaining[1:]
                rounds_time = append_unschedulable_round_trace(
                    skipped[0],
                    skipped[1],
                    round_traces,
                    layer_start=layer_start,
                    elapsed_rounds_time=rounds_time,
                    qpu_ports_used=tuple(used_ports),
                )
                rounds_here += 1
                remaining = next_remaining
                continue

            remaining = next_remaining

            # Round duration is the max remote cost in this round + optional reconfig
            if is_switch_like and sw_reconf > 0.0:
                round_max_cost += sw_reconf
            round_start = layer_start + rounds_time
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
                    start_time=round_start,
                    end_time=round_start + round_max_cost,
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
            layer_start=layer_start,
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


def audit_topology_schedule_plan(
    plan: TopologySchedulePlan,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
    mapped: QuantumCircuit | None = None,
) -> tuple[str, ...]:
    """Re-derive a schedule plan's numbers from the outside, and report mismatches.

    The estimator computes the summary and the trace in one pass, which means
    nothing in it cross-checks the two, or checks either against the resource
    budgets the plan claims to respect. A downstream consumer of
    ``schedule_trace.json`` has no way to tell a sound manifest from a
    self-consistent-looking wrong one. This walks the finished plan and rebuilds
    each figure independently:

    - layer and round intervals chain, and every ``end_time`` is its
      ``start_time`` plus its duration;
    - a layer lasts ``max(local_duration, sum of its round durations)``;
    - each round's ``qpu_ports_used`` and ``link_utilization`` are exactly what
      routing its ``qpu_pairs`` over shortest paths consumes, and neither exceeds
      ``comm_qubits_per_qpu`` or ``link_capacity``;
    - each round lasts as long as its slowest placed operation;
    - the six summary fields agree with the trace they summarise.

    Passing ``mapped`` adds the one check that needs the circuit: that the plan
    accounts for exactly the operations that actually span QPUs, counted the way
    :func:`quport.distributed.split_into_qpus` counts them.

    What this does *not* re-derive is the cost model itself -- per-hop EPR time,
    classical-RTT overlap, and the round-packing policy are taken as given, since
    they are modelling choices rather than claims. It checks that the plan is a
    faithful, feasible account of those choices.

    Returns
    -------
    tuple[str, ...]
        One description per inconsistency, in the order found. Empty means every
        figure was reproduced.
    """
    lat = _validate_schedule_inputs(arch, model)
    cfg = arch.cfg
    n_qpus = cfg.n_qpus
    sp = arch.qpu_shortest_paths()
    ports = _validated_nonnegative_int(
        cfg.comm_qubits_per_qpu, label="comm_qubits_per_qpu"
    )
    link_cap = _validated_nonnegative_int(
        getattr(cfg, "link_capacity", 1), label="link_capacity"
    )
    is_switch_like = cfg.inter_topology in ("switch", "mesh") or (
        cfg.inter_topology == "clos" and cfg.comm_qubits_per_qpu >= 2
    )
    reconfig = (
        _validated_nonnegative_finite(
            getattr(cfg, "switch_reconfig_delay", 0.0), label="switch_reconfig_delay"
        )
        if is_switch_like
        else 0.0
    )
    classical_eff = _effective_classical_rtt(cfg, lat)

    problems: list[str] = []

    def close(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)

    summary = plan.summary
    elapsed = 0.0
    seen_rounds = 0
    seen_remote = 0
    peak_link = 0
    peak_ports = 0

    for index, layer in enumerate(plan.layers):
        where = f"layer {index}"
        if layer.layer_index != index:
            problems.append(f"{where}: layer_index is {layer.layer_index}")
        if not close(layer.start_time, elapsed):
            problems.append(
                f"{where}: starts at {layer.start_time}, but the layers before it "
                f"end at {elapsed}"
            )
        if not close(layer.end_time, layer.start_time + layer.duration):
            problems.append(
                f"{where}: ends at {layer.end_time}, not start + duration "
                f"({layer.start_time + layer.duration})"
            )

        rounds_time = 0.0
        for r_index, rnd in enumerate(layer.remote_rounds):
            spot = f"{where} round {r_index}"
            if rnd.layer_index != index:
                problems.append(f"{spot}: layer_index is {rnd.layer_index}")
            if rnd.round_index != r_index:
                problems.append(f"{spot}: round_index is {rnd.round_index}")
            if not close(rnd.start_time, layer.start_time + rounds_time):
                problems.append(
                    f"{spot}: starts at {rnd.start_time}, but the rounds before it "
                    f"end at {layer.start_time + rounds_time}"
                )
            if not close(rnd.end_time, rnd.start_time + rnd.duration):
                problems.append(
                    f"{spot}: ends at {rnd.end_time}, not start + duration "
                    f"({rnd.start_time + rnd.duration})"
                )
            rounds_time += rnd.duration

            if len(rnd.qpu_ports_used) != n_qpus:
                problems.append(
                    f"{spot}: qpu_ports_used has {len(rnd.qpu_ports_used)} entries "
                    f"for {n_qpus} QPUs"
                )
            for qpu, used in enumerate(rnd.qpu_ports_used):
                if used > ports:
                    problems.append(
                        f"{spot}: QPU {qpu} holds {used} ports, budget is {ports}"
                    )
                peak_ports = max(peak_ports, used)
            for edge, count in rnd.link_utilization:
                if count > link_cap:
                    problems.append(
                        f"{spot}: link {edge} carries {count}, capacity is {link_cap}"
                    )
                peak_link = max(peak_link, count)

            if rnd.unschedulable_ops:
                # A penalty round stands for operations that could not be
                # placed. It still lists the pair it was going to serve, for
                # diagnostics, so its operands are named twice and count once.
                if not close(rnd.duration, UNSCHEDULABLE_PENALTY):
                    problems.append(
                        f"{spot}: penalty round lasts {rnd.duration}, "
                        f"not {UNSCHEDULABLE_PENALTY}"
                    )
                if len(rnd.qpu_pairs) != rnd.unschedulable_ops:
                    problems.append(
                        f"{spot}: names {len(rnd.qpu_pairs)} pairs for "
                        f"{rnd.unschedulable_ops} unschedulable operations"
                    )
            else:
                expected_ports = [0] * n_qpus
                expected_link: dict[QpuEdge, int] = {}
                worst = 0.0
                for a, b in rnd.qpu_pairs:
                    expected_ports[a] += 1
                    expected_ports[b] += 1
                    for edge in path_edges(sp, a, b):
                        expected_link[edge] = expected_link.get(edge, 0) + 1
                    worst = max(
                        worst,
                        sp.dist[a][b] * lat.epr_gen
                        + classical_eff
                        + lat.remote_gate_overhead,
                    )
                if tuple(expected_ports) != tuple(rnd.qpu_ports_used):
                    problems.append(
                        f"{spot}: reports ports {list(rnd.qpu_ports_used)}, but its "
                        f"pairs consume {expected_ports}"
                    )
                if tuple(sorted(expected_link.items())) != tuple(rnd.link_utilization):
                    problems.append(
                        f"{spot}: reports links {list(rnd.link_utilization)}, but its "
                        f"pairs consume {sorted(expected_link.items())}"
                    )
                if rnd.qpu_pairs and not close(rnd.duration, worst + reconfig):
                    problems.append(
                        f"{spot}: lasts {rnd.duration}, but its slowest operation "
                        f"takes {worst + reconfig}"
                    )

            seen_rounds += 1
            seen_remote += rnd.unschedulable_ops or len(rnd.qpu_pairs)

        if not close(layer.duration, max(layer.local_duration, rounds_time)):
            problems.append(
                f"{where}: lasts {layer.duration}, not max(local {layer.local_duration},"
                f" rounds {rounds_time})"
            )
        counted = sum(
            rnd.unschedulable_ops or len(rnd.qpu_pairs) for rnd in layer.remote_rounds
        )
        if layer.remote_ops != counted:
            problems.append(
                f"{where}: claims {layer.remote_ops} remote ops, its rounds hold "
                f"{counted}"
            )
        elapsed += layer.duration

    if summary.layers != len(plan.layers):
        problems.append(
            f"summary: claims {summary.layers} layers, the trace has "
            f"{len(plan.layers)}"
        )
    if not close(summary.makespan, elapsed):
        problems.append(
            f"summary: makespan {summary.makespan}, layer durations sum to {elapsed}"
        )
    if summary.remote_ops != seen_remote:
        problems.append(
            f"summary: claims {summary.remote_ops} remote ops, the trace holds "
            f"{seen_remote}"
        )
    if summary.remote_rounds != seen_rounds:
        problems.append(
            f"summary: claims {summary.remote_rounds} rounds, the trace has "
            f"{seen_rounds}"
        )
    if summary.peak_link_util != peak_link:
        problems.append(
            f"summary: peak_link_util {summary.peak_link_util}, the trace peaks at "
            f"{peak_link}"
        )
    if summary.peak_qpu_ports_used != peak_ports:
        problems.append(
            f"summary: peak_qpu_ports_used {summary.peak_qpu_ports_used}, the trace "
            f"peaks at {peak_ports}"
        )

    if mapped is not None:
        actual = _count_cross_qpu_operations(mapped, arch)
        if summary.remote_ops != actual:
            problems.append(
                f"summary: accounts for {summary.remote_ops} remote ops, the circuit "
                f"has {actual} operations spanning more than one QPU"
            )

    return tuple(problems)


def _count_cross_qpu_operations(
    mapped: QuantumCircuit, arch: MultiQPUArchitecture
) -> int:
    """Count operations spanning more than one QPU, as one remote event each.

    A wide operation is charged once, between its leading QPU and the first
    operand elsewhere, matching :func:`quport.distributed.split_into_qpus` and
    every schedule estimator.
    """
    qindex, phys_to_qpu = _qubit_qpu_indices(mapped, arch)
    count = 0
    for instruction in mapped.data:
        if getattr(instruction.operation, "_directive", False):
            continue
        qpus = _instruction_qpus(instruction.qubits, qindex, phys_to_qpu)
        if len(qpus) >= 2 and _first_remote_partner(qpus) is not None:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Entanglement-aware event-driven scheduling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntanglementScheduleSummary:
    """Makespan and resource usage under an aggregated entanglement plan.

    Attributes
    ----------
    makespan:
        End of the last activity on any qubit, comm port, or link.
    epr_pairs:
        EPR pairs the plan consumes, counting two per teleport block.
    entanglement_time:
        Total wall time links spend distributing entanglement, summed over
        links. Divided by ``makespan`` it gives average interconnect occupancy.
    unschedulable_gates:
        Cross-QPU gates that no port/link budget could serve. Each is charged
        :data:`UNSCHEDULABLE_PENALTY` so infeasible designs stay comparable
        rather than raising.
    peak_ports_in_use:
        Per QPU, the largest number of comm ports occupied at any instant. Never
        exceeds the QPU's port budget.
    port_busy_time / qpu_busy_time:
        Per QPU, total comm-port occupancy and total local gate time. Both are
        summed over transactions and gates respectively, so either can exceed the
        makespan: a QPU holds several ports at once and runs gates on disjoint
        qubits concurrently, and every one of those is counted.
    link_busy_time:
        Per inter-QPU link, total occupancy, sorted by link.
    """

    makespan: float
    blocks: int
    epr_pairs: int
    remote_gates: int
    unschedulable_gates: int
    entanglement_time: float
    peak_ports_in_use: tuple[int, ...]
    port_busy_time: tuple[float, ...]
    qpu_busy_time: tuple[float, ...]
    link_busy_time: tuple[tuple[QpuEdge, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation of the summary."""
        return {
            "makespan": _json_ready_nonnegative_float(
                self.makespan, label="entanglement.makespan"
            ),
            "blocks": _json_ready_nonnegative_int(
                self.blocks, label="entanglement.blocks"
            ),
            "epr_pairs": _json_ready_nonnegative_int(
                self.epr_pairs, label="entanglement.epr_pairs"
            ),
            "remote_gates": _json_ready_nonnegative_int(
                self.remote_gates, label="entanglement.remote_gates"
            ),
            "unschedulable_gates": _json_ready_nonnegative_int(
                self.unschedulable_gates, label="entanglement.unschedulable_gates"
            ),
            "entanglement_time": _json_ready_nonnegative_float(
                self.entanglement_time, label="entanglement.entanglement_time"
            ),
            "peak_ports_in_use": [
                _json_ready_nonnegative_int(
                    value, label=f"entanglement.peak_ports_in_use[{index}]"
                )
                for index, value in enumerate(self.peak_ports_in_use)
            ],
            "port_busy_time": [
                _json_ready_nonnegative_float(
                    value, label=f"entanglement.port_busy_time[{index}]"
                )
                for index, value in enumerate(self.port_busy_time)
            ],
            "qpu_busy_time": [
                _json_ready_nonnegative_float(
                    value, label=f"entanglement.qpu_busy_time[{index}]"
                )
                for index, value in enumerate(self.qpu_busy_time)
            ],
            "link_busy_time": [
                {
                    "edge": _json_ready_pair(
                        edge, label=f"entanglement.link_busy_time[{index}].edge"
                    ),
                    "busy": _json_ready_nonnegative_float(
                        busy, label=f"entanglement.link_busy_time[{index}].busy"
                    ),
                }
                for index, (edge, busy) in enumerate(self.link_busy_time)
            ],
        }


class _ResourcePool:
    """A fixed set of interchangeable servers tracked by next-free time.

    ``acquire`` returns the earliest time any server frees up and removes it
    from the pool; ``release`` returns it with a new free time. Holding a server
    across many instructions is what lets a cat copy pin a comm port for its
    whole window.
    """

    __slots__ = ("_free", "capacity")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._free: list[float] = [0.0] * capacity

    def available(self) -> bool:
        return bool(self._free)

    def acquire(self) -> float:
        return heapq.heappop(self._free)

    def release(self, free_at: float) -> None:
        heapq.heappush(self._free, free_at)

    def horizon(self) -> float:
        return max(self._free, default=0.0)


@dataclass
class _BlockRuntime:
    """Mutable schedule state for one in-flight communication block."""

    ready: float = 0.0
    port_start: float = 0.0
    hops: int = 0
    edges: tuple[QpuEdge, ...] = ()
    holds_port: bool = False
    feasible: bool = True


def estimate_entanglement_schedule(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
    *,
    plan: AggregationPlan | None = None,
    ports_per_qpu: int | Sequence[int] | None = None,
) -> EntanglementScheduleSummary:
    """Schedule a mapped circuit as an entanglement-resource-constrained system.

    How this differs from the other estimators
    ------------------------------------------
    :func:`estimate_parallel_makespan_layered` and
    :func:`estimate_parallel_makespan_topology` slice the circuit into DAG layers
    and charge each layer its slowest operation. That imposes a global barrier
    between layers, which over-serialises a machine whose QPUs only ever
    synchronise on shared qubits, and it charges one entanglement transaction per
    cross-QPU gate.

    This estimator instead runs an as-soon-as-possible list schedule in program
    order against explicit resources:

    - one timeline per physical qubit, so independent QPUs drift apart freely;
    - a pool of ``comm_qubits_per_qpu`` ports per QPU, each **held for a whole
      block** rather than for a single gate, which is what makes port scarcity
      bite;
    - ``link_capacity`` channels on every inter-QPU link along the routed path;
    - hop-scaled, probabilistic entanglement distribution
      (:meth:`quport.config.LatencyModel.expected_epr_time`).

    Gates are grouped by :func:`quport.aggregation.aggregate_remote_operations`,
    so a run of gates sharing one root and one remote QPU costs a single EPR pair
    and a single protocol setup.

    Parameters
    ----------
    plan:
        A pre-computed aggregation plan. When omitted, one is built with the
        architecture's own port budget. A supplied plan must have been built with
        the same budget the schedule uses, otherwise it asks for ports that do
        not exist and the call raises.
    ports_per_qpu:
        Override the comm-port budget, matching the parameter of
        :func:`quport.aggregation.aggregate_remote_operations`. Passing a large
        value measures the port-unconstrained makespan.

    Raises
    ------
    ValueError
        If ``plan`` holds more concurrent cat copies on some QPU than the
        schedule's port budget allows.

    Returns
    -------
    EntanglementScheduleSummary
    """
    lat = _validate_schedule_inputs(arch, model)
    success = validate_epr_success_prob(model.epr_success_prob)

    cfg = arch.cfg
    n_qpus = cfg.n_qpus
    qindex, phys_to_qpu = _qubit_qpu_indices(mapped, arch)
    n_phys = len(mapped.qubits)

    ports = _normalized_port_budget(ports_per_qpu, arch)
    link_cap = _validated_nonnegative_int(
        getattr(cfg, "link_capacity", 1), label="link_capacity"
    )

    if plan is None:
        plan = aggregate_remote_operations(mapped, arch, ports_per_qpu=ports)
    elif not isinstance(plan, AggregationPlan):
        raise ValueError("plan must be an AggregationPlan")
    else:
        # A plan built against a larger budget would silently exhaust the port
        # pools here and be reported as unschedulable; say so instead.
        for qpu, peak in enumerate(plan.peak_cat_copies[:n_qpus]):
            if peak > ports[qpu]:
                raise ValueError(
                    "aggregation plan exceeds the schedule's comm-port budget "
                    f"(QPU {qpu} holds {peak} cat copies, budget is {ports[qpu]}); "
                    "build the plan with the same ports_per_qpu"
                )

    sp = arch.qpu_shortest_paths()
    classical_eff = _effective_classical_rtt(cfg, lat)

    blocks = plan.blocks
    runtime = [_BlockRuntime() for _ in blocks]
    starts: dict[int, list[int]] = {}
    members: dict[int, list[int]] = {}
    ends: dict[int, list[int]] = {}
    for ordinal, block in enumerate(blocks):
        starts.setdefault(block.start_index, []).append(ordinal)
        ends.setdefault(block.end_index, []).append(ordinal)
        for gate_index in block.gate_indices:
            members.setdefault(gate_index, []).append(ordinal)

    qubit_ready = [0.0] * n_phys
    port_pools = [_ResourcePool(ports[qpu]) for qpu in range(n_qpus)]
    link_pools: dict[QpuEdge, _ResourcePool] = {}
    link_busy: dict[QpuEdge, float] = {}
    port_intervals: list[list[tuple[float, float]]] = [[] for _ in range(n_qpus)]
    port_busy = [0.0] * n_qpus
    qpu_busy = [0.0] * n_qpus
    entanglement_time = 0.0
    unschedulable = 0
    remote_gates = 0

    def link_pool(edge: QpuEdge) -> _ResourcePool:
        pool = link_pools.get(edge)
        if pool is None:
            pool = _ResourcePool(link_cap)
            link_pools[edge] = pool
        return pool

    def reserve_links(
        edges: tuple[QpuEdge, ...], earliest: float, duration: float
    ) -> tuple[float, float]:
        """Occupy one channel per edge for ``duration``.

        Returns the ``(start, finish)`` window actually granted, which can begin
        later than ``earliest`` when a link along the path is saturated.
        """
        nonlocal entanglement_time
        held: list[tuple[QpuEdge, _ResourcePool]] = []
        start = earliest
        for edge in edges:
            pool = link_pool(edge)
            start = max(start, pool.acquire())
            held.append((edge, pool))
        finish = start + duration
        for edge, pool in held:
            pool.release(finish)
            link_busy[edge] = link_busy.get(edge, 0.0) + duration
            entanglement_time += duration
        return start, finish

    def establish(ordinal: int) -> None:
        nonlocal unschedulable
        block = blocks[ordinal]
        state = runtime[ordinal]
        source = block.root_qpu
        host = block.remote_qpu
        hops = sp.dist[source][host]

        if (
            hops >= UNREACHABLE_DISTANCE
            or link_cap == 0
            or not port_pools[host].available()
            or not port_pools[source].available()
        ):
            state.feasible = False
            state.ready = qubit_ready[block.root_phys] + UNSCHEDULABLE_PENALTY
            qubit_ready[block.root_phys] = state.ready
            unschedulable += block.size()
            return

        state.hops = hops
        state.edges = tuple(path_edges(sp, source, host))

        host_port = port_pools[host].acquire()
        source_port = port_pools[source].acquire()
        earliest = max(qubit_ready[block.root_phys], host_port, source_port)

        # Distribute the pair, then run the entangler (a local CX, a Z-basis
        # measurement, and one classical message) plus the protocol overhead.
        distribute = model.expected_epr_time(hops)
        start, distributed = reserve_links(state.edges, earliest, distribute)
        ready = distributed + classical_eff + lat.remote_gate_overhead

        state.ready = ready
        state.port_start = start
        state.holds_port = True
        # The root's own port is only needed until the entangler completes.
        port_pools[source].release(ready)
        port_intervals[source].append((start, ready))
        port_busy[source] += ready - start
        qubit_ready[block.root_phys] = ready

    def release(ordinal: int) -> None:
        block = blocks[ordinal]
        state = runtime[ordinal]
        host = block.remote_qpu

        if not state.feasible:
            qubit_ready[block.root_phys] = max(
                qubit_ready[block.root_phys], state.ready
            )
            return

        if block.protocol == "teleport":
            # The return trip is a second EPR pair back to the root's QPU.
            _start, distributed = reserve_links(
                state.edges, state.ready, model.expected_epr_time(state.hops)
            )
            finish = distributed + classical_eff
        else:
            # Cat-disentangler: an X-basis measurement plus one classical message.
            finish = state.ready + classical_eff

        qubit_ready[block.root_phys] = max(qubit_ready[block.root_phys], finish)
        if state.holds_port:
            port_pools[host].release(finish)
            port_intervals[host].append((state.port_start, finish))
            port_busy[host] += finish - state.port_start
            state.holds_port = False

    for index, instruction in enumerate(mapped.data):
        operation = instruction.operation
        qubits = [qindex[qubit] for qubit in instruction.qubits]

        if is_directive(operation):
            targets = qubits if qubits else list(range(n_phys))
            if targets:
                sync = max(qubit_ready[qubit] for qubit in targets)
                for qubit in targets:
                    qubit_ready[qubit] = sync
            continue

        if not qubits:
            continue

        for ordinal in starts.get(index, ()):
            establish(ordinal)

        ordinals = members.get(index)
        if ordinals is not None:
            remote_gates += 1
            host = blocks[ordinals[0]].remote_qpu
            roots = {blocks[ordinal].root_phys for ordinal in ordinals}
            duration = lat.swap if operation.name == "swap" else lat.twoq
            start = max(
                [runtime[ordinal].ready for ordinal in ordinals]
                + [qubit_ready[qubit] for qubit in qubits if qubit not in roots]
            )
            finish = start + duration
            for ordinal in ordinals:
                runtime[ordinal].ready = finish
            for qubit in qubits:
                if qubit not in roots:
                    qubit_ready[qubit] = finish
            qpu_busy[host] += duration

            for ordinal in ends.get(index, ()):
                release(ordinal)
            continue

        qpus = {phys_to_qpu[qubit] for qubit in qubits}
        if len(qpus) > 1:
            # A cross-QPU gate the aggregator could not serve at all.
            remote_gates += 1
            unschedulable += 1
            start = max(qubit_ready[qubit] for qubit in qubits)
            finish = start + UNSCHEDULABLE_PENALTY
            for qubit in qubits:
                qubit_ready[qubit] = finish
            continue

        qpu = phys_to_qpu[qubits[0]]
        if len(qubits) == 1:
            duration = lat.oneq
        elif operation.name == "swap":
            duration = lat.swap
        else:
            duration = lat.twoq
        start = max(qubit_ready[qubit] for qubit in qubits)
        finish = start + duration
        for qubit in qubits:
            qubit_ready[qubit] = finish
        qpu_busy[qpu] += duration

    # Any block still holding a port ran past the end of the instruction list.
    for ordinal, state in enumerate(runtime):
        if state.holds_port:
            release(ordinal)

    makespan = max(qubit_ready, default=0.0)
    for pool in port_pools:
        makespan = max(makespan, pool.horizon())
    for pool in link_pools.values():
        makespan = max(makespan, pool.horizon())

    return EntanglementScheduleSummary(
        makespan=makespan,
        blocks=len(blocks),
        epr_pairs=plan.epr_pairs,
        remote_gates=remote_gates,
        unschedulable_gates=unschedulable,
        entanglement_time=entanglement_time,
        peak_ports_in_use=tuple(
            _peak_overlap(intervals) for intervals in port_intervals
        ),
        port_busy_time=tuple(port_busy),
        qpu_busy_time=tuple(qpu_busy),
        link_busy_time=tuple(sorted(link_busy.items())),
    )


def _serialised_qubit_time(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    lat: _ValidatedLatencyValues,
) -> float:
    """Longest total duration of the gates serialised on any single qubit.

    Every gate touching a qubit occupies that qubit's timeline, so no schedule
    can finish sooner than the busiest qubit's own work. Cross-QPU gates are
    excluded: they run against a cat copy on the host QPU, on that copy's
    timeline rather than the root's, which is exactly the parallelism the
    entanglement model buys.
    """
    qindex, phys_to_qpu = _qubit_qpu_indices(mapped, arch)
    load = [0.0] * len(mapped.qubits)
    for instruction in mapped.data:
        if is_directive(instruction.operation):
            continue
        qubits = [qindex[qubit] for qubit in instruction.qubits]
        if not qubits:
            continue
        if len({phys_to_qpu[qubit] for qubit in qubits}) > 1:
            continue
        if len(qubits) == 1:
            duration = lat.oneq
        elif instruction.operation.name == "swap":
            duration = lat.swap
        else:
            duration = lat.twoq
        for qubit in qubits:
            load[qubit] += duration
    return max(load, default=0.0)


def audit_entanglement_schedule(
    summary: EntanglementScheduleSummary,
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    model: LatencyModel,
    *,
    plan: AggregationPlan | None = None,
    ports_per_qpu: int | Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Check an entanglement schedule against the properties it must satisfy.

    :func:`estimate_entanglement_schedule` returns aggregates and no trace, so
    unlike :func:`audit_topology_schedule_plan` there is no event log to replay.
    What can still be checked from the outside is everything the aggregates are
    related by *theorem* rather than by convention:

    - a QPU with ``P`` ports cannot hold more than ``P`` cat copies at once, nor
      accrue more than ``P * makespan`` of port-busy time; a link with ``C``
      channels cannot accrue more than ``C * makespan``;
    - ``entanglement_time`` is by definition the total link occupancy, so it
      equals the sum of ``link_busy_time``;
    - every gate on a qubit occupies that qubit's timeline, so the busiest
      qubit's own work is a lower bound on the makespan;
    - ``remote_gates`` is the number of operations in the circuit that span more
      than one QPU, counted as :func:`quport.distributed.split_into_qpus` counts
      them, and no more gates can be unschedulable than there are remote ones;
    - with ``plan``, the summary describes that plan's blocks and EPR pairs.

    ``ports_per_qpu`` must be whatever the schedule was produced with; the
    default reads the architecture's own budget, matching the estimator.

    A caller who wants monotonicity instead -- that widening a resource never
    lengthens a schedule -- can get it by scheduling one fixed ``plan`` twice and
    comparing, which is a property of the estimator rather than of one result.

    Returns
    -------
    tuple[str, ...]
        One description per violated property, in the order found. Empty means
        the summary is consistent and feasible.
    """
    lat = _validate_schedule_inputs(arch, model)
    n_qpus = arch.cfg.n_qpus
    ports = _normalized_port_budget(ports_per_qpu, arch)
    link_cap = _validated_nonnegative_int(
        getattr(arch.cfg, "link_capacity", 1), label="link_capacity"
    )
    makespan = summary.makespan
    problems: list[str] = []

    for label, values in (
        ("peak_ports_in_use", summary.peak_ports_in_use),
        ("port_busy_time", summary.port_busy_time),
        ("qpu_busy_time", summary.qpu_busy_time),
    ):
        if len(values) != n_qpus:
            problems.append(f"{label} has {len(values)} entries for {n_qpus} QPUs")

    for qpu, peak in enumerate(summary.peak_ports_in_use[:n_qpus]):
        if peak > ports[qpu]:
            problems.append(
                f"QPU {qpu} holds {peak} cat copies at once, budget is {ports[qpu]}"
            )
    for qpu, busy in enumerate(summary.port_busy_time[:n_qpus]):
        if busy > ports[qpu] * makespan + 1e-6:
            problems.append(
                f"QPU {qpu} accrues {busy} port-busy time, but {ports[qpu]} ports "
                f"over a {makespan} makespan allow at most {ports[qpu] * makespan}"
            )
    for edge, busy in summary.link_busy_time:
        if busy > link_cap * makespan + 1e-6:
            problems.append(
                f"link {edge} accrues {busy} busy time, but {link_cap} channels "
                f"over a {makespan} makespan allow at most {link_cap * makespan}"
            )

    total_link = math.fsum(busy for _edge, busy in summary.link_busy_time)
    if not math.isclose(
        summary.entanglement_time, total_link, rel_tol=1e-9, abs_tol=1e-6
    ):
        problems.append(
            f"entanglement_time is {summary.entanglement_time}, but the per-link "
            f"busy times sum to {total_link}"
        )

    floor = _serialised_qubit_time(mapped, arch, lat)
    if makespan < floor - 1e-6:
        problems.append(
            f"makespan is {makespan}, below the {floor} of work serialised on the "
            f"busiest single qubit"
        )

    if summary.unschedulable_gates > summary.remote_gates:
        problems.append(
            f"{summary.unschedulable_gates} gates unschedulable, but only "
            f"{summary.remote_gates} are remote"
        )

    actual = _count_cross_qpu_operations(mapped, arch)
    if summary.remote_gates != actual:
        problems.append(
            f"accounts for {summary.remote_gates} remote gates, the circuit has "
            f"{actual} operations spanning more than one QPU"
        )

    if plan is not None:
        if summary.blocks != len(plan.blocks):
            problems.append(
                f"reports {summary.blocks} blocks, the plan has {len(plan.blocks)}"
            )
        if summary.epr_pairs != plan.epr_pairs:
            problems.append(
                f"reports {summary.epr_pairs} EPR pairs, the plan spends "
                f"{plan.epr_pairs}"
            )

    return tuple(problems)


def _normalized_port_budget(
    ports_per_qpu: int | Sequence[int] | None, arch: MultiQPUArchitecture
) -> list[int]:
    """Resolve the comm-port budget to one non-negative integer per QPU."""
    n_qpus = arch.cfg.n_qpus
    if ports_per_qpu is None:
        return [
            _validated_nonnegative_int(
                arch.cfg.comm_qubits_per_qpu, label="comm_qubits_per_qpu"
            )
        ] * n_qpus
    if not isinstance(ports_per_qpu, bool) and isinstance(ports_per_qpu, Integral):
        return [
            _validated_nonnegative_int(ports_per_qpu, label="ports_per_qpu")
        ] * n_qpus
    if isinstance(ports_per_qpu, str | bytes | bytearray) or not isinstance(
        ports_per_qpu, Sequence
    ):
        raise ValueError("ports_per_qpu must be an integer or a sequence of integers")
    if len(ports_per_qpu) != n_qpus:
        raise ValueError("ports_per_qpu length must match n_qpus")
    return [
        _validated_nonnegative_int(value, label=f"ports_per_qpu[{index}]")
        for index, value in enumerate(ports_per_qpu)
    ]


def _peak_overlap(intervals: Sequence[tuple[float, float]]) -> int:
    """Return the largest number of intervals that overlap at one instant.

    A half-open convention is used: an interval that ends exactly when another
    begins does not count as concurrent, which matches a port being handed
    straight from one block to the next.
    """
    if not intervals:
        return 0
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return 0
    # Releases at a shared timestamp are applied before acquisitions.
    events.sort(key=lambda event: (event[0], event[1]))
    live = 0
    peak = 0
    for _time, delta in events:
        live += delta
        if live > peak:
            peak = live
    return peak
