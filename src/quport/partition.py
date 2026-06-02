# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, SupportsFloat, SupportsIndex, cast

from quport.interaction import cut_weight, degree
from quport.network import (
    UNREACHABLE_DISTANCE,
    QpuShortestPaths,
    compute_boundary_counts,
    compute_traffic_matrix,
    congestion_metrics,
    route_link_loads,
)


def _validate_and_normalize_partition_inputs(
    n: int,
    weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
) -> dict[tuple[int, int], float]:
    """Validate partitioning inputs and normalize undirected edge weights.

    Returns
    -------
    dict[(i, j), w]
        Deduplicated, canonicalized edge map with keys ordered as (min, max),
        excluding zero-weight and self-loop edges.
    """
    if type(n) is not int:
        raise ValueError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if type(n_qpus) is not int:
        raise ValueError("n_qpus must be an integer")
    if n_qpus <= 0:
        raise ValueError("n_qpus must be positive")
    if type(capacity) is not int:
        raise ValueError("capacity must be an integer")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if n > n_qpus * capacity:
        raise RuntimeError("Insufficient capacity")
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping of 2-tuples to numeric values")

    if n == 0:
        if weights:
            raise ValueError("weights must be empty when n == 0")
        return {}

    normalized: dict[tuple[int, int], float] = {}
    for edge, w_raw in weights.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("weights keys must be 2-tuples of logical indices")
        u, v = edge
        if type(u) is not int or type(v) is not int:
            raise ValueError("weights keys must contain integer logical indices")
        if u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError("weights contain out-of-range logical indices")
        if type(w_raw) is bool:
            raise ValueError("weights must be numeric values, not booleans")
        try:
            w = float(w_raw)
        except (TypeError, ValueError):
            raise ValueError("weights must be numeric") from None
        if not math.isfinite(w):
            raise ValueError("weights must be finite")
        if w < 0.0:
            raise ValueError("weights must be non-negative")
        if u == v or w == 0.0:
            continue
        a, b = (u, v) if u < v else (v, u)
        normalized[(a, b)] = normalized.get((a, b), 0.0) + w

    return normalized


def _build_weighted_adjacency(
    n: int,
    normalized_weights: Mapping[tuple[int, int], float],
) -> list[list[tuple[int, float]]]:
    """Build symmetric weighted adjacency lists from canonical edge weights."""
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), w in normalized_weights.items():
        nbrs[i].append((j, w))
        nbrs[j].append((i, w))
    return nbrs


def _coerce_finite_float(value: object, *, label: str) -> float:
    """Return a finite float for numeric runtime parameters."""
    if type(value) is bool:
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        out = float(cast(SupportsFloat | SupportsIndex | str, value))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return out


def _validate_positive_float(value: object, *, label: str) -> float:
    """Return a finite float that is strictly greater than zero."""
    out = _coerce_finite_float(value, label=label)
    if out <= 0.0:
        raise ValueError(f"{label} must be positive")
    return out


def _validate_nonnegative_float(value: object, *, label: str) -> float:
    """Return a finite float that is greater than or equal to zero."""
    out = _coerce_finite_float(value, label=label)
    if out < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _validate_probability(value: object, *, label: str) -> float:
    """Return a finite probability in the closed interval [0, 1]."""
    out = _coerce_finite_float(value, label=label)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return out


def _validate_sa_parameters(
    *,
    steps: int,
    temp0: float,
    temp_end: float,
    p_swap: float,
) -> tuple[int, float, float, float]:
    """Validate and normalize simulated-annealing controls for TPCCAP-SA."""
    if type(steps) is not int:
        raise ValueError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    return (
        steps,
        _validate_positive_float(temp0, label="temp0"),
        _validate_positive_float(temp_end, label="temp_end"),
        _validate_probability(p_swap, label="p_swap"),
    )


def _validate_tpccap_parameters(
    *,
    comm_ports_per_qpu: int,
    max_passes: int | None = None,
    max_candidate_qpus: int | None = None,
) -> None:
    """Validate TPCCAP control parameters."""
    if type(comm_ports_per_qpu) is not int:
        raise ValueError("comm_ports_per_qpu must be an integer")
    if comm_ports_per_qpu < 0:
        raise ValueError("comm_ports_per_qpu must be non-negative")

    if max_passes is not None:
        if type(max_passes) is not int:
            raise ValueError("max_passes must be an integer")
        if max_passes < 0:
            raise ValueError("max_passes must be non-negative")

    if max_candidate_qpus is not None:
        if type(max_candidate_qpus) is not int:
            raise ValueError("max_candidate_qpus must be an integer")
        if max_candidate_qpus <= 0:
            raise ValueError("max_candidate_qpus must be positive")


def _validate_tpccap_objective_parameters(
    *,
    w_dist: float,
    w_port: float,
    w_cong: float,
    congestion_routing: Literal["single_path", "ecmp"],
) -> tuple[float, float, float]:
    """Validate and normalize TPCCAP objective/routing parameters."""
    if congestion_routing not in ("single_path", "ecmp"):
        raise ValueError("congestion_routing must be 'single_path' or 'ecmp'")

    return (
        _validate_nonnegative_float(w_dist, label="w_dist"),
        _validate_nonnegative_float(w_port, label="w_port"),
        _validate_nonnegative_float(w_cong, label="w_cong"),
    )


def _validate_alpha_balance(alpha_balance: float) -> float:
    """Validate and normalize the greedy balance coefficient."""
    return _validate_nonnegative_float(alpha_balance, label="alpha_balance")


class DSU:
    """Disjoint Set Union (Union-Find) with size tracking."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        return self.union_roots(ra, rb)

    def union_roots(self, ra: int, rb: int) -> bool:
        """Union by precomputed roots (avoids duplicate find operations)."""
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def components(self) -> dict[int, list[int]]:
        comps: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            comps.setdefault(r, []).append(i)
        return comps


def heavy_edge_clustering_partition(
    n: int, weights: Mapping[tuple[int, int], float], n_qpus: int, capacity: int
) -> list[int]:
    """Baseline partitioner: heavy-edge clustering + first-fit decreasing bin packing.

    Parameters
    ----------
    weights:
        Logical interaction weights keyed by undirected qubit pairs. Values must
        be finite, non-boolean numerics and are treated as non-negative edge
        strengths.
    """
    normalized_weights = _validate_and_normalize_partition_inputs(
        n=n,
        weights=weights,
        n_qpus=n_qpus,
        capacity=capacity,
    )
    if n == 0:
        return []

    # Fast path: when one QPU can host everything, first-fit deterministically
    # places all qubits on QPU 0.
    if capacity >= n:
        return [0] * n

    # Fast path: when every logical qubit must remain a singleton cluster
    # (unit capacity) or edges cannot induce merges (empty normalized edge set),
    # first-fit placement is deterministic and equivalent to i // capacity.
    if capacity == 1 or not normalized_weights:
        return [i // capacity for i in range(n)]

    dsu = DSU(n)
    find = dsu.find
    size = dsu.size
    union_roots = dsu.union_roots
    get_weight = normalized_weights.__getitem__
    # Deterministic tie-breaks by edge endpoints avoid dependence on mapping
    # insertion order when multiple edges have equal weights.
    for u, v in sorted(
        normalized_weights,
        key=lambda e: (-get_weight(e), e[0], e[1]),
    ):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if size[ru] + size[rv] <= capacity:
            union_roots(ru, rv)

    clusters = list(dsu.components().values())
    clusters.sort(key=len, reverse=True)

    part = [-1] * n
    load = [0] * n_qpus

    for cluster in clusters:
        placed = False
        for q in range(n_qpus):
            if load[q] + len(cluster) <= capacity:
                for x in cluster:
                    part[x] = q
                load[q] += len(cluster)
                placed = True
                break
        if not placed:
            # Place individually
            q_cursor = 0
            for x in cluster:
                while q_cursor < n_qpus and load[q_cursor] >= capacity:
                    q_cursor += 1
                if q_cursor >= n_qpus:
                    raise RuntimeError("Insufficient capacity")
                part[x] = q_cursor
                load[q_cursor] += 1

    return part


@dataclass
class PartitionResult:
    part: list[int]
    cut: float
    loads: list[int]


def _balanced_greedy_partition_from_normalized(
    n: int,
    normalized_weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
    seed: int | None,
    alpha_balance: float,
) -> PartitionResult:
    """Balanced greedy partitioning core with pre-normalized weights."""
    if n == 0:
        return PartitionResult(part=[], cut=0.0, loads=[0] * n_qpus)

    nbrs = _build_weighted_adjacency(n, normalized_weights)
    deg = degree(normalized_weights, n)
    order = list(range(n))
    order.sort(key=lambda i: deg[i], reverse=True)

    part = [-1] * n
    loads = [0] * n_qpus
    inv_capacity = 1.0 / max(1, capacity)

    for v in order:
        best_q = None
        best_score = None

        for q in range(n_qpus):
            if loads[q] >= capacity:
                continue

            intra = 0.0
            for u, w in nbrs[v]:
                if part[u] == q:
                    intra += w

            load_frac = loads[q] * inv_capacity
            score = intra - alpha_balance * load_frac

            if best_score is None or score > best_score:
                best_score = score
                best_q = q
            elif score == best_score and best_q is not None:
                if loads[q] < loads[best_q]:
                    best_q = q

        if best_q is None:
            raise RuntimeError("No capacity to place qubit")

        part[v] = best_q
        loads[best_q] += 1

    part, loads = refine_local_moves(
        part=part,
        loads=loads,
        nbrs=nbrs,
        capacity=capacity,
        max_passes=8,
        seed=seed,
    )

    return PartitionResult(
        part=part, cut=cut_weight(normalized_weights, part), loads=loads
    )


def balanced_greedy_partition(
    n: int,
    weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
    seed: int | None = None,
    alpha_balance: float = 0.25,
) -> PartitionResult:
    """Research-grade heuristic multi-way partitioner.

    Objective (informal):
    - maximize intra-QPU interaction weight (equivalently minimize cut weight),
      while respecting per-QPU capacity

    Approach:
    1) Order qubits by weighted degree
    2) Greedily assign each qubit to the QPU that maximizes:
         score = (sum of weights to already-placed qubits in that QPU)
                 - alpha_balance * load_fraction
    3) Local refinement with greedy moves to reduce cut weight.

    This has no external deps (unlike METIS) but is strong enough for research baselines.
    """
    normalized_weights = _validate_and_normalize_partition_inputs(
        n=n,
        weights=weights,
        n_qpus=n_qpus,
        capacity=capacity,
    )
    alpha = _validate_alpha_balance(alpha_balance)
    return _balanced_greedy_partition_from_normalized(
        n=n,
        normalized_weights=normalized_weights,
        n_qpus=n_qpus,
        capacity=capacity,
        seed=seed,
        alpha_balance=alpha,
    )


def refine_local_moves(
    part: list[int],
    loads: list[int],
    nbrs: list[list[tuple[int, float]]],
    capacity: int,
    max_passes: int = 8,
    seed: int | None = None,
) -> tuple[list[int], list[int]]:
    """Greedy refinement: try moving a vertex to another QPU if it reduces cut."""
    rng = random.Random(seed)
    n = len(part)
    n_qpus = len(loads)

    def delta_move(v: int, new_q: int) -> float:
        """Cut change if v moved to new_q. Negative is good."""
        old_q = part[v]
        if old_q == new_q:
            return 0.0
        delta = 0.0
        for u, w in nbrs[v]:
            if part[u] == old_q:
                # edge (v,u) was internal, becomes cut
                delta += w
            elif part[u] == new_q:
                # edge was cut, becomes internal
                delta -= w
        return delta

    order = list(range(n))
    for _pass in range(max_passes):
        improved = False
        rng.shuffle(order)

        for v in order:
            old_q = part[v]
            best_q = old_q
            best_delta = 0.0

            for q in range(n_qpus):
                if q == old_q:
                    continue
                if loads[q] >= capacity:
                    continue
                d = delta_move(v, q)
                if d < best_delta:
                    best_delta = d
                    best_q = q

            if best_q != old_q:
                # apply move
                part[v] = best_q
                loads[old_q] -= 1
                loads[best_q] += 1
                improved = True

        if not improved:
            break

    return part, loads


def _empty_partition_result(n_qpus: int) -> PartitionResult:
    """Construct an empty partition result for zero-qubit inputs."""
    return PartitionResult(part=[], cut=0.0, loads=[0] * n_qpus)


def _zero_partition_diagnostics() -> PartitionDiagnostics:
    """Construct zero-valued TPCCAP diagnostics."""
    return PartitionDiagnostics(
        weighted_cut_distance=0.0,
        port_overflow_l2=0.0,
        congestion_l2=0.0,
        congestion_max=0.0,
    )


@dataclass(frozen=True)
class PartitionDiagnostics:
    """Extra diagnostics for advanced partitioners."""

    weighted_cut_distance: float
    port_overflow_l2: float
    congestion_l2: float
    congestion_max: float


def _remove_unroutable_traffic(
    traffic: list[list[float]], sp: QpuShortestPaths
) -> tuple[float, float]:
    """Remove disconnected-pair traffic and return virtual congestion penalties.

    ``route_link_loads`` is intentionally strict and raises when asked to route
    positive traffic between disconnected QPUs.  TPCCAP evaluates many tentative
    partitions, so an unroutable cut should be represented as a very expensive
    objective value rather than as an exception.

    Returns
    -------
    (max_penalty_load, l2_penalty_load)
        Penalties in the same load units as routed congestion metrics.  Each
        unreachable QPU pair contributes a virtual load of
        ``traffic[src][dst] * UNREACHABLE_DISTANCE``.  The L2 penalty is summed
        per unreachable pair, matching the way routed congestion sums squared
        link loads without adding artificial cross terms between independent
        QPU pairs.
    """
    max_penalty_load = 0.0
    l2_penalty_load = 0.0

    for src in range(len(traffic)):
        for dst in range(src + 1, len(traffic)):
            load = traffic[src][dst]
            if load <= 0.0 or sp.dist[src][dst] < UNREACHABLE_DISTANCE:
                continue

            penalty_load = load * float(UNREACHABLE_DISTANCE)
            max_penalty_load = max(max_penalty_load, penalty_load)
            l2_penalty_load += penalty_load * penalty_load
            traffic[src][dst] = 0.0
            traffic[dst][src] = 0.0

    return max_penalty_load, l2_penalty_load


def _objective_tpccap(
    weights: Mapping[tuple[int, int], float],
    part: list[int],
    n_qpus: int,
    comm_ports_per_qpu: int,
    sp: QpuShortestPaths,
    w_dist: float,
    w_port: float,
    w_cong: float,
    congestion_routing: Literal["single_path", "ecmp"],
) -> tuple[float, PartitionDiagnostics]:
    """Compute the TPCCAP objective and diagnostics.

    Objective (minimize):
        w_dist * sum_{cut edges} weight * dist(qpu_i, qpu_j)
      + w_port * sum_q max(0, boundary_q - comm_ports)^2
      + w_cong * sum_{links} load(link)^2

    Notes
    -----
    - dist term makes the partitioner interconnect-aware.
    - boundary/port term approximates comm-qubit scarcity.
    - congestion term approximates bottlenecks on limited-degree fabrics.
    """
    # Weighted cut distance
    wcd = 0.0
    for (i, j), w in weights.items():
        qi, qj = part[i], part[j]
        if qi != qj:
            d = sp.dist[qi][qj]
            wcd += float(w) * float(d)

    # Port pressure (boundary unique counts)
    boundary = compute_boundary_counts(weights, part, n_qpus)
    port_overflow_l2 = 0.0
    for q in range(n_qpus):
        overflow = max(0, boundary[q] - comm_ports_per_qpu)
        port_overflow_l2 += float(overflow * overflow)

    # Congestion (route traffic along shortest paths).  Disconnected topologies
    # can make some candidate cuts unroutable; remove that traffic before
    # invoking strict shortest-path routing and add virtual high-cost loads to
    # the diagnostics/objective instead.
    traffic = compute_traffic_matrix(weights, part, n_qpus)
    unreachable_max, unreachable_l2 = _remove_unroutable_traffic(traffic, sp)
    loads = route_link_loads(traffic, sp, mode=congestion_routing)
    cong = congestion_metrics(loads)
    congestion_l2 = cong.l2_load + unreachable_l2
    congestion_max = max(cong.max_load, unreachable_max)

    obj = w_dist * wcd + w_port * port_overflow_l2 + w_cong * congestion_l2
    diag = PartitionDiagnostics(
        weighted_cut_distance=wcd,
        port_overflow_l2=port_overflow_l2,
        congestion_l2=congestion_l2,
        congestion_max=congestion_max,
    )
    return obj, diag


def _validate_sp_dimensions(sp: QpuShortestPaths, n_qpus: int) -> None:
    """Validate shortest-path tables for shape and basic metric consistency."""
    if len(sp.dist) != n_qpus or len(sp.next_hop) != n_qpus:
        raise ValueError("shortest-path dimensions do not match n_qpus")

    dist_rows: list[list[float]] = []
    for row in sp.dist:
        if len(row) != n_qpus:
            raise ValueError("shortest-path dimensions do not match n_qpus")
        out_row: list[float] = []
        for d in row:
            if type(d) is bool:
                raise ValueError("shortest-path distances must be numeric")
            try:
                d_float = float(d)
            except (TypeError, ValueError):
                raise ValueError("shortest-path distances must be numeric") from None
            if not math.isfinite(d_float):
                raise ValueError("shortest-path distances must be finite")
            if d_float < 0:
                raise ValueError("shortest-path distances must be non-negative")
            out_row.append(d_float)
        dist_rows.append(out_row)

    for i in range(n_qpus):
        if not math.isclose(dist_rows[i][i], 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("shortest-path distance diagonal must be zero")
        for j in range(i + 1, n_qpus):
            if not math.isclose(
                dist_rows[i][j], dist_rows[j][i], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("shortest-path distances must be symmetric")

    for i, row in enumerate(sp.next_hop):
        if len(row) != n_qpus:
            raise ValueError("shortest-path dimensions do not match n_qpus")
        for j, hop in enumerate(row):
            if type(hop) is not int:
                raise ValueError("shortest-path next_hop must contain integers")
            if hop < -1 or hop >= n_qpus:
                raise ValueError("shortest-path next_hop contains invalid indices")
            if i == j and hop not in (-1, i):
                raise ValueError("shortest-path next_hop diagonal must be -1 or self")


def _tpccap_partition_from_normalized(
    n: int,
    normalized_weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
    comm_ports_per_qpu: int,
    sp: QpuShortestPaths,
    seed: int | None,
    w_dist: float,
    w_port: float,
    w_cong: float,
    max_passes: int,
    max_candidate_qpus: int,
    congestion_routing: Literal["single_path", "ecmp"],
) -> tuple[PartitionResult, PartitionDiagnostics]:
    """TPCCAP search core operating on validated, normalized weights."""
    rng = random.Random(seed)

    base = _balanced_greedy_partition_from_normalized(
        n=n,
        normalized_weights=normalized_weights,
        n_qpus=n_qpus,
        capacity=capacity,
        seed=seed,
        alpha_balance=0.25,
    )
    part = base.part[:]
    loads = base.loads[:]

    nbrs = _build_weighted_adjacency(n, normalized_weights)

    best_obj, best_diag = _objective_tpccap(
        weights=normalized_weights,
        part=part,
        n_qpus=n_qpus,
        comm_ports_per_qpu=comm_ports_per_qpu,
        sp=sp,
        w_dist=w_dist,
        w_port=w_port,
        w_cong=w_cong,
        congestion_routing=congestion_routing,
    )

    topk = min(max_candidate_qpus, n_qpus)

    def candidate_qpus(v: int) -> list[int]:
        aff = [0.0] * n_qpus
        for u, w in nbrs[v]:
            q = part[u]
            if q >= 0:
                aff[q] += w

        out = heapq.nlargest(topk, range(n_qpus), key=aff.__getitem__)
        cur = part[v]
        if cur not in out:
            out.append(cur)
        return out

    order = list(range(n))
    for _pass in range(max_passes):
        improved = False
        rng.shuffle(order)

        for v in order:
            old_q = part[v]
            cands = candidate_qpus(v)
            best_q = old_q
            best_here_obj = best_obj
            best_here_diag = best_diag

            for q in cands:
                if q == old_q:
                    continue
                if loads[q] >= capacity:
                    continue

                part[v] = q
                loads[old_q] -= 1
                loads[q] += 1

                obj, diag = _objective_tpccap(
                    weights=normalized_weights,
                    part=part,
                    n_qpus=n_qpus,
                    comm_ports_per_qpu=comm_ports_per_qpu,
                    sp=sp,
                    w_dist=w_dist,
                    w_port=w_port,
                    w_cong=w_cong,
                    congestion_routing=congestion_routing,
                )

                loads[q] -= 1
                loads[old_q] += 1
                part[v] = old_q

                if obj < best_here_obj - 1e-9:
                    best_here_obj = obj
                    best_here_diag = diag
                    best_q = q

            if best_q != old_q:
                part[v] = best_q
                loads[old_q] -= 1
                loads[best_q] += 1
                best_obj = best_here_obj
                best_diag = best_here_diag
                improved = True

        if not improved:
            break

    res = PartitionResult(
        part=part, cut=cut_weight(normalized_weights, part), loads=loads
    )
    return res, best_diag


def tpccap_partition(
    n: int,
    weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
    comm_ports_per_qpu: int,
    sp: QpuShortestPaths,
    seed: int | None = None,
    # Objective weights (research knobs)
    w_dist: float = 1.0,
    w_port: float = 5.0,
    w_cong: float = 0.05,
    # Search params
    max_passes: int = 6,
    max_candidate_qpus: int = 4,
    congestion_routing: Literal["single_path", "ecmp"] = "ecmp",
) -> tuple[PartitionResult, PartitionDiagnostics]:
    """Topology- and Port-Constrained Congestion-Aware Partitioning (TPCCAP).

    This is the *novel* partitioning algorithm in QuPort.

    Motivation
    ----------
    Existing DQC compilation frameworks often partition circuits to minimize
    nonlocal interactions (edge cuts), but practical multi-QPU architectures
    have two additional constraints:

    1) **Communication-qubit scarcity**: only a few comm/port qubits exist per QPU.
    2) **Interconnect bottlenecks**: limited-degree topologies create congestion.

    TPCCAP explicitly optimizes for these by:
    - minimizing weighted cut distance (favoring closer QPU pairs)
    - penalizing port overflow via boundary-qubit counts
    - penalizing routed traffic congestion via L2 link-load

    Parameters
    ----------
    n:
        Number of logical qubits.
    weights:
        Undirected interaction weights (logical qubit pairs -> weight).
    n_qpus:
        Number of QPUs.
    capacity:
        Per-QPU logical capacity (compute + comm).
    comm_ports_per_qpu:
        Number of comm qubits (ports) per QPU.
    sp:
        All-pairs shortest paths on the QPU-level interconnect.

    Returns
    -------
    (PartitionResult, PartitionDiagnostics)

    Notes
    -----
    This method is dependency-free. For extremely large circuits, consider
    coupling it with external hypergraph partitioners; for NISQ-scale studies
    (tens to low-hundreds of qubits), this is typically sufficient.
    """
    _validate_tpccap_parameters(
        comm_ports_per_qpu=comm_ports_per_qpu,
        max_passes=max_passes,
        max_candidate_qpus=max_candidate_qpus,
    )
    w_dist_value, w_port_value, w_cong_value = _validate_tpccap_objective_parameters(
        w_dist=w_dist,
        w_port=w_port,
        w_cong=w_cong,
        congestion_routing=congestion_routing,
    )
    _validate_sp_dimensions(sp, n_qpus)
    normalized_weights = _validate_and_normalize_partition_inputs(
        n=n,
        weights=weights,
        n_qpus=n_qpus,
        capacity=capacity,
    )
    if n == 0:
        return _empty_partition_result(n_qpus), _zero_partition_diagnostics()

    return _tpccap_partition_from_normalized(
        n=n,
        normalized_weights=normalized_weights,
        n_qpus=n_qpus,
        capacity=capacity,
        comm_ports_per_qpu=comm_ports_per_qpu,
        sp=sp,
        seed=seed,
        w_dist=w_dist_value,
        w_port=w_port_value,
        w_cong=w_cong_value,
        max_passes=max_passes,
        max_candidate_qpus=max_candidate_qpus,
        congestion_routing=congestion_routing,
    )


@dataclass(frozen=True)
class AnnealDiagnostics:
    """Diagnostics for the simulated-annealing refinement stage."""

    steps: int
    accepted: int
    improved: int
    best_objective: float


def tpccap_sa_partition(
    n: int,
    weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
    comm_ports_per_qpu: int,
    sp: QpuShortestPaths,
    seed: int | None = None,
    # annealing knobs
    steps: int = 2000,
    temp0: float = 1.0,
    temp_end: float = 0.02,
    p_swap: float = 0.25,
) -> tuple[PartitionResult, PartitionDiagnostics, AnnealDiagnostics]:
    """TPCCAP + simulated annealing refinement (TPCCAP-SA).

    Rationale
    ---------
    Greedy/local-search partitioners can get stuck in poor local minima, especially
    on random circuits where the interaction graph is noisy. Several DQC compilation
    frameworks use annealing-style refinement to improve placements under multi-objective
    cost functions.

    This function:
    1) Runs :func:`tpccap_partition` to get a good initial solution.
    2) Refines the assignment using simulated annealing over move and swap proposals.

    Objective
    ---------
    Uses the same objective as TPCCAP (distance + port overflow + congestion), with
    soft penalties for capacity overflow (kept very large so feasible solutions dominate).

    Notes
    -----
    - This is designed for research workloads where n_qpus is small (e.g., 10).
    - For speed, objective is recomputed each accepted move. This is still fast for
      typical random circuits (<= few thousand unique 2Q pairs).

    Returns
    -------
    (PartitionResult, PartitionDiagnostics, AnnealDiagnostics)
    """
    rng = random.Random(seed)
    steps, temp0, temp_end, p_swap = _validate_sa_parameters(
        steps=steps,
        temp0=temp0,
        temp_end=temp_end,
        p_swap=p_swap,
    )
    _validate_tpccap_parameters(comm_ports_per_qpu=comm_ports_per_qpu)
    _validate_sp_dimensions(sp, n_qpus)
    normalized_weights = _validate_and_normalize_partition_inputs(
        n=n,
        weights=weights,
        n_qpus=n_qpus,
        capacity=capacity,
    )
    if n == 0:
        return (
            _empty_partition_result(n_qpus),
            _zero_partition_diagnostics(),
            AnnealDiagnostics(steps=steps, accepted=0, improved=0, best_objective=0.0),
        )

    pres, _ = _tpccap_partition_from_normalized(
        n=n,
        normalized_weights=normalized_weights,
        n_qpus=n_qpus,
        capacity=capacity,
        comm_ports_per_qpu=comm_ports_per_qpu,
        sp=sp,
        seed=seed,
        w_dist=1.0,
        w_port=5.0,
        w_cong=0.05,
        max_passes=6,
        max_candidate_qpus=4,
        congestion_routing="ecmp",
    )

    part = list(pres.part)
    loads = list(pres.loads)

    def objective(part_: list[int]) -> tuple[float, PartitionDiagnostics]:
        return _objective_tpccap(
            weights=normalized_weights,
            part=part_,
            n_qpus=n_qpus,
            comm_ports_per_qpu=comm_ports_per_qpu,
            sp=sp,
            w_dist=1.0,
            w_port=5.0,
            w_cong=0.2,
            congestion_routing="ecmp",
        )

    cur_obj, best_diag = objective(part)
    best_part = list(part)
    best_obj = cur_obj

    accepted = 0
    improved = 0

    # Maintain per-QPU node lists with O(1) updates for swaps/moves.
    nodes_by_qpu: list[list[int]] = [[] for _ in range(n_qpus)]
    node_pos = [0] * n
    for i, q in enumerate(part):
        node_pos[i] = len(nodes_by_qpu[q])
        nodes_by_qpu[q].append(i)

    def move_node(node: int, old_q: int, new_q: int) -> None:
        """Move node between QPU buckets in O(1)."""
        if old_q == new_q:
            return

        idx = node_pos[node]
        last = nodes_by_qpu[old_q][-1]
        nodes_by_qpu[old_q][idx] = last
        node_pos[last] = idx
        nodes_by_qpu[old_q].pop()
        node_pos[node] = len(nodes_by_qpu[new_q])
        nodes_by_qpu[new_q].append(node)

        refresh_nonempty_qpu(old_q)
        refresh_nonempty_qpu(new_q)

    def swap_nodes(node_a: int, node_b: int, q_a: int, q_b: int) -> None:
        """Swap two nodes between QPU buckets in O(1)."""
        if q_a == q_b:
            return
        idx_a = node_pos[node_a]
        idx_b = node_pos[node_b]
        nodes_by_qpu[q_a][idx_a] = node_b
        nodes_by_qpu[q_b][idx_b] = node_a
        node_pos[node_a] = idx_b
        node_pos[node_b] = idx_a

    free_qpus: list[int] = []
    free_qpu_pos = [-1] * n_qpus
    nonempty_qpus: list[int] = []
    nonempty_qpu_pos = [-1] * n_qpus

    def free_qpu_add(q: int) -> None:
        if free_qpu_pos[q] >= 0:
            return
        free_qpu_pos[q] = len(free_qpus)
        free_qpus.append(q)

    def free_qpu_discard(q: int) -> None:
        pos = free_qpu_pos[q]
        if pos < 0:
            return
        last = free_qpus[-1]
        free_qpus[pos] = last
        free_qpu_pos[last] = pos
        free_qpus.pop()
        free_qpu_pos[q] = -1

    def refresh_free_qpu(q: int) -> None:
        if loads[q] < capacity:
            free_qpu_add(q)
        else:
            free_qpu_discard(q)

    def nonempty_qpu_add(q: int) -> None:
        if nonempty_qpu_pos[q] >= 0:
            return
        nonempty_qpu_pos[q] = len(nonempty_qpus)
        nonempty_qpus.append(q)

    def nonempty_qpu_discard(q: int) -> None:
        pos = nonempty_qpu_pos[q]
        if pos < 0:
            return
        last = nonempty_qpus[-1]
        nonempty_qpus[pos] = last
        nonempty_qpu_pos[last] = pos
        nonempty_qpus.pop()
        nonempty_qpu_pos[q] = -1

    def refresh_nonempty_qpu(q: int) -> None:
        if loads[q] > 0:
            nonempty_qpu_add(q)
        else:
            nonempty_qpu_discard(q)

    for q in range(n_qpus):
        refresh_free_qpu(q)
        refresh_nonempty_qpu(q)

    def pick_free_qpu_excluding(exclude: int) -> int | None:
        """Sample a free-capacity QPU excluding ``exclude`` without per-step allocations."""
        m = len(free_qpus)
        if m == 0:
            return None
        ex_pos = free_qpu_pos[exclude]
        available = m - (1 if ex_pos >= 0 else 0)
        if available <= 0:
            return None

        idx = rng.randrange(available)
        if ex_pos >= 0 and idx >= ex_pos:
            idx += 1
        return free_qpus[idx]

    for step in range(steps):
        # Exponential temperature schedule
        if steps <= 1:
            temp = temp_end
        else:
            frac = step / float(steps - 1)
            temp = temp0 * ((temp_end / temp0) ** frac)

        proposal_swap = rng.random() < p_swap
        v = -1
        u = -1
        q0 = -1
        q1 = -1
        old_q = -1
        new_q = -1
        if proposal_swap and len(nonempty_qpus) < 2:
            # Preserve swap-only mode semantics (e.g., p_swap == 1.0): when a
            # swap is requested but infeasible, skip the step rather than
            # silently falling back to a load-changing move.
            continue

        if proposal_swap:
            # pick two distinct non-empty QPUs without temporary allocations
            nonempty_count = len(nonempty_qpus)
            i0 = rng.randrange(nonempty_count)
            i1 = rng.randrange(nonempty_count - 1)
            if i1 >= i0:
                i1 += 1
            q0 = nonempty_qpus[i0]
            q1 = nonempty_qpus[i1]
            v = rng.choice(nodes_by_qpu[q0])
            u = rng.choice(nodes_by_qpu[q1])
            # swap assignments
            part[v], part[u] = part[u], part[v]
            # loads unchanged
        else:
            v = rng.randrange(n)
            old_q = part[v]
            # choose a new QPU with free capacity
            new_q_opt = pick_free_qpu_excluding(old_q)
            if new_q_opt is None:
                continue
            new_q = new_q_opt
            part[v] = new_q
            loads[old_q] -= 1
            loads[new_q] += 1

            refresh_free_qpu(old_q)
            refresh_free_qpu(new_q)

        new_obj, new_diag = objective(part)

        delta = new_obj - cur_obj
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12))

        if accept:
            accepted += 1
            cur_obj = new_obj
            if cur_obj < best_obj - 1e-9:
                best_obj = cur_obj
                best_part = list(part)
                best_diag = new_diag
                improved += 1
            if proposal_swap:
                swap_nodes(v, u, q0, q1)
            else:
                move_node(v, old_q, new_q)
        else:
            # revert
            if proposal_swap:
                part[v], part[u] = part[u], part[v]
            else:
                loads[new_q] -= 1
                loads[old_q] += 1
                part[v] = old_q
                refresh_free_qpu(old_q)
                refresh_free_qpu(new_q)

    # finalize
    part = best_part
    loads = [0] * n_qpus
    for x in part:
        loads[x] += 1
    res = PartitionResult(
        part=part, cut=cut_weight(normalized_weights, part), loads=loads
    )
    return (
        res,
        best_diag,
        AnnealDiagnostics(
            steps=steps, accepted=accepted, improved=improved, best_objective=best_obj
        ),
    )
