# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""QPU-level interconnect modeling utilities.

This module provides a lightweight graph model of the *inter-QPU* network.

Why this exists
--------------
Qiskit's :class:`~qiskit.transpiler.CouplingMap` describes connectivity between
*physical qubits* (directed). For multi-QPU research, it is useful to also
reason about connectivity at the *QPU* level, for example:

- distance-aware partitioning (prefer cuts between closer QPUs)
- congestion estimation on limited-degree topologies
- comm-port pressure (how many boundary qubits compete for limited ports)

All routines here are dependency-free and designed for n_qpus ~ O(10..100).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Literal, SupportsFloat, SupportsIndex, cast, get_args

from quport.config import InterTopology, MultiQPUConfig

QpuEdge = tuple[int, int]  # undirected edge (min, max)

# Finite sentinel used by all-pairs shortest paths for disconnected QPU pairs.
UNREACHABLE_DISTANCE = 10**9

_SUPPORTED_INTER_TOPOLOGIES = frozenset(get_args(InterTopology))


def _is_nonstring_sequence(value: object) -> bool:
    """Return True when value is a sequence suitable for index-based rows."""
    # Exclude string-like/buffer containers that are technically sequences but
    # semantically not valid adjacency/next-hop rows.
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray | memoryview
    )


def _as_nonstring_sequence(value: object, *, label: str) -> Sequence[object]:
    """Validate and return a sequence while excluding string-like containers."""
    if not _is_nonstring_sequence(value):
        raise ValueError(f"{label} must be a sequence of rows")
    return cast(Sequence[object], value)


def _coerce_int(value: object, *, label: str) -> int:
    """Coerce runtime config values to ints with strict typing."""
    if type(value) is bool:
        raise ValueError(f"{label} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _require_nonnegative_int(value: object, *, label: str) -> int:
    """Coerce and require a non-negative integer."""
    out = _coerce_int(value, label=label)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _clamp_nonnegative_int(value: object, *, label: str) -> int:
    """Coerce to int and clamp negatives to zero."""
    return max(0, _coerce_int(value, label=label))


def build_qpu_graph(cfg: MultiQPUConfig) -> list[list[int]]:
    """Build an undirected adjacency list for the inter-QPU network."""
    n = _require_nonnegative_int(cfg.n_qpus, label="n_qpus")
    comm_qubits_per_qpu = _require_nonnegative_int(
        cfg.comm_qubits_per_qpu, label="comm_qubits_per_qpu"
    )
    adj_sets: list[set[int]] = [set() for _ in range(n)]

    def add_all_to_all() -> None:
        for u in range(n):
            for v in range(u + 1, n):
                add(u, v)

    def add(u: int, v: int) -> None:
        if u == v:
            return
        adj_sets[u].add(v)
        adj_sets[v].add(u)

    topo = cfg.inter_topology
    if not isinstance(topo, str) or topo not in _SUPPORTED_INTER_TOPOLOGIES:
        raise ValueError(f"Unknown inter_topology={topo}")

    if comm_qubits_per_qpu == 0 or n <= 1:
        return [sorted(neighbors) for neighbors in adj_sets]

    if topo in ("switch", "mesh"):
        # All-to-all at QPU level
        add_all_to_all()

    elif topo == "ring":
        for u in range(n):
            add(u, (u + 1) % n)

    elif topo == "degree_d":
        # Circulant ring-lattice approximation with bounded degree.
        #
        # For even target degrees, connect k neighbors on each side (2k total).
        # For odd target degrees on even n, add a diametric matching edge to
        # realize the extra +1 degree exactly.
        target_degree = _clamp_nonnegative_int(cfg.inter_degree, label="inter_degree")
        max_degree = n - 1
        d = min(target_degree, max_degree)

        # Fast path: requested degree saturates the graph to complete.
        if d == max_degree:
            add_all_to_all()
        elif d > 0:
            k = min(d // 2, (n - 1) // 2)
            for u in range(n):
                for step in range(1, k + 1):
                    add(u, (u + step) % n)

            # Odd regular ring-lattice is only exactly realizable when n is even.
            if d % 2 == 1 and n % 2 == 0:
                half = n // 2
                for u in range(half):
                    add(u, u + half)

    elif topo == "clos":
        # QPU-level abstraction of the Clos approximation used in architecture.py.
        # With at least two comm ports per QPU, port0 provides within-pod links and
        # port1 provides spine links across all QPUs, which is all-to-all at the
        # QPU level.  With only one comm port, architecture.py deliberately falls
        # back to a ring, so the analysis graph must do the same.
        if comm_qubits_per_qpu < 2:
            for u in range(n):
                add(u, (u + 1) % n)
        else:
            add_all_to_all()

    elif topo == "fat_tree":
        # Lightweight k-ary fat-tree style approximation at QPU level.
        #
        # We do not model switches as explicit nodes; instead we create a 3-tier graph:
        # - edge layer: groups of size g (pods) fully connect to an implicit ToR
        # - aggregation layer: ToRs connect to aggregation nodes
        # - core layer: aggregation nodes connect to cores
        #
        # The resulting QPU graph has multiple short paths and good path diversity,
        # capturing key scheduling/congestion behaviors.
        #
        # Parameters are derived from n; override by editing config if desired.
        g = max(2, round(math.sqrt(n)))
        pods = (n + g - 1) // g

        # Connect QPUs within each pod densely (models in-pod switching)
        for p in range(pods):
            start = p * g
            end = min(n, (p + 1) * g)
            for u in range(start, end):
                for v in range(u + 1, end):
                    add(u, v)

        # Add sparse inter-pod connectivity (models core/aggregation)
        reps = [min(p * g, n - 1) for p in range(pods)]
        for i in range(pods):
            add(reps[i], reps[(i + 1) % pods])
        if pods >= 4:
            for i in range(pods):
                add(reps[i], reps[(i + 2) % pods])

    # normalize ordering
    return [sorted(neighbors) for neighbors in adj_sets]


def _validate_partition_assignments(part: list[int], n_qpus: int) -> None:
    if n_qpus <= 0:
        raise ValueError("n_qpus must be positive")
    for idx, qpu in enumerate(part):
        if type(qpu) is not int:
            raise ValueError(
                f"partition assignment must be an integer at logical qubit {idx}: {qpu}"
            )
        if qpu < 0 or qpu >= n_qpus:
            raise ValueError(
                f"partition assignment out of range at logical qubit {idx}: {qpu}"
            )


def _iter_validated_positive_weight_edges(
    weights: Mapping[tuple[int, int], float],
    *,
    part_len: int,
) -> Iterator[tuple[int, int, float]]:
    """Yield validated positive non-self-loop logical edges.

    Emits edges as provided by ``weights`` (orientation preserved). The helper is
    shared by callers that either need orientation-preserving iteration or their
    own aggregation strategy.
    """
    if part_len <= 0:
        if weights:
            raise ValueError("partition cannot be empty when weights are provided")
        return

    for edge, w_raw in weights.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("weights keys must be 2-tuples of logical indices")
        i, j = edge
        if type(i) is not int or type(j) is not int:
            raise ValueError("weights keys must contain integer logical indices")
        if i < 0 or j < 0:
            raise ValueError("weights contain negative logical index")
        if i >= part_len or j >= part_len:
            raise ValueError("partition length is smaller than max logical index")
        if type(w_raw) is bool:
            raise ValueError("weights must be numeric values, not booleans")
        try:
            w = float(w_raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("weights must be numeric") from None
        if not math.isfinite(w):
            raise ValueError("weights must be finite")
        if w < 0:
            raise ValueError("weights must be non-negative")
        if w == 0.0 or i == j:
            continue
        yield i, j, w


@dataclass(frozen=True)
class QpuShortestPaths:
    """All-pairs shortest path result for an undirected unweighted QPU graph."""

    dist: list[list[int]]
    next_hop: list[list[int]]
    adj: list[list[int]] | None = None

    def path(self, src: int, dst: int) -> list[int]:
        """Return one shortest path from src to dst (inclusive)."""
        next_hop = self.next_hop
        if not _is_nonstring_sequence(next_hop):
            raise ValueError("shortest-path next_hop must be a sequence of rows")
        n = len(next_hop)
        if type(src) is not int or type(dst) is not int:
            raise ValueError("src and dst must be integer QPU indices")
        if src < 0 or dst < 0 or src >= n or dst >= n:
            raise ValueError("src and dst must be valid QPU indices")

        out = [src]
        cur = src
        for _ in range(n):
            if cur == dst:
                return out
            row = next_hop[cur]
            if not _is_nonstring_sequence(row):
                raise ValueError("shortest-path next_hop rows must be sequences")
            if dst >= len(row):
                raise ValueError("shortest-path next_hop dimensions are inconsistent")
            nxt = row[dst]
            if type(nxt) is not int:
                raise ValueError("shortest-path next_hop must contain integer indices")
            if nxt < 0:
                return []
            if nxt >= n:
                raise ValueError("shortest-path next_hop contains invalid indices")
            out.append(nxt)
            cur = nxt

        raise ValueError("shortest-path next_hop contains a cycle")


def all_pairs_shortest_paths(adj: list[list[int]]) -> QpuShortestPaths:
    """Compute all-pairs shortest paths using repeated BFS (unweighted)."""
    norm_adj = _normalize_undirected_adjacency(adj)
    n = len(norm_adj)
    inf = UNREACHABLE_DISTANCE
    dist = [[inf] * n for _ in range(n)]
    nxt = [[-1] * n for _ in range(n)]

    from collections import deque

    for s in range(n):
        dist_s = dist[s]
        nxt_s = nxt[s]
        dist_s[s] = 0
        nxt_s[s] = s
        q = deque([s])
        while q:
            u = q.popleft()
            dist_u = dist_s[u]
            first_hop = nxt_s[u]
            for v in norm_adj[u]:
                if dist_s[v] == inf:
                    dist_s[v] = dist_u + 1
                    nxt_s[v] = v if u == s else first_hop
                    q.append(v)

    return QpuShortestPaths(
        dist=dist, next_hop=nxt, adj=[list(row) for row in norm_adj]
    )


def _normalize_undirected_adjacency(
    adj: Sequence[Sequence[int]],
    *,
    label: str = "adjacency",
) -> list[tuple[int, ...]]:
    """Validate and normalize an undirected adjacency list.

    - enforces top-level sequence type
    - enforces row sequence type
    - enforces strict integer indices (rejects bool)
    - enforces index bounds
    - removes self loops and duplicates
    - enforces undirected symmetry
    """
    if not _is_nonstring_sequence(adj):
        raise ValueError(f"{label} must be a sequence of rows")

    n = len(adj)
    normalized: list[set[int]] = [set() for _ in range(n)]

    for u, row in enumerate(adj):
        if not _is_nonstring_sequence(row):
            raise ValueError(f"{label} rows must be sequences")
        for v in row:
            if type(v) is not int:
                raise ValueError(f"{label} must contain integer indices")
            if v < 0 or v >= n:
                raise ValueError(f"{label} contains invalid indices")
            if v == u:
                continue
            normalized[u].add(v)

    for u, nbrs in enumerate(normalized):
        for v in nbrs:
            if u not in normalized[v]:
                raise ValueError(f"{label} must be symmetric for undirected graphs")

    return [tuple(sorted(nbrs)) for nbrs in normalized]


def _validate_square_rows(
    rows: Sequence[object],
    n: int,
    *,
    label: str,
    dimensions_error: str,
) -> list[Sequence[object]]:
    """Validate that rows is an n x n matrix of non-string sequences."""
    if len(rows) != n:
        raise ValueError(dimensions_error)

    validated_rows: list[Sequence[object]] = []
    for row in rows:
        if not _is_nonstring_sequence(row):
            raise ValueError(f"{label} rows must be sequences")
        row_seq = cast(Sequence[object], row)
        if len(row_seq) != n:
            raise ValueError(dimensions_error)
        validated_rows.append(row_seq)
    return validated_rows


def _coerce_traffic_value(value: object) -> float:
    """Coerce a traffic-matrix entry to float with deterministic errors."""
    if type(value) is bool:
        raise ValueError("traffic matrix must contain numeric values, not booleans")
    if isinstance(value, str | bytes | bytearray | memoryview):
        raise ValueError("traffic matrix must contain numeric values")
    try:
        out = float(cast(SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("traffic matrix must contain numeric values") from None
    if not math.isfinite(out):
        raise ValueError("traffic matrix must contain finite values")
    return out


def compute_traffic_matrix(
    weights: Mapping[tuple[int, int], float], part: list[int], n_qpus: int
) -> list[list[float]]:
    """Compute symmetric QPU-to-QPU traffic matrix from a logical partition.

    traffic[a][b] is the total 2Q interaction weight between logical qubits
    assigned to QPU a and those assigned to QPU b (a != b). Diagonal is 0.
    """
    _validate_partition_assignments(part, n_qpus)
    traffic = [[0.0] * n_qpus for _ in range(n_qpus)]
    if not weights:
        return traffic

    # Stream validated edges directly into the matrix. Mirrored logical edges
    # ((i, j) and (j, i)) naturally accumulate to the same QPU pair, so an
    # intermediate normalized map is unnecessary.
    part_len = len(part)
    part_assignments = part
    for i, j, w in _iter_validated_positive_weight_edges(weights, part_len=part_len):
        a, b = part_assignments[i], part_assignments[j]
        if a == b:
            continue
        traffic_a = traffic[a]
        traffic_b = traffic[b]
        traffic_a[b] += w
        traffic_b[a] += w
    return traffic


def compute_boundary_counts(
    weights: Mapping[tuple[int, int], float], part: list[int], n_qpus: int
) -> list[int]:
    """Count boundary logical qubits per QPU.

    A logical qubit is a boundary qubit for its QPU if it has at least one
    interaction edge to a qubit in another QPU.
    """
    _validate_partition_assignments(part, n_qpus)
    if not weights:
        return [0] * n_qpus

    part_len = len(part)

    # Track boundary membership per logical qubit with O(1) checks and compact
    # storage (0/1 byte markers).
    is_boundary = bytearray(part_len)
    counts = [0] * n_qpus
    for i, j, _w in _iter_validated_positive_weight_edges(weights, part_len=part_len):
        a, b = part[i], part[j]
        if a == b:
            continue
        if not is_boundary[i]:
            is_boundary[i] = 1
            counts[a] += 1
        if not is_boundary[j]:
            is_boundary[j] = 1
            counts[b] += 1
    return counts


def _validate_ecmp_distance_matrix(
    dist: object,
    *,
    n: int,
) -> list[list[int]]:
    """Validate an ECMP shortest-path distance matrix."""
    dist_rows = _as_nonstring_sequence(dist, label="shortest-path")
    dist_matrix_raw = _validate_square_rows(
        dist_rows,
        n,
        label="shortest-path",
        dimensions_error="shortest-path dimensions do not match traffic matrix",
    )
    dist_matrix: list[list[int]] = []
    max_finite_distance = n - 1
    unreachable = UNREACHABLE_DISTANCE
    for row_idx, dist_row in enumerate(dist_matrix_raw):
        out_row: list[int] = []
        for col_idx, d in enumerate(dist_row):
            out_raw = _require_nonnegative_int(d, label="shortest-path distances")
            out = unreachable if out_raw >= unreachable else out_raw
            if row_idx == col_idx and out != 0:
                raise ValueError("shortest-path distances must have zero diagonal")
            if row_idx != col_idx and out == 0:
                raise ValueError(
                    "shortest-path distances must be positive off-diagonal"
                )
            if col_idx < row_idx:
                if out != dist_matrix[col_idx][row_idx]:
                    raise ValueError("shortest-path distances must be symmetric")
                # Validate the finite-distance bound exactly once per undirected
                # pair (on lower triangle) after symmetry is confirmed so error
                # ordering remains deterministic.
                if 0 < out < unreachable and out > max_finite_distance:
                    raise ValueError(
                        "shortest-path distances exceed maximum finite graph distance"
                    )
            out_row.append(out)
        dist_matrix.append(out_row)

    return dist_matrix


def route_link_loads(
    traffic: list[list[float]],
    sp: QpuShortestPaths,
    mode: Literal["single_path", "ecmp"] = "single_path",
) -> dict[QpuEdge, float]:
    """Route traffic along shortest paths and accumulate per-link loads.

    For each QPU pair (a,b), we route traffic[a][b] along one shortest path.
    Loads are accumulated on undirected edges.

    Returns
    -------
    loads: dict[(u,v) -> load]
        where (u,v) is sorted (min,max)
    """
    traffic_rows = _as_nonstring_sequence(traffic, label="traffic matrix")
    n = len(traffic_rows)
    loads: dict[QpuEdge, float] = {}

    traffic_matrix = _validate_square_rows(
        traffic_rows,
        n,
        label="traffic matrix",
        dimensions_error="traffic matrix must be square",
    )
    if mode not in ("single_path", "ecmp"):
        raise ValueError("routing mode must be 'single_path' or 'ecmp'")

    zero_tol = 1e-12
    coerce_traffic = _coerce_traffic_value

    def add_load(u: int, v: int, w: float) -> None:
        e = (u, v) if u < v else (v, u)
        loads[e] = loads.get(e, 0.0) + w

    def iter_routed_pairs() -> Iterator[tuple[int, int, float]]:
        for a in range(n):
            row_a = traffic_matrix[a]
            diag = coerce_traffic(row_a[a])
            if not math.isclose(diag, 0.0, rel_tol=0.0, abs_tol=zero_tol):
                raise ValueError("traffic matrix diagonal must be zero")
            for b in range(a + 1, n):
                w = coerce_traffic(row_a[b])
                w_back = coerce_traffic(traffic_matrix[b][a])
                if w < 0 or w_back < 0:
                    raise ValueError("traffic matrix must be non-negative")
                if not math.isclose(w, w_back, rel_tol=0.0, abs_tol=zero_tol):
                    raise ValueError("traffic matrix must be symmetric")
                if math.isclose(w, 0.0, rel_tol=0.0, abs_tol=zero_tol):
                    continue
                yield a, b, w

    if mode == "ecmp":
        dist_matrix = _validate_ecmp_distance_matrix(sp.dist, n=n)
        neighbors = _neighbors_from_distance_matrix(dist_matrix)
        # Lazily cache dist[:, dst] columns to avoid repeated O(n) rebuilds for
        # each pair that shares the same destination.
        dist_to_dst_cache: list[list[int] | None] = [None] * n
        if sp.adj is not None:
            if len(sp.adj) != n:
                raise ValueError(
                    "shortest-path adjacency dimensions do not match traffic matrix"
                )
            normalized_adj = _normalize_undirected_adjacency(
                sp.adj, label="shortest-path adjacency"
            )
            if normalized_adj != neighbors:
                raise ValueError(
                    "shortest-path adjacency must align with unit distances"
                )

        for a, b, w in iter_routed_pairs():
            dist_to_dst = dist_to_dst_cache[b]
            if dist_to_dst is None:
                dist_to_dst = [dist_row[b] for dist_row in dist_matrix]
                dist_to_dst_cache[b] = dist_to_dst
            _route_ecmp_pair(a, b, w, dist_matrix, dist_to_dst, neighbors, add_load)
        return loads

    hop_rows = _as_nonstring_sequence(sp.next_hop, label="shortest-path")
    hop_matrix = _validate_square_rows(
        hop_rows,
        n,
        label="shortest-path",
        dimensions_error="shortest-path dimensions do not match traffic matrix",
    )
    for hop_row in hop_matrix:
        for hop in hop_row:
            if type(hop) is not int:
                raise ValueError("shortest-path next_hop must contain integer indices")
            if hop < -1 or hop >= n:
                raise ValueError("shortest-path next_hop contains invalid indices")

    for a, b, w in iter_routed_pairs():
        cur = a
        steps = 0
        while cur != b:
            nxt = cast(int, hop_matrix[cur][b])
            if nxt < 0:
                raise ValueError(
                    f"no path between QPU {a} and {b} for traffic load {w}"
                )
            add_load(cur, nxt, w)
            cur = nxt
            steps += 1
            if steps > n:
                raise ValueError("shortest-path routing contains a cycle")

    return loads


def _neighbors_from_distance_matrix(
    dist: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    """Build neighbor lists from a validated shortest-path distance matrix."""
    return [
        tuple(v for v, d in enumerate(row) if v != u and d == 1)
        for u, row in enumerate(dist)
    ]


def _route_ecmp_pair(
    src: int,
    dst: int,
    weight: float,
    dist: Sequence[Sequence[int]],
    dist_to_dst: Sequence[int],
    neighbors: list[tuple[int, ...]],
    add_load: Callable[[int, int, float], None],
) -> None:
    """Split traffic equally across all shortest paths for a single pair."""
    dist_src = dist[src]
    n = len(dist)
    d = dist_src[dst]
    if d >= UNREACHABLE_DISTANCE:
        raise ValueError(
            f"no path between QPU {src} and {dst} for traffic load {weight}"
        )
    if d <= 0:
        return
    if d == 1:
        add_load(src, dst, weight)
        return

    # Build shortest-path DAG once (successors + predecessors) from reachable frontier.
    layers: list[list[int]] = [[] for _ in range(d + 1)]
    succ: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    seen = [False] * n
    layers[0].append(src)
    seen[src] = True
    for depth in range(d):
        for u in layers[depth]:
            dist_src_u = dist_src[u]
            for v in neighbors[u]:
                if dist_src[v] != dist_src_u + 1:
                    continue
                if dist_to_dst[v] + dist_src_u + 1 != d:
                    continue
                succ[u].append(v)
                pred[v].append(u)
                if not seen[v]:
                    seen[v] = True
                    layers[depth + 1].append(v)

    # Count number of shortest paths from src to each node in the DAG.
    sigma = [0] * n
    sigma[src] = 1
    for depth in range(d):
        for u in layers[depth]:
            su = sigma[u]
            if su == 0:
                continue
            for v in succ[u]:
                sigma[v] += su

    if sigma[dst] <= 0:
        raise ValueError(
            f"no path between QPU {src} and {dst} for traffic load {weight}"
        )

    # Backward flow accumulation. flow[v] is traffic entering v from src via shortest DAG.
    flow = [0.0] * n
    flow[dst] = weight
    for depth in range(d, 0, -1):
        for v in layers[depth]:
            fv = flow[v]
            if fv == 0.0:
                continue
            sv = sigma[v]
            if sv == 0:
                continue
            for u in pred[v]:
                share = fv * (sigma[u] / sv)
                if share:
                    add_load(u, v, share)
                    flow[u] += share

    if not math.isclose(flow[src], weight, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("ecmp flow conservation check failed")


@dataclass(frozen=True)
class CongestionMetrics:
    """Congestion metrics derived from routed link loads."""

    max_load: float
    l2_load: float


def congestion_metrics(loads: Mapping[QpuEdge, float]) -> CongestionMetrics:
    """Compute max and L2 link-load metrics with strict key/value validation.

    Edge keys may be provided in either orientation for undirected links.
    Opposite orientations of the same edge are aggregated.
    """
    if not isinstance(loads, Mapping):
        raise TypeError("link loads must be provided as a mapping")
    if not loads:
        return CongestionMetrics(max_load=0.0, l2_load=0.0)

    edge_totals: dict[QpuEdge, float] = {}
    mx = 0.0
    l2 = 0.0
    for edge, w_raw in loads.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("link-load keys must be 2-tuples of QPU indices")
        u, v = edge
        if type(u) is not int or type(v) is not int:
            raise ValueError("link-load keys must contain integer QPU indices")
        if u < 0 or v < 0:
            raise ValueError("link-load keys must contain non-negative QPU indices")
        if u == v:
            raise ValueError("link-load keys cannot be self-loops")

        if type(w_raw) is bool:
            raise ValueError("link loads must be numeric values, not booleans")
        try:
            w = float(w_raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("link loads must be numeric") from None
        if not math.isfinite(w):
            raise ValueError("link loads must be finite")
        if w < 0:
            raise ValueError("link loads must be non-negative")

        key = (u, v) if u < v else (v, u)
        prev = edge_totals.get(key, 0.0)
        total = prev + w
        edge_totals[key] = total

        l2 += total * total - prev * prev
        if total > mx:
            mx = total

    return CongestionMetrics(max_load=mx, l2_load=l2)


def path_edges(sp: QpuShortestPaths, src: int, dst: int) -> list[QpuEdge]:
    """Return the undirected edges along one shortest path from src to dst."""
    path = sp.path(src, dst)
    if len(path) < 2:
        return []
    out: list[QpuEdge] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        out.append((u, v) if u < v else (v, u))
    return out


@dataclass(frozen=True)
class TopologyMetrics:
    """Structural metrics for an inter-QPU topology.

    ``diameter`` and ``average_shortest_path`` ignore unreachable QPU pairs;
    ``unreachable_pairs`` records how many unordered pairs have no path.
    This keeps disconnected or zero-port designs analyzable without silently
    treating the finite reachable subgraph as globally connected.
    """

    n_qpus: int
    edges: int
    min_degree: int
    max_degree: int
    average_degree: float
    connected: bool
    components: int
    diameter: int
    average_shortest_path: float
    unreachable_pairs: int


def topology_metrics(adj: Sequence[Sequence[int]]) -> TopologyMetrics:
    """Compute validated structural metrics for an undirected QPU graph.

    The helper is useful for comparing candidate interconnects before running an
    expensive compile/sweep. It validates the same undirected adjacency contract
    used by routing helpers, so malformed custom graphs fail early with clear
    errors.
    """
    norm_adj = _normalize_undirected_adjacency(adj)
    n = len(norm_adj)
    if n == 0:
        return TopologyMetrics(
            n_qpus=0,
            edges=0,
            min_degree=0,
            max_degree=0,
            average_degree=0.0,
            connected=True,
            components=0,
            diameter=0,
            average_shortest_path=0.0,
            unreachable_pairs=0,
        )

    degrees = [len(row) for row in norm_adj]
    edges = sum(degrees) // 2
    components = 0
    assigned_component = [False] * n
    finite_pairs = 0
    distance_sum = 0
    diameter = 0
    unreachable_pairs = 0

    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue: deque[int] = deque([src])
        if not assigned_component[src]:
            components += 1
            assigned_component[src] = True

        while queue:
            u = queue.popleft()
            for v in norm_adj[u]:
                if dist[v] >= 0:
                    continue
                dist[v] = dist[u] + 1
                assigned_component[v] = True
                queue.append(v)

        for dst in range(src + 1, n):
            d = dist[dst]
            if d < 0:
                unreachable_pairs += 1
            else:
                finite_pairs += 1
                distance_sum += d
                diameter = max(diameter, d)

    return TopologyMetrics(
        n_qpus=n,
        edges=edges,
        min_degree=min(degrees),
        max_degree=max(degrees),
        average_degree=(sum(degrees) / n),
        connected=components <= 1,
        components=components,
        diameter=diameter,
        average_shortest_path=(distance_sum / finite_pairs if finite_pairs else 0.0),
        unreachable_pairs=unreachable_pairs,
    )
