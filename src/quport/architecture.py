# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import get_args

from qiskit.transpiler import CouplingMap

from quport.config import InterTopology, IntraTopology, MultiQPUConfig
from quport.network import QpuShortestPaths, all_pairs_shortest_paths, build_qpu_graph

_SUPPORTED_INTRA_TOPOLOGIES = frozenset(get_args(IntraTopology))
_SUPPORTED_INTER_TOPOLOGIES = frozenset(get_args(InterTopology))


def _coerce_integral(value: object, *, label: str) -> int:
    """Return an integer value, rejecting bools and non-integral objects."""
    if type(value) is bool or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _validate_nonnegative_int(value: object, *, label: str) -> int:
    """Return a non-negative integer."""
    out = _coerce_integral(value, label=label)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _validate_positive_int(value: object, *, label: str) -> int:
    """Return a strictly positive integer."""
    out = _coerce_integral(value, label=label)
    if out <= 0:
        raise ValueError(f"{label} must be positive")
    return out


def _validate_architecture_config(cfg: MultiQPUConfig) -> None:
    """Validate structural architecture settings before deriving indices/edges."""
    _validate_positive_int(cfg.n_qpus, label="n_qpus")
    _validate_nonnegative_int(
        cfg.compute_qubits_per_qpu, label="compute_qubits_per_qpu"
    )
    _validate_nonnegative_int(cfg.comm_qubits_per_qpu, label="comm_qubits_per_qpu")
    if (
        not isinstance(cfg.intra_topology, str)
        or cfg.intra_topology not in _SUPPORTED_INTRA_TOPOLOGIES
    ):
        raise ValueError(f"Unknown intra_topology: {cfg.intra_topology}")
    if (
        not isinstance(cfg.inter_topology, str)
        or cfg.inter_topology not in _SUPPORTED_INTER_TOPOLOGIES
    ):
        raise ValueError(f"Unknown inter_topology: {cfg.inter_topology}")


def _add_bidir(edges: list[tuple[int, int]], a: int, b: int) -> None:
    """Add a symmetric (bidirectional) connection to a directed CouplingMap."""
    if a == b:
        return
    edges.append((a, b))
    edges.append((b, a))


def _all_pairs(nodes: list[int]) -> Iterable[tuple[int, int]]:
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            yield nodes[i], nodes[j]


def _grid_dimensions(
    n: int, rows: object | None, cols: object | None
) -> tuple[int, int]:
    """Infer and validate row-major grid dimensions for ``n`` qubits."""
    if n <= 0:
        return 0, 0

    if rows is None and cols is None:
        resolved_rows = max(1, int(math.sqrt(n)))
        resolved_cols = max(1, math.ceil(n / resolved_rows))
    elif rows is None:
        resolved_cols = _validate_positive_int(cols, label="grid_cols")
        resolved_rows = math.ceil(n / resolved_cols)
    elif cols is None:
        resolved_rows = _validate_positive_int(rows, label="grid_rows")
        resolved_cols = math.ceil(n / resolved_rows)
    else:
        resolved_rows = _validate_positive_int(rows, label="grid_rows")
        resolved_cols = _validate_positive_int(cols, label="grid_cols")
        if resolved_rows * resolved_cols < n:
            raise ValueError(
                "grid_rows * grid_cols must cover all local qubits "
                f"({resolved_rows} * {resolved_cols} < {n})"
            )

    return resolved_rows, resolved_cols


@dataclass(frozen=True)
class QPUBlock:
    compute: list[int]
    comm: list[int]


class MultiQPUArchitecture:
    """A multi-QPU architecture encoded as a global directed CouplingMap.

    Qiskit's CouplingMap is directed: each edge (u, v) indicates an allowed **direction**
    for 2Q operations (e.g., CNOT control->target). If your physical connectivity is
    symmetric, you must add **both** directions.

    This class builds:
    - intra-QPU connectivity among compute+comm qubits
    - inter-QPU connectivity among comm qubits only, based on a chosen topology
    """

    def __init__(self, cfg: MultiQPUConfig):
        _validate_architecture_config(cfg)
        self.cfg = cfg
        self.block = cfg.compute_qubits_per_qpu + cfg.comm_qubits_per_qpu
        self.n_phys = cfg.n_qpus * self.block

    def qpu_of_phys(self, p: int) -> int:
        phys = _coerce_integral(p, label="physical qubit index")
        if phys < 0 or phys >= self.n_phys:
            raise ValueError(f"physical qubit index out of range: {p}")
        if self.block <= 0:
            raise ValueError("physical qubit index out of range: no physical qubits")
        return phys // self.block

    def block_of_qpu(self, qpu_id: int) -> QPUBlock:
        qpu = _coerce_integral(qpu_id, label="qpu_id")
        if qpu < 0 or qpu >= self.cfg.n_qpus:
            raise ValueError("qpu_id out of range")
        base = qpu * self.block
        comp = list(range(base, base + self.cfg.compute_qubits_per_qpu))
        comm = list(range(base + self.cfg.compute_qubits_per_qpu, base + self.block))
        return QPUBlock(comp, comm)

    def all_blocks(self) -> list[QPUBlock]:
        return [self.block_of_qpu(i) for i in range(self.cfg.n_qpus)]

    def _intra_edges(self, block: QPUBlock) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        local = block.compute + block.comm

        if self.cfg.intra_topology == "clique":
            for a, b in _all_pairs(local):
                _add_bidir(edges, a, b)

        elif self.cfg.intra_topology == "line":
            for i in range(len(local) - 1):
                _add_bidir(edges, local[i], local[i + 1])

        elif self.cfg.intra_topology == "ring":
            for i in range(len(local) - 1):
                _add_bidir(edges, local[i], local[i + 1])
            if len(local) > 2:
                _add_bidir(edges, local[-1], local[0])

        elif self.cfg.intra_topology == "grid2d":
            n = len(local)
            rows, cols = _grid_dimensions(n, self.cfg.grid_rows, self.cfg.grid_cols)
            # place local qubits row-major in a grid
            for idx, qid in enumerate(local):
                r, c = divmod(idx, cols)
                # right neighbor: only connect within the same declared grid row.
                if c + 1 < cols and idx + 1 < len(local):
                    _add_bidir(edges, qid, local[idx + 1])
                # down neighbor
                down = idx + cols
                if r + 1 < rows and down < len(local):
                    _add_bidir(edges, qid, local[down])

        else:
            raise ValueError(f"Unknown intra_topology: {self.cfg.intra_topology}")

        return edges

    def _inter_edges(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        if self.cfg.comm_qubits_per_qpu <= 0:
            return edges

        comms = [self.block_of_qpu(q).comm for q in range(self.cfg.n_qpus)]

        topo = self.cfg.inter_topology
        if topo in ("switch", "mesh"):
            # all-to-all among comm qubits
            for a in range(self.cfg.n_qpus):
                for b in range(a + 1, self.cfg.n_qpus):
                    for qa in comms[a]:
                        for qb in comms[b]:
                            _add_bidir(edges, qa, qb)

        elif topo == "ring":
            for a in range(self.cfg.n_qpus):
                b = (a + 1) % self.cfg.n_qpus
                for qa in comms[a]:
                    for qb in comms[b]:
                        _add_bidir(edges, qa, qb)

        elif topo in ("degree_d", "fat_tree"):
            qg = build_qpu_graph(self.cfg)
            use_first_comm_only = topo == "fat_tree"
            for a, neighbors in enumerate(qg):
                for b in neighbors:
                    if a >= b:
                        continue
                    if use_first_comm_only:
                        # fat-tree model: one representative inter-QPU link per adjacency.
                        _add_bidir(edges, comms[a][0], comms[b][0])
                    else:
                        # degree_d model: all comm ports can connect across each QPU edge.
                        for qa in comms[a]:
                            for qb in comms[b]:
                                _add_bidir(edges, qa, qb)

        elif topo == "clos":
            # 2-level approximation using two comm ports per QPU:
            # - port0: full mesh within pods
            # - port1: full mesh across pods (spine)
            if self.cfg.comm_qubits_per_qpu < 2:
                # fallback to degree-2 ring if not enough ports
                for a in range(self.cfg.n_qpus):
                    b = (a + 1) % self.cfg.n_qpus
                    qa = comms[a][0]
                    qb = comms[b][0]
                    _add_bidir(edges, qa, qb)
            else:
                pod_size = max(2, math.ceil(math.sqrt(self.cfg.n_qpus)))
                pods: list[list[int]] = []
                qpus = list(range(self.cfg.n_qpus))
                for i in range(0, len(qpus), pod_size):
                    pods.append(qpus[i : i + pod_size])

                # within-pod using port0
                for pod in pods:
                    for i in range(len(pod)):
                        for j in range(i + 1, len(pod)):
                            a, b = pod[i], pod[j]
                            _add_bidir(edges, comms[a][0], comms[b][0])

                # across pods using port1 (spine): full mesh across all QPUs' port1
                for a in range(self.cfg.n_qpus):
                    for b in range(a + 1, self.cfg.n_qpus):
                        _add_bidir(edges, comms[a][1], comms[b][1])
        else:
            raise ValueError(f"Unknown inter_topology: {topo}")

        return edges

    def build_coupling_map(self) -> CouplingMap:
        edges: list[tuple[int, int]] = []
        for q in range(self.cfg.n_qpus):
            edges.extend(self._intra_edges(self.block_of_qpu(q)))
        edges.extend(self._inter_edges())

        cm = CouplingMap(edges)
        # CouplingMap only creates qubit nodes that appear in an edge.  Preserve
        # isolated physical qubits (for example a single-qubit QPU, or QPUs with
        # zero comm ports) so transpilation and metrics still see the full device.
        for p in range(self.n_phys):
            if p not in cm.physical_qubits:
                cm.add_physical_qubit(p)
        return cm

    # -------------------------
    # QPU-level network helpers
    # -------------------------

    def qpu_graph(self) -> list[list[int]]:
        """Return the inter-QPU adjacency list (undirected)."""
        return build_qpu_graph(self.cfg)

    def build_intra_coupling_map(self, qpu_id: int) -> CouplingMap:
        """Coupling map that contains **only intra-QPU edges** for a single QPU.

        This is useful for *distributed compilation* where routing is performed
        independently within each QPU block, and inter-QPU gates are implemented
        by teleportation-based remote-gate protocols (rather than SWAPping quantum
        state across QPUs).

        The returned coupling map is defined over the full physical index range
        ``0..n_phys-1``, but contains edges only within ``block_of_qpu(qpu_id)``.
        """
        cm = CouplingMap()
        blk = self.block_of_qpu(qpu_id)
        phys = list(range(self.n_phys))
        # Ensure nodes exist even if isolated (CouplingMap lazily adds nodes)
        for p in phys:
            if p not in cm.physical_qubits:
                cm.add_physical_qubit(p)
        for u, v in self._intra_edges(blk):
            cm.add_edge(u, v)
        return cm

    def build_interconnect_coupling_map(self) -> CouplingMap:
        """Coupling map with only inter-QPU edges between comm qubits.

        This is mostly for analysis; global transpilation can also use
        :func:`build_coupling_map`.
        """
        cm = CouplingMap()
        for q in range(self.cfg.n_qpus):
            blk = self.block_of_qpu(q)
            for p in blk.compute + blk.comm:
                cm.add_physical_qubit(p)
        for u, v in self._inter_edges():
            cm.add_edge(u, v)
        return cm

    def qpu_shortest_paths(self) -> QpuShortestPaths:
        """All-pairs shortest paths on the inter-QPU network."""
        return all_pairs_shortest_paths(self.qpu_graph())
