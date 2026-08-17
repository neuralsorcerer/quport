# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact capacity-constrained partitioning, for measuring the heuristics.

QuPort's partitioners are heuristics, and a heuristic without a reference is a
number without a scale: "TPCCAP-SA reduced e-bits by 12%" says nothing about how
much was left on the table. This module solves the same partitioning problems
exactly, by branch and bound, on instances small enough for that to terminate --
which is enough to calibrate the heuristics and to catch a heuristic that
returns something *better* than the proved optimum, since one of the two would
then be wrong.

Both objectives QuPort optimises are supported:

``"cut"``
    Total weight of interactions crossing a QPU boundary -- the classical
    objective, and what :func:`quport.interaction.cut_weight` reports.

``"ebits"``
    The lambda-1 e-bit count of :mod:`quport.hypergraph`: EPR pairs consumed
    once gates are aggregated onto shared cat copies.

Search
------
Qubits are assigned one at a time in decreasing degree, so the bound bites
early. Three things keep the tree small:

*Canonical form.* Both objectives are invariant under relabelling QPUs, and the
capacity is uniform, so a partition and any permutation of its QPU labels cost
the same. Only restricted-growth assignments are explored -- a qubit may join a
QPU already in use, or open the lowest-numbered unused one -- which collapses
``n_qpus**n`` candidates to set partitions of at most ``n_qpus`` blocks.

*Monotone bounds.* Every cost counted from the qubits assigned so far can only
grow as more are assigned, so it is an admissible lower bound and a node whose
partial cost already reaches the incumbent can be cut.

*A seeded incumbent.* The search starts from the first-fit partition in search
order, so pruning has something to prune against from the first node.

None of the three is a heuristic shortcut: the result is the proved optimum
unless ``max_nodes`` is exhausted, which the return value reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from quport.hypergraph import PacketDecomposition
from quport.interaction import WeightValue, _iter_validated_weights

__all__ = [
    "DEFAULT_MAX_NODES",
    "ExactPartition",
    "PartitionGap",
    "optimal_partition",
    "partition_gap",
]

Objective = Literal["cut", "ebits"]

#: Default ceiling on explored nodes, so a call cannot run unboundedly.
DEFAULT_MAX_NODES: int = 2_000_000


@dataclass(frozen=True)
class ExactPartition:
    """The best assignment found, and whether it was proved optimal.

    Attributes
    ----------
    part:
        Logical-qubit-to-QPU assignment, one entry per qubit.
    objective:
        Its cost under the chosen objective.
    proved_optimal:
        True when the search closed, i.e. every remaining branch was bounded
        away. False means ``max_nodes`` ran out first and ``objective`` is an
        upper bound on the optimum rather than the optimum.
    nodes:
        Nodes explored, useful for judging whether a larger budget would help.
    """

    part: tuple[int, ...]
    objective: float
    proved_optimal: bool
    nodes: int


@dataclass(frozen=True)
class PartitionGap:
    """How far a heuristic partition sits above the exact optimum.

    Attributes
    ----------
    heuristic:
        Cost of the supplied partition.
    optimal:
        Cost of the best partition the search found.
    proved_optimal:
        Whether ``optimal`` is the proved optimum. When false the search ran out
        of nodes, so the true optimum may be lower and ``absolute`` is a
        *lower* bound on what the heuristic leaves behind.
    """

    heuristic: float
    optimal: float
    proved_optimal: bool

    @property
    def absolute(self) -> float:
        """Cost the heuristic leaves on the table."""
        return self.heuristic - self.optimal

    @property
    def relative(self) -> float:
        """Excess as a fraction of the optimum; zero when the optimum is zero."""
        if self.optimal <= 0.0:
            return 0.0
        return self.absolute / self.optimal


class _IncrementalCost(Protocol):
    """A cost that can be extended one assignment at a time and rolled back.

    ``assign`` returns only the cost that the new assignment *settles*, so the
    running total counts nothing that a deeper assignment could take back. That
    is what makes the running total an admissible bound.
    """

    def assign(self, qubit: int, qpu: int) -> float:
        """Assign ``qubit`` to ``qpu`` and return the cost this adds."""

    def undo(self, qubit: int, qpu: int) -> None:
        """Reverse the most recent :meth:`assign` for ``qubit``."""


class _CutCost:
    """Weight of interactions already decided to cross a boundary."""

    __slots__ = ("_neighbours", "_assigned")

    def __init__(self, n: int, edges: Sequence[tuple[int, int, float]]) -> None:
        neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i, j, weight in edges:
            neighbours[i].append((j, weight))
            neighbours[j].append((i, weight))
        self._neighbours = neighbours
        self._assigned = [-1] * n

    def assign(self, qubit: int, qpu: int) -> float:
        delta = 0.0
        assigned = self._assigned
        for other, weight in self._neighbours[qubit]:
            placed = assigned[other]
            if placed >= 0 and placed != qpu:
                delta += weight
        assigned[qubit] = qpu
        return delta

    def undo(self, qubit: int, qpu: int) -> None:
        self._assigned[qubit] = -1


class _EbitCost:
    """E-bits already forced by the qubits assigned so far.

    Each packet tracks, as a bitmask over QPUs, which QPUs its assigned partners
    occupy and which of those have already been charged. A packet whose root is
    still unassigned charges nothing, because which partners count as remote is
    not yet decided -- so the running total never over-counts, and because a
    charged QPU is never un-charged it can only rise.
    """

    __slots__ = (
        "_roots_of",
        "_partner_in",
        "_gates_of",
        "_root_qpu",
        "_counted",
        "_partners",
        "_gate_mask",
        "_undo",
    )

    def __init__(self, decomposition: PacketDecomposition) -> None:
        n = decomposition.n_qubits
        packets = decomposition.packets
        self._roots_of: list[list[int]] = [[] for _ in range(n)]
        self._partner_in: list[list[int]] = [[] for _ in range(n)]
        self._gates_of: list[list[int]] = [[] for _ in range(n)]

        for index, packet in enumerate(packets):
            self._roots_of[packet.root].append(index)
            for partner in packet.partners:
                self._partner_in[partner].append(index)
        for index, gate in enumerate(decomposition.unpackable_gates):
            for qubit in gate.qubits:
                self._gates_of[qubit].append(index)

        self._root_qpu = [-1] * len(packets)
        self._counted = [0] * len(packets)
        self._partners = [0] * len(packets)
        self._gate_mask = [0] * len(decomposition.unpackable_gates)
        # One saved-state frame per assigned qubit, popped on backtrack.
        self._undo: list[
            tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]
        ] = []

    def assign(self, qubit: int, qpu: int) -> float:
        bit = 1 << qpu
        delta = 0.0
        packet_frame: list[tuple[int, int, int, int]] = []
        gate_frame: list[tuple[int, int]] = []

        for index in self._roots_of[qubit]:
            packet_frame.append(
                (
                    index,
                    self._root_qpu[index],
                    self._counted[index],
                    self._partners[index],
                )
            )
            self._root_qpu[index] = qpu
            # Every QPU an assigned partner already occupies, other than the
            # root's own, is now settled as remote.
            counted = self._partners[index] & ~bit
            self._counted[index] = counted
            delta += float(counted.bit_count())

        for index in self._partner_in[qubit]:
            packet_frame.append(
                (
                    index,
                    self._root_qpu[index],
                    self._counted[index],
                    self._partners[index],
                )
            )
            self._partners[index] |= bit
            root = self._root_qpu[index]
            if root >= 0 and root != qpu and not self._counted[index] & bit:
                self._counted[index] |= bit
                delta += 1.0

        for index in self._gates_of[qubit]:
            mask = self._gate_mask[index]
            gate_frame.append((index, mask))
            if not mask & bit:
                # Every QPU beyond the first costs a teleport there and back.
                if mask:
                    delta += 2.0
                self._gate_mask[index] = mask | bit

        self._undo.append((packet_frame, gate_frame))
        return delta

    def undo(self, qubit: int, qpu: int) -> None:
        packet_frame, gate_frame = self._undo.pop()
        for index, root, counted, partners in reversed(packet_frame):
            self._root_qpu[index] = root
            self._counted[index] = counted
            self._partners[index] = partners
        for index, mask in reversed(gate_frame):
            self._gate_mask[index] = mask


def _validate_positive_int(value: object, *, label: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _validate_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _edge_list(
    weights: Mapping[tuple[int, int], WeightValue] | None, n: int
) -> list[tuple[int, int, float]]:
    """Normalize interaction weights exactly as :func:`cut_weight` does.

    Sharing the validator is what makes the exact cut objective and the reported
    cut weight the same function of the same inputs, rather than two readings
    that happen to agree on the cases someone tested.
    """
    if weights is None:
        return []
    return list(_iter_validated_weights(weights, n))


def _search_order(
    n: int,
    edges: Sequence[tuple[int, int, float]],
    decomposition: PacketDecomposition | None,
) -> list[int]:
    """Qubits in decreasing influence on the objective, ties by index.

    Any fixed order gives a correct search; a heavy-first order makes the bound
    bite sooner, which is what keeps the tree small. Degrees come from whichever
    structure the objective is actually costed from, so the e-bit objective is
    not left ordering by index when no interaction weights were supplied.
    """
    degree = [0.0] * n
    for i, j, weight in edges:
        degree[i] += weight
        degree[j] += weight

    if not edges and decomposition is not None:
        for packet in decomposition.packets:
            degree[packet.root] += float(len(packet.partners))
            for partner in packet.partners:
                degree[partner] += 1.0
        for gate in decomposition.unpackable_gates:
            span = float(len(gate.qubits) - 1)
            for qubit in gate.qubits:
                degree[qubit] += span

    return sorted(range(n), key=lambda qubit: (-degree[qubit], qubit))


def _first_fit(order: Sequence[int], n_qpus: int, capacity: int) -> list[int]:
    """Fill QPUs in order, giving the search an incumbent to prune against.

    Feasible whenever ``len(order) <= n_qpus * capacity``, which the caller
    checks, and canonical by construction.
    """
    assignment = [0] * len(order)
    qpu = 0
    used = 0
    for qubit in order:
        if used == capacity:
            qpu += 1
            used = 0
        assignment[qubit] = qpu
        used += 1
    return assignment


def optimal_partition(
    n: int,
    n_qpus: int,
    capacity: int,
    *,
    objective: Objective = "cut",
    weights: Mapping[tuple[int, int], WeightValue] | None = None,
    packets: PacketDecomposition | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> ExactPartition:
    """Solve capacity-constrained partitioning exactly by branch and bound.

    Parameters
    ----------
    n:
        Number of logical qubits to place.
    n_qpus, capacity:
        QPU count and the uniform per-QPU capacity. Uniformity is what makes
        the QPU labels interchangeable and the canonical form valid.
    objective:
        ``"cut"`` minimises crossing interaction weight and needs ``weights``;
        ``"ebits"`` minimises the lambda-1 e-bit count and needs ``packets``.
    weights:
        Undirected interaction weights, as the partitioners take them. Also used
        for search ordering when minimising e-bits, where it is optional.
    packets:
        A :class:`~quport.hypergraph.PacketDecomposition` of the same circuit.
    max_nodes:
        Ceiling on explored nodes. Reaching it stops the search and clears
        ``proved_optimal``; the returned partition is still feasible and still
        the best one found.

    Returns
    -------
    ExactPartition

    Notes
    -----
    The tree is over set partitions of at most ``n_qpus`` blocks, so this is for
    calibration on small instances -- roughly a dozen qubits -- not for
    compiling. Use it to measure how much a heuristic leaves behind, then trust
    the heuristic at scale.
    """
    n_value = _validate_nonnegative_int(n, label="n")
    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    capacity_value = _validate_nonnegative_int(capacity, label="capacity")
    max_nodes_value = _validate_positive_int(max_nodes, label="max_nodes")

    if objective not in ("cut", "ebits"):
        raise ValueError("objective must be 'cut' or 'ebits'")
    if n_value > n_qpus_value * capacity_value:
        raise RuntimeError("Insufficient capacity")

    edges = _edge_list(weights, n_value)

    if objective == "cut":
        if weights is None:
            raise ValueError("objective 'cut' requires weights")
        cost: _IncrementalCost = _CutCost(n_value, edges)
    else:
        if packets is None:
            raise ValueError("objective 'ebits' requires packets")
        if packets.n_qubits != n_value:
            raise ValueError("packets must describe the same number of qubits as n")
        cost = _EbitCost(packets)

    if n_value == 0:
        return ExactPartition(part=(), objective=0.0, proved_optimal=True, nodes=0)

    order = _search_order(n_value, edges, packets)

    # Seed the incumbent with first-fit, so the very first node has something to
    # be bounded against and `best_part` is never empty however small the budget.
    best_part = _first_fit(order, n_qpus_value, capacity_value)
    best_cost = 0.0
    for qubit in order:
        best_cost += cost.assign(qubit, best_part[qubit])
    for qubit in reversed(order):
        cost.undo(qubit, best_part[qubit])

    assignment = [0] * n_value
    loads = [0] * n_qpus_value
    nodes = 0
    exhausted = False

    def descend(depth: int, running: float, blocks_used: int) -> None:
        nonlocal best_part, best_cost, nodes, exhausted

        if depth == n_value:
            if running < best_cost:
                best_cost = running
                best_part = list(assignment)
            return

        qubit = order[depth]
        # Canonical form: an existing block, or the first unused one. Both
        # objectives are label-invariant, so the rest are relabellings.
        limit = min(blocks_used + 1, n_qpus_value)
        for qpu in range(limit):
            if loads[qpu] >= capacity_value:
                continue

            nodes += 1
            if nodes > max_nodes_value:
                exhausted = True
                return

            delta = cost.assign(qubit, qpu)
            total = running + delta
            if total < best_cost:
                assignment[qubit] = qpu
                loads[qpu] += 1
                descend(depth + 1, total, max(blocks_used, qpu + 1))
                loads[qpu] -= 1
            cost.undo(qubit, qpu)

            if exhausted:
                return

    descend(0, 0.0, 0)

    return ExactPartition(
        part=tuple(best_part),
        objective=best_cost,
        proved_optimal=not exhausted,
        nodes=nodes,
    )


def _validate_feasible(part: Sequence[int], n_qpus: int, capacity: int) -> list[int]:
    """Check a heuristic partition against the same constraints the search obeys.

    Without this an infeasible input -- an overfull QPU, an out-of-range label --
    could score below the optimum and be reported as a heuristic that beat the
    proved optimum, when the real fault is that it solved a different problem.
    """
    assignments: list[int] = []
    loads = [0] * n_qpus
    for index, qpu in enumerate(part):
        if type(qpu) is bool or not isinstance(qpu, int):
            raise ValueError(f"part[{index}] must be an integer QPU index")
        if qpu < 0 or qpu >= n_qpus:
            raise ValueError(f"part[{index}] is outside the valid QPU range")
        loads[qpu] += 1
        if loads[qpu] > capacity:
            raise ValueError(f"part places more than {capacity} qubits on QPU {qpu}")
        assignments.append(qpu)
    return assignments


def partition_gap(
    part: Sequence[int],
    n_qpus: int,
    capacity: int,
    *,
    objective: Objective = "cut",
    weights: Mapping[tuple[int, int], WeightValue] | None = None,
    packets: PacketDecomposition | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> PartitionGap:
    """Score a heuristic partition against the exact optimum.

    Raises
    ------
    ValueError
        If ``part`` is infeasible for ``n_qpus`` and ``capacity``, or if it
        scores *below* the proved optimum -- which means one of the two is
        wrong, and is worth failing loudly over rather than reporting as a
        negative gap.
    """
    from quport.hypergraph import ebit_cost
    from quport.interaction import cut_weight

    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    capacity_value = _validate_nonnegative_int(capacity, label="capacity")
    assignments = _validate_feasible(part, n_qpus_value, capacity_value)

    exact = optimal_partition(
        len(assignments),
        n_qpus_value,
        capacity_value,
        objective=objective,
        weights=weights,
        packets=packets,
        max_nodes=max_nodes,
    )

    if objective == "cut":
        assert weights is not None  # guaranteed by optimal_partition
        heuristic = cut_weight(weights, assignments)
    else:
        assert packets is not None
        heuristic = float(ebit_cost(packets, assignments, n_qpus_value))

    if exact.proved_optimal and heuristic < exact.objective - 1e-9:
        raise ValueError(
            f"heuristic cost {heuristic} beats the proved optimum "
            f"{exact.objective}; one of the two is wrong"
        )

    return PartitionGap(
        heuristic=heuristic,
        optimal=exact.objective,
        proved_optimal=exact.proved_optimal,
    )
