# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time-varying qubit placement, and what it would save.

Every partitioner in QuPort places a logical qubit on one QPU for the whole
circuit. That is a real restriction, not just an implementation detail: a qubit
that interacts with one neighbourhood early and a different one late pays for
the mismatch on every gate of whichever half it is on the wrong side of. A
machine that can teleport a qubit between QPUs can instead *move* it once and
pay a single EPR pair.

This module models that trade-off exactly, under the same e-bit accounting as
:mod:`quport.hypergraph`:

* the instruction stream is cut into contiguous **windows**;
* each window gets its own assignment of logical qubits to QPUs;
* a qubit whose QPU differs between neighbouring windows is teleported there,
  costing one EPR pair;
* everything else is costed exactly as :func:`quport.hypergraph.ebit_cost` does.

The generalisation is faithful in the strong sense: with one window, or with the
same assignment in every window, the cost is *identically* the static e-bit
count. Time-varying placement can therefore only be reported as a saving when it
actually is one, and :func:`optimize_temporal_partition` is seeded with the
static assignment so it can never return something worse.

Cat copies across a window boundary
-----------------------------------
A packet is not cut at a window boundary. A cat copy stays valid as long as its
root stays put, so the cost is counted over **root epochs** -- maximal runs of a
packet's gates during which the root's QPU does not change. Within an epoch, one
e-bit is charged per distinct remote QPU the partners occupy *at the time their
own gates run*, so a partner that migrates mid-packet correctly costs a second
copy. Teleporting the root invalidates every copy of it, which is exactly what
starting a new epoch expresses.

Scope
-----
This is an analysis of what time-varying placement is worth on a given circuit.
``compile_distributed`` still emits a single static placement; the windows and
their assignments are a plan a scheduler could act on, and a number a designer
can use to decide whether teleport-based migration is worth building.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from quport.hypergraph import PacketDecomposition, ebit_cost

__all__ = [
    "TemporalCost",
    "TemporalPartition",
    "TemporalResult",
    "TemporalWindow",
    "optimize_temporal_partition",
    "split_windows",
    "static_temporal_partition",
    "temporal_ebit_cost",
]


@dataclass(frozen=True)
class TemporalWindow:
    """A half-open range of instruction indices, ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is bool or not isinstance(self.start, int):
            raise ValueError("window start must be an integer")
        if type(self.stop) is bool or not isinstance(self.stop, int):
            raise ValueError("window stop must be an integer")
        if self.start < 0:
            raise ValueError("window start must be non-negative")
        if self.stop <= self.start:
            raise ValueError("window stop must be greater than start")


@dataclass(frozen=True)
class TemporalPartition:
    """Where every logical qubit sits, window by window.

    Attributes
    ----------
    windows:
        Contiguous, non-overlapping instruction ranges in increasing order.
    assignments:
        One qubit-to-QPU assignment per window, same length as ``windows``.
    """

    windows: tuple[TemporalWindow, ...]
    assignments: tuple[tuple[int, ...], ...]

    @property
    def n_qubits(self) -> int:
        """Number of logical qubits the partition places."""
        return len(self.assignments[0]) if self.assignments else 0

    def migrations(self) -> tuple[tuple[int, int, int, int], ...]:
        """Every move, as ``(qubit, boundary, from_qpu, to_qpu)``.

        ``boundary`` is the index of the window a qubit leaves, so the move
        happens between windows ``boundary`` and ``boundary + 1``.
        """
        moves: list[tuple[int, int, int, int]] = []
        for boundary in range(len(self.assignments) - 1):
            before = self.assignments[boundary]
            after = self.assignments[boundary + 1]
            for qubit, (source, target) in enumerate(zip(before, after)):
                if source != target:
                    moves.append((qubit, boundary, source, target))
        return tuple(moves)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready representation of the placement plan."""
        return {
            "windows": [
                {"start": window.start, "stop": window.stop} for window in self.windows
            ],
            "assignments": [list(assignment) for assignment in self.assignments],
            "migrations": [
                {
                    "qubit": qubit,
                    "boundary": boundary,
                    "from_qpu": source,
                    "to_qpu": target,
                }
                for qubit, boundary, source, target in self.migrations()
            ],
        }


@dataclass(frozen=True)
class TemporalCost:
    """EPR pairs a temporal partition consumes, split by where they go.

    Attributes
    ----------
    packet_ebits:
        Cat copies for distributable packets, counted over root epochs.
    unpackable_ebits:
        Teleports for gates no single bipartite copy can serve.
    migration_ebits:
        Teleports that move a qubit between windows.
    moves:
        How many qubit migrations those teleports pay for.
    """

    packet_ebits: int
    unpackable_ebits: int
    migration_ebits: int
    moves: int

    @property
    def total(self) -> int:
        """Total EPR pairs."""
        return self.packet_ebits + self.unpackable_ebits + self.migration_ebits


@dataclass(frozen=True)
class TemporalResult:
    """What the search found, against the static placement it started from.

    Attributes
    ----------
    partition:
        The best temporal placement found.
    cost:
        Its cost.
    static_cost:
        Cost of holding the **seed** assignment for the whole circuit. The
        search starts there, so ``cost.total <= static_cost`` always.

        This is the seed's cost, not the best static placement. The
        neighbourhood includes the whole-circuit interval, so the search also
        re-places qubits for the entire circuit, and part of the improvement
        over ``static_cost`` can be better static placement rather than
        migration. To separate the two, run the same search with a single
        window -- migration is then impossible by construction -- and compare
        against that, which is what ``stationary_cost`` records.
    stationary_cost:
        Cost of the best placement the same search reaches with migration
        forbidden. The temporal phase starts from it, so
        ``cost.total <= stationary_cost <= static_cost`` always, and the
        difference between the first two is what migration actually bought.
    passes:
        Improvement passes performed before the search stalled, over both
        phases.
    """

    partition: TemporalPartition
    cost: TemporalCost
    static_cost: int
    stationary_cost: int
    passes: int

    @property
    def saved(self) -> int:
        """EPR pairs saved against the seed placement."""
        return self.static_cost - self.cost.total

    @property
    def reduction(self) -> float:
        """Fraction of the seed's cost saved; zero when that cost is zero.

        Measured against the seed, so it combines better static placement with
        whatever migration buys. Use :attr:`migration_reduction` for the latter
        alone.
        """
        if self.static_cost <= 0:
            return 0.0
        return self.saved / self.static_cost

    @property
    def migration_saved(self) -> int:
        """EPR pairs that letting qubits move saved, over re-placing them."""
        return self.stationary_cost - self.cost.total

    @property
    def migration_reduction(self) -> float:
        """Fraction of the best stationary cost that migration saved.

        This is the honest headline: it holds the placement search constant and
        varies only whether qubits may move.
        """
        if self.stationary_cost <= 0:
            return 0.0
        return self.migration_saved / self.stationary_cost


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


def _relevant_indices(decomposition: PacketDecomposition) -> list[int]:
    """Instruction indices that cost anything, in increasing order.

    Windows are cut on these rather than on raw instruction indices, because a
    boundary that falls inside a run of local gates divides no communication and
    only adds a place for qubits to move at no benefit.
    """
    indices: set[int] = set()
    for packet in decomposition.packets:
        indices.update(packet.gate_indices)
    for gate in decomposition.unpackable_gates:
        indices.add(gate.index)
    return sorted(indices)


def split_windows(
    decomposition: PacketDecomposition, count: int
) -> tuple[TemporalWindow, ...]:
    """Cut the instruction stream into ``count`` windows of similar traffic.

    Windows are balanced by the number of communication-relevant gates they
    contain, not by raw index range: a window holding only local gates offers
    nothing to place differently. Fewer windows are returned than asked for when
    the circuit does not have enough such gates to fill them.

    The last window is left open-ended -- its ``stop`` is one past the last
    relevant instruction -- so any trailing local gates fall inside it.
    """
    count_value = _validate_positive_int(count, label="count")
    if not isinstance(decomposition, PacketDecomposition):
        raise ValueError("decomposition must be a PacketDecomposition")

    indices = _relevant_indices(decomposition)
    if not indices:
        return (TemporalWindow(start=0, stop=1),)

    windows_wanted = min(count_value, len(indices))
    per_window = len(indices) / windows_wanted

    windows: list[TemporalWindow] = []
    start = 0
    for slot in range(windows_wanted):
        last = int(round((slot + 1) * per_window)) - 1
        last = min(max(last, slot), len(indices) - 1)
        stop = indices[last] + 1
        if stop <= start:
            continue
        windows.append(TemporalWindow(start=start, stop=stop))
        start = stop
    return tuple(windows)


def _window_lookup(windows: Sequence[TemporalWindow]) -> tuple[list[int], int]:
    """Validate window contiguity and return ``(starts, last_stop)`` for bisect."""
    if not windows:
        raise ValueError("windows must not be empty")
    if windows[0].start != 0:
        # Instructions before the first window would have no assignment to be
        # costed against, and a lookup for one would silently fall through to
        # the last window.
        raise ValueError("the first window must start at instruction 0")
    starts: list[int] = []
    expected = windows[0].start
    for index, window in enumerate(windows):
        if not isinstance(window, TemporalWindow):
            raise ValueError("windows must contain TemporalWindow instances")
        if window.start != expected:
            raise ValueError(
                f"windows must be contiguous: window {index} starts at "
                f"{window.start}, previous ends at {expected}"
            )
        starts.append(window.start)
        expected = window.stop
    return starts, expected


def _validate_temporal_partition(
    partition: TemporalPartition, *, n_qubits: int, n_qpus: int
) -> tuple[list[int], int]:
    if not isinstance(partition, TemporalPartition):
        raise ValueError("partition must be a TemporalPartition")
    if len(partition.assignments) != len(partition.windows):
        raise ValueError("partition needs one assignment per window")
    starts, last_stop = _window_lookup(partition.windows)

    for index, assignment in enumerate(partition.assignments):
        if len(assignment) != n_qubits:
            raise ValueError(
                f"assignment {index} places {len(assignment)} qubits, the circuit "
                f"has {n_qubits}"
            )
        for qubit, qpu in enumerate(assignment):
            if type(qpu) is bool or not isinstance(qpu, int):
                raise ValueError(
                    f"assignment {index} entry {qubit} must be an integer QPU index"
                )
            if qpu < 0 or qpu >= n_qpus:
                raise ValueError(
                    f"assignment {index} entry {qubit} is outside the valid QPU range"
                )
    return starts, last_stop


def _cost(
    decomposition: PacketDecomposition,
    assignments: Sequence[Sequence[int]],
    starts: Sequence[int],
    last_stop: int,
    migration_cost: int,
) -> TemporalCost:
    """Unvalidated cost evaluation, for the search loop."""
    limit = len(starts) - 1

    def window_of(index: int) -> int:
        if index >= last_stop:
            return limit
        return min(bisect_right(starts, index) - 1, limit)

    packet_ebits = 0
    for packet in decomposition.packets:
        root = packet.root
        epoch_root = -1
        charged: set[int] = set()
        for gate_index, partner in zip(packet.gate_indices, packet.gate_partners):
            window = window_of(gate_index)
            root_qpu = assignments[window][root]
            if root_qpu != epoch_root:
                # The root moved, so every copy of it died with the teleport.
                epoch_root = root_qpu
                charged = set()
            partner_qpu = assignments[window][partner]
            if partner_qpu != root_qpu and partner_qpu not in charged:
                charged.add(partner_qpu)
                packet_ebits += 1

    unpackable_ebits = 0
    for gate in decomposition.unpackable_gates:
        window = window_of(gate.index)
        assignment = assignments[window]
        spanned = {assignment[qubit] for qubit in gate.qubits}
        unpackable_ebits += 2 * (len(spanned) - 1)

    moves = 0
    for boundary in range(len(assignments) - 1):
        before = assignments[boundary]
        after = assignments[boundary + 1]
        for qubit, source in enumerate(before):
            if source != after[qubit]:
                moves += 1

    return TemporalCost(
        packet_ebits=packet_ebits,
        unpackable_ebits=unpackable_ebits,
        migration_ebits=moves * migration_cost,
        moves=moves,
    )


def temporal_ebit_cost(
    decomposition: PacketDecomposition,
    partition: TemporalPartition,
    n_qpus: int,
    *,
    migration_cost: int = 1,
) -> TemporalCost:
    """EPR pairs a time-varying placement consumes.

    Parameters
    ----------
    migration_cost:
        EPR pairs to teleport one qubit between QPUs. One by default, which is
        what a standard teleport costs; pass a larger value to price a fabric
        where moving state is dearer than sharing a cat copy.

    Notes
    -----
    With a single window, or with the same assignment in every window, this
    returns exactly :func:`quport.hypergraph.ebit_cost` for that assignment --
    the generalisation adds no cost of its own, which is what makes a reported
    saving meaningful.
    """
    if not isinstance(decomposition, PacketDecomposition):
        raise ValueError("decomposition must be a PacketDecomposition")
    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    migration = _validate_nonnegative_int(migration_cost, label="migration_cost")
    starts, last_stop = _validate_temporal_partition(
        partition, n_qubits=decomposition.n_qubits, n_qpus=n_qpus_value
    )
    return _cost(decomposition, partition.assignments, starts, last_stop, migration)


def static_temporal_partition(
    part: Sequence[int], windows: Sequence[TemporalWindow]
) -> TemporalPartition:
    """Hold one assignment for every window, i.e. today's static placement."""
    assignment = tuple(int(qpu) for qpu in part)
    if not windows:
        raise ValueError("windows must not be empty")
    return TemporalPartition(
        windows=tuple(windows),
        assignments=tuple(assignment for _ in windows),
    )


def optimize_temporal_partition(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
    capacity: int,
    windows: Sequence[TemporalWindow],
    *,
    migration_cost: int = 1,
    max_passes: int = 8,
    seed: int | None = None,
) -> TemporalResult:
    """Let qubits move between windows, if moving them pays for itself.

    Seeded with ``part`` held across every window -- which costs exactly the
    static e-bit count -- then improved by first-improvement local search.

    The neighbourhood
    -----------------
    Both changes act on a **contiguous run of windows** rather than on one
    window, which is the difference between a search that works and one that
    stalls. A qubit relocated for a single window pays two migrations, in and
    out, and can only earn them back from that one window's traffic. The change
    that actually pays is usually to move a qubit *once* and leave it for
    several windows: two migrations against the traffic of the whole run. A
    single-window neighbourhood can only reach that through a sequence of
    individually worse states, so it never does.

    *Interval moves.* Put one qubit on a different QPU for windows ``a..b``.
    ``a == b`` is the single-window move; ``a = 0, b = last`` re-places it for
    the whole circuit, so the static neighbourhood is included, not replaced.

    *Interval swaps.* Exchange two qubits over windows ``a..b``. Needed because
    capacity is normally sized to the circuit, leaving no free slot for a plain
    move to use.

    Only a qubit some window in the run actually touches is considered: moving
    an idle qubit changes no packet cost and can only add migrations.

    Because the whole-circuit interval is included, the search also improves the
    *static* placement it was seeded with, and a result can beat ``static_cost``
    with zero migrations. That is a real improvement and worth keeping, but it
    means ``reduction`` alone does not say what migration bought. Running the
    same search with one window makes migration impossible by construction and
    gives the control to compare against.

    Parameters
    ----------
    part:
        Static assignment to start from, e.g. from any QuPort partitioner.
    capacity:
        Qubits per QPU, enforced in every window.
    max_passes:
        Improvement sweeps before giving up. The search also stops early as soon
        as a sweep finds nothing.
    seed:
        Orders the candidates within a sweep. Results are deterministic given a
        seed; a different one explores a different local optimum.

    Returns
    -------
    TemporalResult
        Whose ``cost.total`` never exceeds ``static_cost``.

    Notes
    -----
    Each candidate is scored by re-evaluating the whole cost, which is linear in
    the circuit's communication gates. That is affordable at the sizes this is
    meant for and keeps the scoring identical to :func:`temporal_ebit_cost`, so
    the search cannot optimise a slightly different function from the one that
    reports the result.
    """
    if not isinstance(decomposition, PacketDecomposition):
        raise ValueError("decomposition must be a PacketDecomposition")
    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    capacity_value = _validate_nonnegative_int(capacity, label="capacity")
    migration = _validate_nonnegative_int(migration_cost, label="migration_cost")
    passes_allowed = _validate_nonnegative_int(max_passes, label="max_passes")

    n_qubits = decomposition.n_qubits
    seed_partition = static_temporal_partition(part, windows)
    starts, last_stop = _validate_temporal_partition(
        seed_partition, n_qubits=n_qubits, n_qpus=n_qpus_value
    )

    loads = [0] * n_qpus_value
    for qpu in seed_partition.assignments[0]:
        loads[qpu] += 1
        if loads[qpu] > capacity_value:
            raise ValueError(
                f"part places more than {capacity_value} qubits on QPU {qpu}"
            )

    static_cost = ebit_cost(decomposition, list(part), n_qpus_value)
    touched = _qubits_per_window(decomposition, starts, last_stop, len(windows))

    # Phase one: improve the placement without any migration, by searching a
    # single window spanning the whole stream. Phase two then starts from the
    # best stationary placement rather than from the seed, which is what makes
    # the temporal result provably no worse than the stationary one. Seeding it
    # with the raw input instead lets the larger neighbourhood settle in a worse
    # basin, and a "time-varying saving" that loses to plain re-placement is not
    # a saving at all.
    stationary_assignment, stationary_cost, stationary_passes = _descend(
        decomposition=decomposition,
        assignments=[list(seed_partition.assignments[0])],
        starts=[0],
        last_stop=last_stop,
        touched=[set().union(*touched) if touched else set()],
        loads=[list(loads)],
        n_qpus=n_qpus_value,
        capacity=capacity_value,
        migration=migration,
        max_passes=passes_allowed,
        seed=seed,
    )
    base = stationary_assignment[0]

    if len(windows) == 1:
        assignments = [list(base)]
        best = stationary_cost
        passes = stationary_passes
    else:
        window_loads = [0] * n_qpus_value
        for qpu in base:
            window_loads[qpu] += 1
        assignments, best, passes = _descend(
            decomposition=decomposition,
            assignments=[list(base) for _ in windows],
            starts=starts,
            last_stop=last_stop,
            touched=touched,
            loads=[list(window_loads) for _ in windows],
            n_qpus=n_qpus_value,
            capacity=capacity_value,
            migration=migration,
            max_passes=passes_allowed,
            seed=seed,
        )
        passes += stationary_passes

    partition = TemporalPartition(
        windows=tuple(windows),
        assignments=tuple(tuple(assignment) for assignment in assignments),
    )
    return TemporalResult(
        partition=partition,
        cost=best,
        static_cost=static_cost,
        stationary_cost=stationary_cost.total,
        passes=passes,
    )


def _descend(
    *,
    decomposition: PacketDecomposition,
    assignments: list[list[int]],
    starts: Sequence[int],
    last_stop: int,
    touched: Sequence[set[int]],
    loads: list[list[int]],
    n_qpus: int,
    capacity: int,
    migration: int,
    max_passes: int,
    seed: int | None,
) -> tuple[list[list[int]], TemporalCost, int]:
    """First-improvement descent over interval moves and interval swaps.

    Mutates ``assignments`` and ``loads`` in place and returns them alongside
    the cost of the local optimum reached and the sweeps it took. Used twice:
    once over a single window to settle the stationary placement, then over the
    real windows to look for migrations worth paying for.
    """
    n_qubits = len(assignments[0])
    n_windows = len(assignments)
    intervals = [
        (first, last) for first in range(n_windows) for last in range(first, n_windows)
    ]
    rng = random.Random(seed)
    best = _cost(decomposition, assignments, starts, last_stop, migration)
    passes = 0

    for _ in range(max_passes):
        passes += 1
        improved = False
        rng.shuffle(intervals)

        for first, last in intervals:
            span = range(first, last + 1)
            movable = sorted(set().union(*(touched[w] for w in span)))
            if not movable:
                continue
            rng.shuffle(movable)

            for qubit in movable:
                origins = [assignments[w][qubit] for w in span]
                for target in range(n_qpus):
                    if all(origin == target for origin in origins):
                        continue
                    if any(
                        assignments[w][qubit] != target and loads[w][target] >= capacity
                        for w in span
                    ):
                        continue
                    for w in span:
                        assignments[w][qubit] = target
                    candidate = _cost(
                        decomposition, assignments, starts, last_stop, migration
                    )
                    if candidate.total < best.total:
                        best = candidate
                        for offset, w in enumerate(span):
                            origin = origins[offset]
                            if origin != target:
                                loads[w][origin] -= 1
                                loads[w][target] += 1
                        improved = True
                        break
                    for offset, w in enumerate(span):
                        assignments[w][qubit] = origins[offset]

            for alpha in movable:
                for beta in range(n_qubits):
                    if beta == alpha:
                        continue
                    if all(assignments[w][alpha] == assignments[w][beta] for w in span):
                        continue
                    for w in span:
                        assignments[w][alpha], assignments[w][beta] = (
                            assignments[w][beta],
                            assignments[w][alpha],
                        )
                    candidate = _cost(
                        decomposition, assignments, starts, last_stop, migration
                    )
                    if candidate.total < best.total:
                        # A swap puts each qubit in the slot the other left, so
                        # every QPU's load is unchanged in every window.
                        best = candidate
                        improved = True
                        break
                    for w in span:
                        assignments[w][alpha], assignments[w][beta] = (
                            assignments[w][beta],
                            assignments[w][alpha],
                        )

        if not improved:
            break

    return assignments, best, passes


def _qubits_per_window(
    decomposition: PacketDecomposition,
    starts: Sequence[int],
    last_stop: int,
    n_windows: int,
) -> list[set[int]]:
    """Which qubits each window's own gates touch."""
    limit = len(starts) - 1
    touched: list[set[int]] = [set() for _ in range(n_windows)]

    def window_of(index: int) -> int:
        if index >= last_stop:
            return limit
        return min(bisect_right(starts, index) - 1, limit)

    for packet in decomposition.packets:
        for gate_index, partner in zip(packet.gate_indices, packet.gate_partners):
            window = window_of(gate_index)
            touched[window].add(packet.root)
            touched[window].add(partner)
    for gate in decomposition.unpackable_gates:
        window = window_of(gate.index)
        touched[window].update(gate.qubits)
    return touched
