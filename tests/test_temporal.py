# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for time-varying qubit placement.

The load-bearing property is the reduction: with one window, or with the same
assignment in every window, the temporal cost must be *identically* the static
e-bit count. Without it a reported saving would be measured against a different
model, and any number this module produces would be meaningless.
"""

from __future__ import annotations

import itertools
import random

import pytest
from qiskit import QuantumCircuit

from quport.hypergraph import build_distributable_packets, ebit_cost
from quport.temporal import (
    TemporalPartition,
    TemporalWindow,
    optimize_temporal_partition,
    split_windows,
    static_temporal_partition,
    temporal_ebit_cost,
)


def random_circuit(rng, n, depth):
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for qubit in range(n):
            if rng.random() < 0.35:
                getattr(qc, rng.choice(["h", "x", "z", "s", "t", "sx"]))(qubit)
        for _ in range(max(1, n // 2)):
            a, b = rng.sample(range(n), 2)
            name = rng.choice(["cx", "cz", "cp", "swap", "rzz"])
            if name in ("cp", "rzz"):
                getattr(qc, name)(rng.uniform(0.0, 3.0), a, b)
            else:
                getattr(qc, name)(a, b)
    if n >= 3:
        qc.ccx(*rng.sample(range(n), 3))
    return qc


def _shifting_neighbourhood(repeats: int = 5) -> QuantumCircuit:
    """A qubit that talks to one group early and a different group late.

    An ``h`` before each gate closes the packet, so no single cat copy can span
    the phases and the control genuinely pays per gate on whichever side it is
    stranded. This is the shape time-varying placement exists for.
    """
    qc = QuantumCircuit(6)
    for _ in range(repeats):
        for target in (1, 2):
            qc.h(0)
            qc.cz(0, target)
    for _ in range(repeats):
        for target in (3, 4, 5):
            qc.h(0)
            qc.cz(0, target)
    return qc


# ---------------------------------------------------------------------------
# The reduction property
# ---------------------------------------------------------------------------


def test_holding_one_assignment_costs_exactly_the_static_ebit_count():
    """Randomised: the generalisation must add no cost of its own."""
    rng = random.Random(20260818)
    checked = 0

    for _ in range(150):
        n = rng.randint(2, 9)
        n_qpus = rng.randint(1, 4)
        decomposition = build_distributable_packets(
            random_circuit(rng, n, rng.randint(1, 5))
        )
        part = [rng.randrange(n_qpus) for _ in range(n)]
        static = ebit_cost(decomposition, part, n_qpus)

        for count in (1, 2, 3, 5):
            windows = split_windows(decomposition, count)
            cost = temporal_ebit_cost(
                decomposition, static_temporal_partition(part, windows), n_qpus
            )
            assert cost.total == static
            assert cost.moves == 0
            assert cost.migration_ebits == 0
            checked += 1

    assert checked >= 400


def test_a_single_window_is_the_static_model():
    decomposition = build_distributable_packets(_shifting_neighbourhood())
    windows = split_windows(decomposition, 1)
    part = [0, 0, 0, 1, 1, 1]

    assert len(windows) == 1
    cost = temporal_ebit_cost(
        decomposition, static_temporal_partition(part, windows), 2
    )
    assert cost.total == ebit_cost(decomposition, part, 2)


# ---------------------------------------------------------------------------
# Cost semantics
# ---------------------------------------------------------------------------


def test_a_moved_root_loses_its_cat_copies():
    """Teleporting a root invalidates every copy of it, so the copy is re-paid.

    Two `cz` from one control into one remote QPU share a copy and cost one
    e-bit. Move the control between them and the second gate needs a fresh copy,
    so the same two gates cost two -- plus the teleport itself.
    """
    qc = QuantumCircuit(3)
    qc.cz(0, 2)
    qc.cz(0, 2)
    decomposition = build_distributable_packets(qc)
    windows = (TemporalWindow(0, 1), TemporalWindow(1, 2))

    stationary = TemporalPartition(windows=windows, assignments=((0, 0, 1), (0, 0, 1)))
    assert temporal_ebit_cost(decomposition, stationary, 2).total == 1

    moved = TemporalPartition(windows=windows, assignments=((0, 0, 1), (1, 0, 1)))
    cost = temporal_ebit_cost(decomposition, moved, 2)
    assert cost.moves == 1
    assert cost.migration_ebits == 1
    # The first gate is remote, the second is local once the root has moved.
    assert cost.packet_ebits == 1
    assert cost.total == 2


def test_a_moved_partner_needs_a_second_copy():
    """A partner that migrates mid-packet is a second destination, not the same one."""
    qc = QuantumCircuit(4)
    qc.cz(0, 1)
    qc.cz(0, 1)
    decomposition = build_distributable_packets(qc)
    windows = (TemporalWindow(0, 1), TemporalWindow(1, 2))

    # Partner on QPU 1 throughout: one copy.
    settled = TemporalPartition(
        windows=windows, assignments=((0, 1, 0, 1), (0, 1, 0, 1))
    )
    assert temporal_ebit_cost(decomposition, settled, 3).packet_ebits == 1

    # Partner moves to QPU 2: the root now needs a copy there too.
    roaming = TemporalPartition(
        windows=windows, assignments=((0, 1, 0, 1), (0, 2, 0, 1))
    )
    cost = temporal_ebit_cost(decomposition, roaming, 3)
    assert cost.packet_ebits == 2
    assert cost.moves == 1


def test_migration_cost_is_configurable():
    decomposition = build_distributable_packets(_shifting_neighbourhood(2))
    windows = split_windows(decomposition, 2)
    partition = TemporalPartition(
        windows=windows,
        assignments=((0, 0, 0, 1, 1, 1), (1, 0, 0, 0, 1, 1)),
    )

    cheap = temporal_ebit_cost(decomposition, partition, 2, migration_cost=1)
    dear = temporal_ebit_cost(decomposition, partition, 2, migration_cost=4)

    assert cheap.moves == dear.moves == 2
    assert cheap.migration_ebits == 2
    assert dear.migration_ebits == 8
    assert dear.total - cheap.total == 6
    assert dear.packet_ebits == cheap.packet_ebits


def test_free_migration_never_costs_more_than_static():
    decomposition = build_distributable_packets(_shifting_neighbourhood())
    windows = split_windows(decomposition, 3)
    part = [0, 0, 0, 1, 1, 1]

    result = optimize_temporal_partition(
        decomposition, part, 2, 3, windows, migration_cost=0
    )
    assert result.cost.migration_ebits == 0
    assert result.cost.total <= result.static_cost


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_moving_a_qubit_pays_when_its_neighbourhood_shifts():
    decomposition = build_distributable_packets(_shifting_neighbourhood())
    windows = split_windows(decomposition, 2)
    part = [0, 0, 0, 1, 1, 1]

    result = optimize_temporal_partition(decomposition, part, 2, 3, windows, seed=0)

    assert result.static_cost == 15
    assert result.cost.total < result.static_cost
    assert result.cost.moves > 0
    assert result.reduction > 0.3
    # The control is the qubit whose neighbourhood shifts, so it is the one that
    # should move; capacity is full, so something must come back the other way.
    moved = {
        qubit for qubit, _boundary, _source, _target in result.partition.migrations()
    }
    assert 0 in moved
    assert len(moved) >= 2


def test_optimization_never_returns_worse_than_static():
    rng = random.Random(4)
    for _ in range(12):
        n = rng.randint(4, 8)
        n_qpus = rng.randint(2, 3)
        capacity = -(-n // n_qpus)
        decomposition = build_distributable_packets(random_circuit(rng, n, 4))
        part = []
        loads = [0] * n_qpus
        for qubit in range(n):
            choices = [q for q in range(n_qpus) if loads[q] < capacity]
            pick = rng.choice(choices)
            loads[pick] += 1
            part.append(pick)

        for count in (1, 2, 3):
            windows = split_windows(decomposition, count)
            result = optimize_temporal_partition(
                decomposition, part, n_qpus, capacity, windows, seed=rng.randrange(10)
            )
            assert result.cost.total <= result.static_cost
            assert result.saved >= 0
            # The reported cost must be the cost of the reported partition.
            assert (
                temporal_ebit_cost(decomposition, result.partition, n_qpus).total
                == result.cost.total
            )
            # Capacity holds in every window, not just the first.
            for assignment in result.partition.assignments:
                counts = [0] * n_qpus
                for qpu in assignment:
                    counts[qpu] += 1
                assert max(counts) <= capacity


def test_a_temporal_plan_never_loses_to_the_stationary_one():
    """The guarantee that makes a reported migration saving meaningful.

    The search's neighbourhood includes whole-circuit moves, so it improves the
    static placement too. Seeded naively, the larger temporal neighbourhood
    could settle in a worse basin and report a "time-varying saving" that in
    fact loses to plain re-placement. The temporal phase therefore starts from
    the best stationary placement, which orders the three costs by construction.
    """
    rng = random.Random(2024)
    for _ in range(10):
        n = rng.randint(5, 9)
        n_qpus = rng.randint(2, 3)
        capacity = -(-n // n_qpus)
        decomposition = build_distributable_packets(random_circuit(rng, n, 5))
        part = []
        loads = [0] * n_qpus
        for _qubit in range(n):
            pick = rng.choice([q for q in range(n_qpus) if loads[q] < capacity])
            loads[pick] += 1
            part.append(pick)

        for count in (1, 2, 4):
            result = optimize_temporal_partition(
                decomposition,
                part,
                n_qpus,
                capacity,
                split_windows(decomposition, count),
                seed=rng.randrange(100),
            )
            assert result.cost.total <= result.stationary_cost
            assert result.stationary_cost <= result.static_cost
            assert result.migration_saved >= 0
            assert result.migration_reduction >= 0.0


def test_one_window_makes_the_stationary_cost_the_whole_answer():
    decomposition = build_distributable_packets(_shifting_neighbourhood())
    windows = split_windows(decomposition, 1)

    result = optimize_temporal_partition(
        decomposition, [0, 0, 0, 1, 1, 1], 2, 3, windows, seed=0
    )
    assert result.cost.moves == 0
    assert result.cost.total == result.stationary_cost
    assert result.migration_saved == 0
    assert result.migration_reduction == 0.0


def test_an_expensive_teleport_leaves_only_the_stationary_gain():
    """Price migration out of reach and the plan collapses to re-placement."""
    decomposition = build_distributable_packets(_shifting_neighbourhood())
    windows = split_windows(decomposition, 3)

    result = optimize_temporal_partition(
        decomposition, [0, 0, 0, 1, 1, 1], 2, 3, windows, migration_cost=10_000, seed=0
    )
    assert result.cost.moves == 0
    assert result.cost.total == result.stationary_cost


def test_optimization_matches_brute_force_on_small_instances():
    """The search is judged against every feasible placement, not against itself."""
    rng = random.Random(777)
    checked = 0

    for _ in range(25):
        n = rng.randint(2, 4)
        n_qpus = rng.randint(1, 2)
        capacity = rng.randint(1, n)
        if n > n_qpus * capacity:
            continue
        decomposition = build_distributable_packets(random_circuit(rng, n, 2))
        windows = split_windows(decomposition, 2)

        feasible = []
        for candidate in itertools.product(range(n_qpus), repeat=n):
            loads = [0] * n_qpus
            ok = True
            for qpu in candidate:
                loads[qpu] += 1
                if loads[qpu] > capacity:
                    ok = False
                    break
            if ok:
                feasible.append(candidate)

        optimum = min(
            temporal_ebit_cost(
                decomposition,
                TemporalPartition(windows=tuple(windows), assignments=combo),
                n_qpus,
            ).total
            for combo in itertools.product(feasible, repeat=len(windows))
        )
        found = min(
            optimize_temporal_partition(
                decomposition, list(seed), n_qpus, capacity, windows, seed=0
            ).cost.total
            for seed in feasible
        )
        assert found == optimum
        checked += 1

    assert checked >= 10


def test_interval_moves_reach_what_single_window_moves_cannot():
    """A qubit worth relocating for several windows must be relocatable at once.

    Moving it for one window pays two migrations against one window of traffic
    and rarely wins, so a single-window neighbourhood stalls at the static seed.
    Cutting the same circuit into more windows must not make the result worse.
    """
    decomposition = build_distributable_packets(_shifting_neighbourhood(6))
    part = [0, 0, 0, 1, 1, 1]

    coarse = optimize_temporal_partition(
        decomposition, part, 2, 3, split_windows(decomposition, 2), seed=0
    )
    fine = optimize_temporal_partition(
        decomposition, part, 2, 3, split_windows(decomposition, 6), seed=0
    )

    assert coarse.reduction > 0.3
    assert fine.reduction > 0.3
    # Six windows still moves the control just once each way, not once per window.
    assert fine.cost.moves <= 4


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_split_windows_are_contiguous_and_start_at_zero():
    rng = random.Random(9)
    decomposition = build_distributable_packets(random_circuit(rng, 6, 5))

    for count in (1, 2, 3, 7, 40):
        windows = split_windows(decomposition, count)
        assert windows[0].start == 0
        assert len(windows) <= count
        for before, after in zip(windows, windows[1:]):
            assert before.stop == after.start
            assert before.stop > before.start


def test_split_windows_handles_a_circuit_with_no_communication():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.t(1)
    decomposition = build_distributable_packets(qc)

    windows = split_windows(decomposition, 4)
    assert len(windows) == 1
    cost = temporal_ebit_cost(
        decomposition, static_temporal_partition([0, 1, 1], windows), 2
    )
    assert cost.total == 0


def test_migrations_and_to_dict_describe_the_same_plan():
    windows = (TemporalWindow(0, 4), TemporalWindow(4, 9))
    partition = TemporalPartition(windows=windows, assignments=((0, 1, 0), (1, 1, 0)))

    assert partition.n_qubits == 3
    assert partition.migrations() == ((0, 0, 0, 1),)

    payload = partition.to_dict()
    assert payload["windows"] == [
        {"start": 0, "stop": 4},
        {"start": 4, "stop": 9},
    ]
    assert payload["assignments"] == [[0, 1, 0], [1, 1, 0]]
    assert payload["migrations"] == [
        {"qubit": 0, "boundary": 0, "from_qpu": 0, "to_qpu": 1}
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_windows_must_be_contiguous():
    decomposition = build_distributable_packets(_shifting_neighbourhood(2))
    partition = TemporalPartition(
        windows=(TemporalWindow(0, 4), TemporalWindow(6, 9)),
        assignments=((0,) * 6, (0,) * 6),
    )
    with pytest.raises(ValueError, match="contiguous"):
        temporal_ebit_cost(decomposition, partition, 2)


def test_the_first_window_must_start_at_zero():
    decomposition = build_distributable_packets(_shifting_neighbourhood(2))
    partition = TemporalPartition(
        windows=(TemporalWindow(3, 9),), assignments=((0,) * 6,)
    )
    with pytest.raises(ValueError, match="start at instruction 0"):
        temporal_ebit_cost(decomposition, partition, 2)


@pytest.mark.parametrize(
    "bounds, message",
    [
        ((-1, 4), "start must be non-negative"),
        ((4, 4), "stop must be greater than start"),
        ((5, 2), "stop must be greater than start"),
    ],
)
def test_window_bounds_are_validated(bounds, message):
    with pytest.raises(ValueError, match=message):
        TemporalWindow(*bounds)


def test_assignment_shape_is_validated():
    decomposition = build_distributable_packets(_shifting_neighbourhood(2))
    windows = (TemporalWindow(0, 20),)

    with pytest.raises(ValueError, match="one assignment per window"):
        temporal_ebit_cost(
            decomposition,
            TemporalPartition(windows=windows, assignments=((0,) * 6, (0,) * 6)),
            2,
        )
    with pytest.raises(ValueError, match="the circuit has 6"):
        temporal_ebit_cost(
            decomposition,
            TemporalPartition(windows=windows, assignments=((0, 0, 0),)),
            2,
        )
    with pytest.raises(ValueError, match="outside the valid QPU range"):
        temporal_ebit_cost(
            decomposition,
            TemporalPartition(windows=windows, assignments=((0, 0, 0, 0, 0, 9),)),
            2,
        )


def test_optimizer_rejects_an_infeasible_seed():
    decomposition = build_distributable_packets(_shifting_neighbourhood(2))
    windows = split_windows(decomposition, 2)

    with pytest.raises(ValueError, match="more than 2 qubits on QPU 0"):
        optimize_temporal_partition(decomposition, [0, 0, 0, 1, 1, 1], 2, 2, windows)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_qpus": 0}, "n_qpus must be positive"),
        ({"migration_cost": -1}, "migration_cost must be non-negative"),
        ({"migration_cost": 1.5}, "migration_cost must be an integer"),
    ],
)
def test_cost_parameters_are_validated(kwargs, message):
    decomposition = build_distributable_packets(_shifting_neighbourhood(2))
    windows = split_windows(decomposition, 1)
    partition = static_temporal_partition([0, 0, 0, 1, 1, 1], windows)

    call = {"n_qpus": 2}
    call.update(kwargs)
    n_qpus = call.pop("n_qpus")
    with pytest.raises(ValueError, match=message):
        temporal_ebit_cost(decomposition, partition, n_qpus, **call)


def test_result_reduction_is_zero_when_nothing_is_cut():
    qc = QuantumCircuit(4)
    qc.cz(0, 1)
    qc.cz(2, 3)
    decomposition = build_distributable_packets(qc)
    windows = split_windows(decomposition, 2)

    result = optimize_temporal_partition(
        decomposition, [0, 0, 1, 1], 2, 2, windows, seed=0
    )
    assert result.static_cost == 0
    assert result.cost.total == 0
    assert result.reduction == 0.0
    assert result.saved == 0
