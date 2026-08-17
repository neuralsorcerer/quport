# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for exact capacity-constrained partitioning.

The load-bearing tests here are the two that enumerate every feasible
assignment and compare. Branch and bound is only worth having if it provably
returns what exhaustive search returns, and the pruning, the canonical form and
the incremental cost updates are each an opportunity to lose an optimum
silently.
"""

from __future__ import annotations

import itertools
import random

import pytest
from qiskit import QuantumCircuit

from quport.exact import (
    ExactPartition,
    PartitionGap,
    optimal_partition,
    partition_gap,
)
from quport.hypergraph import (
    build_distributable_packets,
    ebit_cost,
    ebit_traffic_matrix,
)
from quport.interaction import cut_weight


def brute_force(n, n_qpus, capacity, score):
    """Minimum over every capacity-feasible assignment, by enumeration."""
    best = float("inf")
    for candidate in itertools.product(range(n_qpus), repeat=n):
        loads = [0] * n_qpus
        if any(
            (loads.__setitem__(q, loads[q] + 1) or loads[q]) > capacity
            for q in candidate
        ):
            continue
        best = min(best, score(list(candidate)))
    return best


def random_weights(rng, n, density):
    return {
        (i, j): rng.randint(1, 4)
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < density
    }


def random_circuit(rng, n, depth):
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.4:
                getattr(qc, rng.choice(["h", "x", "z", "s", "t", "sx"]))(q)
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


# --------------------------------------------------------------------------
# Agreement with exhaustive enumeration
# --------------------------------------------------------------------------


def test_cut_objective_matches_exhaustive_search():
    rng = random.Random(4242)
    checked = 0

    for _ in range(120):
        n = rng.randint(1, 7)
        n_qpus = rng.randint(1, 4)
        capacity = rng.randint(1, n)
        if n > n_qpus * capacity:
            continue

        weights = random_weights(rng, n, rng.choice([0.3, 0.7, 1.0]))
        result = optimal_partition(
            n, n_qpus, capacity, objective="cut", weights=weights
        )

        assert result.proved_optimal
        assert result.objective == brute_force(
            n, n_qpus, capacity, lambda p: cut_weight(weights, p)
        )
        # The reported cost must be the cost of the reported partition.
        assert cut_weight(weights, list(result.part)) == result.objective
        loads = [0] * n_qpus
        for qpu in result.part:
            loads[qpu] += 1
        assert max(loads) <= capacity
        checked += 1

    assert checked >= 60


def test_ebit_objective_matches_exhaustive_search():
    rng = random.Random(99)
    checked = 0

    for _ in range(60):
        n = rng.randint(2, 6)
        n_qpus = rng.randint(1, 3)
        capacity = rng.randint(1, n)
        if n > n_qpus * capacity:
            continue

        packets = build_distributable_packets(random_circuit(rng, n, rng.randint(1, 3)))
        result = optimal_partition(
            n, n_qpus, capacity, objective="ebits", packets=packets
        )

        assert result.proved_optimal
        assert result.objective == brute_force(
            n, n_qpus, capacity, lambda p: float(ebit_cost(packets, p, n_qpus))
        )
        assert float(ebit_cost(packets, list(result.part), n_qpus)) == result.objective
        checked += 1

    assert checked >= 30


def test_incremental_cost_is_rolled_back_exactly():
    """A stale incremental cost would show up as a wrong second answer."""
    rng = random.Random(7)
    qc = random_circuit(rng, 6, 3)
    packets = build_distributable_packets(qc)
    weights = random_weights(rng, 6, 0.8)

    for _ in range(3):
        assert (
            optimal_partition(6, 3, 2, objective="ebits", packets=packets).objective
            == optimal_partition(6, 3, 2, objective="ebits", packets=packets).objective
        )
        assert (
            optimal_partition(6, 3, 2, objective="cut", weights=weights).objective
            == optimal_partition(6, 3, 2, objective="cut", weights=weights).objective
        )


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------


def test_single_qpu_costs_nothing():
    weights = {(0, 1): 3.0, (1, 2): 2.0}
    result = optimal_partition(3, 1, 3, objective="cut", weights=weights)
    assert result.part == (0, 0, 0)
    assert result.objective == 0.0
    assert result.proved_optimal


def test_disconnected_components_are_separated():
    # Two triangles with no edge between them, and room for exactly one each.
    weights = {
        (0, 1): 5.0,
        (1, 2): 5.0,
        (0, 2): 5.0,
        (3, 4): 5.0,
        (4, 5): 5.0,
        (3, 5): 5.0,
    }
    result = optimal_partition(6, 2, 3, objective="cut", weights=weights)
    assert result.objective == 0.0
    assert result.part[0] == result.part[1] == result.part[2]
    assert result.part[3] == result.part[4] == result.part[5]
    assert result.part[0] != result.part[3]


def test_capacity_forces_a_cut():
    # A 4-clique that does not fit on one QPU must pay the 2x2 split: 4 edges.
    weights = {(i, j): 1.0 for i in range(4) for j in range(i + 1, 4)}
    result = optimal_partition(4, 2, 2, objective="cut", weights=weights)
    assert result.objective == 4.0


def test_aggregation_beats_cut_weight_on_a_fan_out():
    """Many gates from one control into one QPU are a single e-bit.

    A 6-qubit star cannot fit on one QPU of capacity 3, so the optimum has to
    cut it. The two objectives then disagree about the price of the same
    partition, which is the whole reason the e-bit model exists.
    """
    qc = QuantumCircuit(6)
    for target in range(1, 6):
        qc.cz(0, target)
    packets = build_distributable_packets(qc)
    weights = {(0, target): 1.0 for target in range(1, 6)}

    best_cut = optimal_partition(6, 2, 3, objective="cut", weights=weights)
    best_ebits = optimal_partition(6, 2, 3, objective="ebits", packets=packets)

    # Root plus two leaves on one QPU, three leaves on the other.
    assert best_cut.objective == 3.0
    # Those same three cut gates share one cat copy.
    assert best_ebits.objective == 1.0
    assert cut_weight(weights, list(best_ebits.part)) == 3.0


def test_empty_instance():
    result = optimal_partition(0, 2, 4, objective="cut", weights={})
    assert result == ExactPartition(
        part=(), objective=0.0, proved_optimal=True, nodes=0
    )


def test_exhausted_budget_still_returns_a_feasible_partition():
    rng = random.Random(11)
    weights = random_weights(rng, 12, 0.6)
    result = optimal_partition(12, 4, 3, objective="cut", weights=weights, max_nodes=5)

    assert not result.proved_optimal
    loads = [0] * 4
    for qpu in result.part:
        loads[qpu] += 1
    assert max(loads) <= 3
    assert cut_weight(weights, list(result.part)) == result.objective


def test_larger_budget_never_returns_a_worse_answer():
    rng = random.Random(12)
    weights = random_weights(rng, 10, 0.5)
    small = optimal_partition(10, 3, 4, objective="cut", weights=weights, max_nodes=20)
    full = optimal_partition(10, 3, 4, objective="cut", weights=weights)

    assert full.proved_optimal
    assert full.objective <= small.objective


def test_ebit_traffic_matrix_sums_to_twice_the_cost():
    """The fused traffic sweep must describe the plan the cost prices."""
    rng = random.Random(5)
    qc = random_circuit(rng, 8, 4)
    packets = build_distributable_packets(qc)
    part = [rng.randrange(3) for _ in range(8)]

    traffic = ebit_traffic_matrix(packets, part, 3)
    total = sum(sum(row) for row in traffic)

    assert total == 2.0 * ebit_cost(packets, part, 3)
    for i in range(3):
        assert traffic[i][i] == 0.0
        for j in range(3):
            assert traffic[i][j] == traffic[j][i]


# --------------------------------------------------------------------------
# Gap reporting
# --------------------------------------------------------------------------


def test_gap_is_zero_for_an_optimal_partition():
    weights = {(0, 1): 4.0, (2, 3): 4.0, (1, 2): 1.0}
    best = optimal_partition(4, 2, 2, objective="cut", weights=weights)
    gap = partition_gap(best.part, 2, 2, objective="cut", weights=weights)

    assert gap == PartitionGap(
        heuristic=best.objective, optimal=best.objective, proved_optimal=True
    )
    assert gap.absolute == 0.0
    assert gap.relative == 0.0


def test_gap_measures_a_bad_partition():
    weights = {(0, 1): 4.0, (2, 3): 4.0, (1, 2): 1.0}
    gap = partition_gap([0, 1, 0, 1], 2, 2, objective="cut", weights=weights)

    assert gap.optimal == 1.0
    assert gap.heuristic == 9.0
    assert gap.absolute == 8.0
    assert gap.relative == 8.0


def test_relative_gap_is_zero_when_the_optimum_is_zero():
    weights = {(0, 1): 2.0}
    gap = partition_gap([0, 0], 2, 2, objective="cut", weights=weights)
    assert gap.optimal == 0.0
    assert gap.relative == 0.0


def test_gap_rejects_a_partition_that_overfills_a_qpu():
    weights = {(0, 1): 1.0}
    with pytest.raises(ValueError, match="more than 1 qubits on QPU 0"):
        partition_gap([0, 0], 2, 1, objective="cut", weights=weights)


def test_gap_rejects_an_out_of_range_qpu():
    weights = {(0, 1): 1.0}
    with pytest.raises(ValueError, match="outside the valid QPU range"):
        partition_gap([0, 2], 2, 2, objective="cut", weights=weights)


def test_no_shipped_strategy_beats_the_proved_optimum():
    """A heuristic below the optimum means one of the two is wrong.

    This is a cross-check between two independent implementations of both
    objectives, so it catches an error in either.
    """
    from qiskit import transpile

    from quport import MultiQPUArchitecture, MultiQPUConfig
    from quport.interaction import extract_twoq_weights
    from quport.partition import (
        balanced_greedy_partition,
        heavy_edge_clustering_partition,
        tpccap_partition,
        tpccap_sa_partition,
    )
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=2,
        inter_topology="ring",
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    capacity = cfg.capacity_per_qpu()
    sp = arch.qpu_shortest_paths()

    for seed in range(3):
        qc = transpile(
            random_benchmark_circuit(n_logical=9, depth=8, seed=seed),
            basis_gates=list(cfg.basis_gates),
            optimization_level=0,
            seed_transpiler=seed,
        )
        weights = extract_twoq_weights(qc)
        packets = build_distributable_packets(qc)
        n = qc.num_qubits
        common = dict(
            n=n,
            weights=weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=cfg.comm_qubits_per_qpu,
            sp=sp,
            seed=seed,
        )

        parts = [
            balanced_greedy_partition(
                n=n, weights=weights, n_qpus=cfg.n_qpus, capacity=capacity, seed=seed
            ).part,
            heavy_edge_clustering_partition(
                n=n, weights=weights, n_qpus=cfg.n_qpus, capacity=capacity
            ),
            tpccap_partition(**common)[0].part,
            tpccap_sa_partition(**common)[0].part,
            tpccap_sa_partition(
                **common,
                w_dist=0.0,
                w_port=0.0,
                w_cong=0.05,
                anneal_w_cong=0.05,
                packets=packets,
                w_ebit=1.0,
                congestion_source="ebits",
            )[0].part,
        ]

        for part in parts:
            # partition_gap raises if the heuristic scores below the optimum.
            assert (
                partition_gap(
                    part, cfg.n_qpus, capacity, objective="cut", weights=weights
                ).absolute
                >= 0.0
            )
            assert (
                partition_gap(
                    part, cfg.n_qpus, capacity, objective="ebits", packets=packets
                ).absolute
                >= 0.0
            )


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(objective="both"), "objective must be"),
        (dict(objective="cut"), "requires weights"),
        (dict(objective="ebits"), "requires packets"),
    ],
)
def test_missing_inputs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        optimal_partition(4, 2, 2, **kwargs)


def test_packet_qubit_count_must_match():
    packets = build_distributable_packets(QuantumCircuit(3))
    with pytest.raises(ValueError, match="same number of qubits"):
        optimal_partition(4, 2, 2, objective="ebits", packets=packets)


@pytest.mark.parametrize(
    "args, message",
    [
        ((-1, 2, 2), "n must be non-negative"),
        ((4, 0, 2), "n_qpus must be positive"),
        ((4, 2, -1), "capacity must be non-negative"),
        ((True, 2, 2), "n must be an integer"),
    ],
)
def test_dimension_validation(args, message):
    with pytest.raises(ValueError, match=message):
        optimal_partition(*args, objective="cut", weights={})


def test_insufficient_capacity_is_rejected():
    with pytest.raises(RuntimeError, match="Insufficient capacity"):
        optimal_partition(5, 2, 2, objective="cut", weights={})


def test_max_nodes_must_be_positive():
    with pytest.raises(ValueError, match="max_nodes must be positive"):
        optimal_partition(4, 2, 2, objective="cut", weights={}, max_nodes=0)


def test_weight_validation_matches_cut_weight():
    with pytest.raises(ValueError, match="out-of-range"):
        optimal_partition(2, 2, 2, objective="cut", weights={(0, 5): 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        optimal_partition(2, 2, 2, objective="cut", weights={(0, 1): -1.0})
