# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="arg-type,list-item,dict-item,var-annotated"
# These tests intentionally pass malformed runtime values to verify validation errors.
from collections import OrderedDict
from typing import Any

import pytest

from quport.network import (
    UNREACHABLE_DISTANCE,
    QpuShortestPaths,
    all_pairs_shortest_paths,
)
from quport.partition import (
    balanced_greedy_partition,
    heavy_edge_clustering_partition,
    tpccap_partition,
    tpccap_sa_partition,
)


def _disconnected_two_qpu_paths() -> QpuShortestPaths:
    return QpuShortestPaths(
        dist=[[0, UNREACHABLE_DISTANCE], [UNREACHABLE_DISTANCE, 0]],
        next_hop=[[0, -1], [-1, 1]],
    )


def _run_tpccap_variant(
    variant: str,
    *,
    n: int,
    weights: dict[tuple[int, int], float],
    n_qpus: int,
    capacity: int,
    comm_ports_per_qpu: int,
    sp: QpuShortestPaths,
) -> tuple[list[int], float, float, float]:
    if variant == "tpccap":
        result, diag = tpccap_partition(
            n=n,
            weights=weights,
            n_qpus=n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=comm_ports_per_qpu,
            sp=sp,
        )
        return (
            result.part,
            result.cut,
            diag.weighted_cut_distance,
            diag.congestion_l2,
        )

    if variant == "tpccap_sa":
        result, diag, _anneal = tpccap_sa_partition(
            n=n,
            weights=weights,
            n_qpus=n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=comm_ports_per_qpu,
            sp=sp,
            steps=2,
            seed=123,
        )
        return (
            result.part,
            result.cut,
            diag.weighted_cut_distance,
            diag.congestion_l2,
        )

    raise AssertionError(f"unknown TPCCAP variant: {variant}")


def test_heavy_edge_partition_assigns_all_qubits_when_cluster_spills() -> None:
    """Regression: per-node fallback placement must reset placement state."""
    # n_qpus * capacity = 6, so total capacity is sufficient.
    # Heavy-edge clustering first creates a 4-node cluster that cannot be placed
    # whole into capacity-3 bins, forcing the per-node fallback path.
    n = 6
    n_qpus = 2
    capacity = 3
    weights = {
        (0, 1): 10,
        (1, 2): 9,
        (2, 3): 8,
    }

    part = heavy_edge_clustering_partition(
        n=n,
        weights=weights,
        n_qpus=n_qpus,
        capacity=capacity,
    )

    assert len(part) == n
    assert all(qpu in (0, 1) for qpu in part)
    assert all(qpu >= 0 for qpu in part)


def test_heavy_edge_partition_zero_qubits() -> None:
    assert (
        heavy_edge_clustering_partition(
            n=0,
            weights={},
            n_qpus=2,
            capacity=3,
        )
        == []
    )


def test_heavy_edge_partition_no_edges_matches_first_fit_singletons() -> None:
    part = heavy_edge_clustering_partition(
        n=7,
        weights={},
        n_qpus=3,
        capacity=3,
    )
    assert part == [0, 0, 0, 1, 1, 1, 2]


def test_heavy_edge_partition_zero_and_self_loop_edges_use_singleton_fast_path() -> (
    None
):
    # All of these edges are ignored by normalization, leaving no mergeable edges.
    part = heavy_edge_clustering_partition(
        n=5,
        weights={(0, 0): 5.0, (1, 2): 0.0, (2, 1): 0.0},
        n_qpus=3,
        capacity=2,
    )
    assert part == [0, 0, 1, 1, 2]


def test_heavy_edge_partition_capacity_one_skips_merging_even_with_edges() -> None:
    part = heavy_edge_clustering_partition(
        n=4,
        weights={(0, 1): 100.0, (1, 2): 50.0, (2, 3): 25.0},
        n_qpus=4,
        capacity=1,
    )
    assert part == [0, 1, 2, 3]


def test_heavy_edge_partition_capacity_covers_all_qubits_uses_single_qpu() -> None:
    part = heavy_edge_clustering_partition(
        n=5,
        weights={(0, 1): 100.0, (1, 2): 50.0, (2, 3): 25.0, (3, 4): 12.5},
        n_qpus=3,
        capacity=5,
    )
    assert part == [0, 0, 0, 0, 0]


def test_heavy_edge_partition_equal_weights_is_order_independent() -> None:
    # Equal-weight edge ordering should not depend on mapping insertion order.
    weights_a = OrderedDict(
        [
            ((0, 1), 1.0),
            ((2, 3), 1.0),
            ((1, 2), 1.0),
        ]
    )
    weights_b = OrderedDict(
        [
            ((1, 2), 1.0),
            ((2, 3), 1.0),
            ((0, 1), 1.0),
        ]
    )

    part_a = heavy_edge_clustering_partition(
        n=4,
        weights=weights_a,
        n_qpus=2,
        capacity=2,
    )
    part_b = heavy_edge_clustering_partition(
        n=4,
        weights=weights_b,
        n_qpus=2,
        capacity=2,
    )
    assert part_a == part_b


@pytest.mark.parametrize(
    ("n", "n_qpus", "capacity"),
    [
        (4, 2, 1),
        (3, 1, 2),
        (1, 2, 0),
    ],
)
def test_heavy_edge_partition_raises_when_total_capacity_insufficient(
    n: int, n_qpus: int, capacity: int
) -> None:
    with pytest.raises(RuntimeError, match="Insufficient capacity"):
        heavy_edge_clustering_partition(
            n=n, weights={}, n_qpus=n_qpus, capacity=capacity
        )


@pytest.mark.parametrize(
    ("n", "n_qpus", "capacity"),
    [
        (-1, 2, 3),
        (1, 0, 3),
        (1, 2, -1),
    ],
)
def test_heavy_edge_partition_rejects_invalid_parameters(
    n: int, n_qpus: int, capacity: int
) -> None:
    with pytest.raises(ValueError):
        heavy_edge_clustering_partition(
            n=n,
            weights={},
            n_qpus=n_qpus,
            capacity=capacity,
        )


def test_heavy_edge_partition_rejects_out_of_range_edge() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        heavy_edge_clustering_partition(
            n=4,
            weights={(0, 4): 1},
            n_qpus=2,
            capacity=2,
        )


def test_heavy_edge_partition_rejects_non_numeric_weight() -> None:
    with pytest.raises(ValueError, match="weights must be numeric"):
        heavy_edge_clustering_partition(
            n=2,
            weights={(0, 1): object()},
            n_qpus=1,
            capacity=2,
        )


def test_heavy_edge_partition_rejects_overflowing_integer_weight() -> None:
    with pytest.raises(ValueError, match="weights must be numeric"):
        heavy_edge_clustering_partition(
            n=2,
            weights={(0, 1): 10**400},
            n_qpus=1,
            capacity=2,
        )


def test_heavy_edge_partition_rejects_bool_weight() -> None:
    with pytest.raises(ValueError, match="not booleans"):
        heavy_edge_clustering_partition(
            n=2,
            weights={(0, 1): True},
            n_qpus=1,
            capacity=2,
        )


def test_heavy_edge_partition_rejects_non_finite_weight() -> None:
    with pytest.raises(ValueError, match="finite"):
        heavy_edge_clustering_partition(
            n=2,
            weights={(0, 1): float("nan")},
            n_qpus=1,
            capacity=2,
        )


def test_heavy_edge_partition_zero_qubits_rejects_non_empty_weights() -> None:
    with pytest.raises(ValueError, match="must be empty"):
        heavy_edge_clustering_partition(
            n=0,
            weights={(0, 0): 1},
            n_qpus=1,
            capacity=0,
        )


def test_heavy_edge_partition_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        heavy_edge_clustering_partition(
            n=2,
            weights={(0, 1): -1},
            n_qpus=1,
            capacity=2,
        )


def test_heavy_edge_partition_rejects_non_tuple_edge_key() -> None:
    with pytest.raises(ValueError, match="2-tuples"):
        heavy_edge_clustering_partition(
            n=2,
            weights={0: 1},  # type: ignore[arg-type]
            n_qpus=1,
            capacity=2,
        )


@pytest.mark.parametrize(
    "partitioner",
    [
        heavy_edge_clustering_partition,
        balanced_greedy_partition,
    ],
)
def test_partitioners_reject_non_mapping_weights(partitioner: Any) -> None:
    with pytest.raises(ValueError, match="weights must be a mapping"):
        partitioner(  # type: ignore[misc]
            n=2,
            weights=[((0, 1), 1.0)],
            n_qpus=1,
            capacity=2,
        )


def test_heavy_edge_partition_rejects_infinite_weight() -> None:
    with pytest.raises(ValueError, match="finite"):
        heavy_edge_clustering_partition(
            n=2,
            weights={(0, 1): float("inf")},
            n_qpus=1,
            capacity=2,
        )


def test_balanced_greedy_partition_rejects_bool_weight() -> None:
    with pytest.raises(ValueError, match="not booleans"):
        balanced_greedy_partition(
            n=2,
            weights={(0, 1): True},
            n_qpus=1,
            capacity=2,
        )


def test_balanced_greedy_partition_zero_qubits() -> None:
    res = balanced_greedy_partition(
        n=0,
        weights={},
        n_qpus=3,
        capacity=1,
    )
    assert res.part == []
    assert res.cut == pytest.approx(0.0)
    assert res.loads == [0, 0, 0]


def test_partitioners_merge_mirrored_edge_weights() -> None:
    weights = {(0, 1): 1.5, (1, 0): 2.5}
    baseline = {(0, 1): 4.0}

    res_a = balanced_greedy_partition(n=2, weights=weights, n_qpus=2, capacity=1)
    res_b = balanced_greedy_partition(n=2, weights=baseline, n_qpus=2, capacity=1)
    assert res_a.cut == pytest.approx(res_b.cut)

    sp = all_pairs_shortest_paths([[1], [0]])
    tp_a, _ = tpccap_partition(
        n=2,
        weights=weights,
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=0,
    )
    tp_b, _ = tpccap_partition(
        n=2,
        weights=baseline,
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=0,
    )
    assert tp_a.cut == pytest.approx(tp_b.cut)


def test_partitioners_reject_non_mapping_weights_when_zero_qubits() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])

    with pytest.raises(ValueError, match="weights must be a mapping"):
        heavy_edge_clustering_partition(
            n=0,
            weights=[],  # type: ignore[arg-type]
            n_qpus=1,
            capacity=0,
        )

    with pytest.raises(ValueError, match="weights must be a mapping"):
        balanced_greedy_partition(
            n=0,
            weights=[],  # type: ignore[arg-type]
            n_qpus=1,
            capacity=0,
        )

    with pytest.raises(ValueError, match="weights must be a mapping"):
        tpccap_partition(
            n=0,
            weights=[],  # type: ignore[arg-type]
            n_qpus=2,
            capacity=0,
            comm_ports_per_qpu=1,
            sp=sp,
        )

    with pytest.raises(ValueError, match="weights must be a mapping"):
        tpccap_sa_partition(
            n=0,
            weights=[],  # type: ignore[arg-type]
            n_qpus=2,
            capacity=0,
            comm_ports_per_qpu=1,
            sp=sp,
            steps=1,
        )


def test_tpccap_partition_zero_qubits() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    res, diag = tpccap_partition(
        n=0,
        weights={},
        n_qpus=2,
        capacity=0,
        comm_ports_per_qpu=1,
        sp=sp,
    )
    assert res.part == []
    assert res.loads == [0, 0]
    assert res.cut == pytest.approx(0.0)
    assert diag.weighted_cut_distance == pytest.approx(0.0)
    assert diag.port_overflow_l2 == pytest.approx(0.0)
    assert diag.congestion_l2 == pytest.approx(0.0)
    assert diag.congestion_max == pytest.approx(0.0)


def test_tpccap_sa_partition_zero_qubits() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    res, diag, anneal = tpccap_sa_partition(
        n=0,
        weights={},
        n_qpus=2,
        capacity=0,
        comm_ports_per_qpu=1,
        sp=sp,
        steps=5,
    )
    assert res.part == []
    assert res.loads == [0, 0]
    assert res.cut == pytest.approx(0.0)
    assert diag.weighted_cut_distance == pytest.approx(0.0)
    assert diag.port_overflow_l2 == pytest.approx(0.0)
    assert diag.congestion_l2 == pytest.approx(0.0)
    assert diag.congestion_max == pytest.approx(0.0)
    assert anneal.steps == 5
    assert anneal.accepted == 0
    assert anneal.improved == 0
    assert anneal.best_objective == pytest.approx(0.0)


def test_tpccap_sa_partition_preserves_capacity_invariants() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    res, _diag, _anneal = tpccap_sa_partition(
        n=4,
        weights={(0, 1): 2.0, (1, 2): 1.5, (2, 3): 1.0},
        n_qpus=2,
        capacity=3,
        comm_ports_per_qpu=2,
        sp=sp,
        seed=7,
        steps=50,
        p_swap=0.6,
    )

    assert sum(res.loads) == 4
    assert all(load <= 3 for load in res.loads)
    assert all(q in (0, 1) for q in res.part)


def test_tpccap_sa_partition_move_only_handles_no_free_targets() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    res, _diag, anneal = tpccap_sa_partition(
        n=2,
        weights={(0, 1): 1.0},
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=5,
        steps=25,
        p_swap=0.0,
    )

    assert sorted(res.loads) == [1, 1]
    assert anneal.accepted == 0


def test_tpccap_sa_partition_swap_only_with_sparse_nonempty_qpus() -> None:
    sp = all_pairs_shortest_paths([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    res, _diag, anneal = tpccap_sa_partition(
        n=2,
        weights={(0, 1): 1.0},
        n_qpus=4,
        capacity=2,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=13,
        steps=20,
        p_swap=1.0,
    )

    assert sum(res.loads) == 2
    assert all(load <= 2 for load in res.loads)
    assert anneal.steps == 20


def test_tpccap_sa_swap_only_single_qubit_has_no_move_fallback() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    res, _diag, anneal = tpccap_sa_partition(
        n=1,
        weights={},
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=7,
        steps=30,
        p_swap=1.0,
    )

    assert sorted(res.loads) == [0, 1]
    assert anneal.accepted == 0


def test_tpccap_sa_swap_only_single_nonempty_qpu_has_no_move_fallback() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    res, _diag, anneal = tpccap_sa_partition(
        n=2,
        weights={(0, 1): 5.0},
        n_qpus=2,
        capacity=2,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=19,
        steps=40,
        p_swap=1.0,
    )

    assert sorted(res.loads) == [0, 2]
    assert anneal.accepted == 0


def test_tpccap_sa_partition_seed_is_reproducible() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    kwargs = dict(
        n=4,
        weights={(0, 1): 2.0, (1, 2): 1.5, (2, 3): 1.0},
        n_qpus=2,
        capacity=3,
        comm_ports_per_qpu=2,
        sp=sp,
        seed=11,
        steps=60,
        p_swap=0.35,
    )

    res_a, diag_a, anneal_a = tpccap_sa_partition(**kwargs)
    res_b, diag_b, anneal_b = tpccap_sa_partition(**kwargs)

    assert res_a.part == res_b.part
    assert res_a.loads == res_b.loads
    assert res_a.cut == pytest.approx(res_b.cut)
    assert diag_a == diag_b
    assert anneal_a == anneal_b


def test_tpccap_sa_partition_rejects_negative_steps() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="steps must be non-negative"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            steps=-1,
        )


def test_tpccap_sa_partition_rejects_non_positive_temperatures() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="temp0 must be positive"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            temp0=0.0,
        )
    with pytest.raises(ValueError, match="temp_end must be positive"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            temp_end=-0.1,
        )


def test_tpccap_sa_partition_rejects_invalid_swap_probability() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            p_swap=1.1,
        )


@pytest.mark.parametrize(
    ("anneal_kwargs", "match"),
    [
        ({"steps": True}, "steps must be an integer"),
        ({"temp0": True}, "temp0 must be numeric, not boolean"),
        ({"temp_end": "invalid"}, "temp_end must be numeric"),
        ({"p_swap": float("inf")}, "p_swap must be finite"),
    ],
)
def test_tpccap_sa_partition_rejects_invalid_annealing_parameter_types(
    anneal_kwargs: dict[str, object], match: str
) -> None:
    sp = all_pairs_shortest_paths([[1], [0]])

    with pytest.raises(ValueError, match=match):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            **anneal_kwargs,
        )


def test_tpccap_sa_partition_normalizes_numeric_annealing_parameters() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])

    res, _diag, anneal = tpccap_sa_partition(
        n=2,
        weights={(0, 1): 1.0},
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        seed=3,
        steps=3,
        temp0="1.0",
        temp_end="0.1",
        p_swap="0.0",
    )

    assert sorted(res.loads) == [1, 1]
    assert anneal.steps == 3


def test_tpccap_partition_rejects_invalid_control_parameters() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="comm_ports_per_qpu must be non-negative"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=-1,
            sp=sp,
        )
    with pytest.raises(ValueError, match="max_passes must be non-negative"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            max_passes=-1,
        )
    with pytest.raises(ValueError, match="max_candidate_qpus must be positive"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            max_candidate_qpus=0,
        )


def test_tpccap_sa_partition_rejects_negative_comm_ports() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="comm_ports_per_qpu must be non-negative"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=-1,
            sp=sp,
        )


def test_tpccap_partition_handles_candidate_window_extremes() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    weights = {(0, 1): 1.0}

    # smallest valid window
    res_small, _ = tpccap_partition(
        n=2,
        weights=weights,
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        max_candidate_qpus=1,
        seed=0,
    )
    assert sorted(res_small.loads) == [1, 1]

    # larger-than-n_qpus window should still work and be clipped internally
    res_large, _ = tpccap_partition(
        n=2,
        weights=weights,
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        max_candidate_qpus=10,
        seed=0,
    )
    assert sorted(res_large.loads) == [1, 1]


def test_tpccap_partition_rejects_invalid_objective_parameters() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="w_dist must be non-negative"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            w_dist=-0.1,
        )
    with pytest.raises(ValueError, match="w_port must be finite"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            w_port=float("nan"),
        )
    with pytest.raises(ValueError, match="congestion_routing"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            congestion_routing="bad",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("objective_kwargs", "match"),
    [
        ({"w_dist": True}, "w_dist must be numeric, not boolean"),
        ({"w_port": "invalid"}, "w_port must be numeric"),
        ({"w_cong": float("inf")}, "w_cong must be finite"),
    ],
)
def test_tpccap_partition_rejects_invalid_objective_parameter_types(
    objective_kwargs: dict[str, object], match: str
) -> None:
    sp = all_pairs_shortest_paths([[1], [0]])

    with pytest.raises(ValueError, match=match):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            **objective_kwargs,
        )


def test_tpccap_partition_normalizes_numeric_objective_parameters() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])

    res, diag = tpccap_partition(
        n=2,
        weights={(0, 1): 1.0},
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=sp,
        w_dist="1.0",
        w_port="5.0",
        w_cong="0.05",
    )

    assert sorted(res.loads) == [1, 1]
    assert diag.weighted_cut_distance == pytest.approx(1.0)


def test_partitioners_reject_bool_capacity_parameters() -> None:
    with pytest.raises(ValueError, match="capacity must be an integer"):
        heavy_edge_clustering_partition(
            n=1,
            weights={},
            n_qpus=1,
            capacity=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="n_qpus must be an integer"):
        balanced_greedy_partition(
            n=1,
            weights={},
            n_qpus=True,  # type: ignore[arg-type]
            capacity=1,
        )


def test_partitioners_reject_non_integer_n() -> None:
    with pytest.raises(ValueError, match="n must be an integer"):
        tpccap_partition(
            n=1.5,  # type: ignore[arg-type]
            weights={},
            n_qpus=1,
            capacity=2,
            comm_ports_per_qpu=0,
            sp=all_pairs_shortest_paths([[]]),
        )


def test_balanced_greedy_partition_rejects_invalid_alpha_balance() -> None:
    with pytest.raises(ValueError, match="alpha_balance must be non-negative"):
        balanced_greedy_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=1,
            capacity=2,
            alpha_balance=-0.1,
        )
    with pytest.raises(ValueError, match="alpha_balance must be finite"):
        balanced_greedy_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=1,
            capacity=2,
            alpha_balance=float("nan"),
        )


def test_tpccap_sa_partition_zero_qubits_still_validates_comm_ports() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="comm_ports_per_qpu must be non-negative"):
        tpccap_sa_partition(
            n=0,
            weights={},
            n_qpus=2,
            capacity=0,
            comm_ports_per_qpu=-1,
            sp=sp,
            steps=1,
        )


def test_tpccap_partition_rejects_shortest_path_dimension_mismatch() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="shortest-path dimensions"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=3,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
        )


def test_tpccap_sa_partition_rejects_shortest_path_dimension_mismatch() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="shortest-path dimensions"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=3,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            steps=1,
        )


def test_tpccap_partition_rejects_invalid_next_hop_values() -> None:
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 2], [0, 1]],
    )
    with pytest.raises(ValueError, match="next_hop"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
        )


def test_tpccap_sa_partition_rejects_negative_distances() -> None:
    sp = QpuShortestPaths(
        dist=[[0, -1], [-1, 0]],
        next_hop=[[0, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="distances must be non-negative"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            steps=1,
        )


def test_tpccap_partition_rejects_asymmetric_distances() -> None:
    sp = QpuShortestPaths(
        dist=[[0, 1], [2, 0]],
        next_hop=[[0, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="distances must be symmetric"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
        )


def test_tpccap_partition_rejects_nonzero_distance_diagonal() -> None:
    sp = QpuShortestPaths(
        dist=[[1, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="distance diagonal"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
        )


def test_tpccap_partition_rejects_invalid_next_hop_diagonal() -> None:
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[1, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="next_hop diagonal"):
        tpccap_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
        )


def test_tpccap_sa_partition_rejects_invalid_next_hop_diagonal() -> None:
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[1, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="next_hop diagonal"):
        tpccap_sa_partition(
            n=2,
            weights={(0, 1): 1.0},
            n_qpus=2,
            capacity=1,
            comm_ports_per_qpu=1,
            sp=sp,
            steps=1,
        )


@pytest.mark.parametrize("variant", ["tpccap", "tpccap_sa"])
def test_tpccap_variants_penalize_unroutable_cuts_without_crashing(
    variant: str,
) -> None:
    part, cut, weighted_cut_distance, congestion_l2 = _run_tpccap_variant(
        variant,
        n=2,
        weights={(0, 1): 1.0},
        n_qpus=2,
        capacity=1,
        comm_ports_per_qpu=1,
        sp=_disconnected_two_qpu_paths(),
    )

    assert part == [0, 1]
    assert cut == 1.0
    assert weighted_cut_distance == float(UNREACHABLE_DISTANCE)
    assert congestion_l2 == float(UNREACHABLE_DISTANCE**2)


@pytest.mark.parametrize("variant", ["tpccap", "tpccap_sa"])
def test_tpccap_variants_avoid_unroutable_cuts_when_capacity_permits(
    variant: str,
) -> None:
    part, cut, weighted_cut_distance, congestion_l2 = _run_tpccap_variant(
        variant,
        n=2,
        weights={(0, 1): 1.0},
        n_qpus=2,
        capacity=2,
        comm_ports_per_qpu=1,
        sp=_disconnected_two_qpu_paths(),
    )

    assert part == [0, 0]
    assert cut == 0.0
    assert weighted_cut_distance == 0.0
    assert congestion_l2 == 0.0


def test_tpccap_partition_combines_routable_and_unroutable_congestion() -> None:
    sp = QpuShortestPaths(
        dist=[
            [0, 1, UNREACHABLE_DISTANCE],
            [1, 0, UNREACHABLE_DISTANCE],
            [UNREACHABLE_DISTANCE, UNREACHABLE_DISTANCE, 0],
        ],
        next_hop=[[0, 1, -1], [0, 1, -1], [-1, -1, 2]],
    )

    result, diag = tpccap_partition(
        n=3,
        weights={(0, 1): 2.0, (0, 2): 3.0},
        n_qpus=3,
        capacity=1,
        comm_ports_per_qpu=2,
        sp=sp,
        congestion_routing="single_path",
    )

    assert sorted(result.part) == [0, 1, 2]
    assert result.cut == 5.0

    edge_loads = [
        (result.part[0], result.part[1], 2.0),
        (result.part[0], result.part[2], 3.0),
    ]
    routable_loads = [
        load
        for src, dst, load in edge_loads
        if sp.dist[src][dst] < UNREACHABLE_DISTANCE
    ]
    unroutable_loads = [
        load
        for src, dst, load in edge_loads
        if sp.dist[src][dst] >= UNREACHABLE_DISTANCE
    ]

    assert len(routable_loads) == 1
    assert len(unroutable_loads) == 1
    assert diag.weighted_cut_distance == sum(routable_loads) + sum(
        load * float(UNREACHABLE_DISTANCE) for load in unroutable_loads
    )
    assert diag.congestion_max == max(unroutable_loads) * float(UNREACHABLE_DISTANCE)
    assert diag.congestion_l2 == sum(load**2 for load in routable_loads) + sum(
        (load * float(UNREACHABLE_DISTANCE)) ** 2 for load in unroutable_loads
    )


def test_tpccap_partition_sums_unroutable_l2_per_qpu_pair() -> None:
    sp = QpuShortestPaths(
        dist=[
            [0, UNREACHABLE_DISTANCE, UNREACHABLE_DISTANCE],
            [UNREACHABLE_DISTANCE, 0, UNREACHABLE_DISTANCE],
            [UNREACHABLE_DISTANCE, UNREACHABLE_DISTANCE, 0],
        ],
        next_hop=[[0, -1, -1], [-1, 1, -1], [-1, -1, 2]],
    )

    result, diag = tpccap_partition(
        n=3,
        weights={(0, 1): 2.0, (0, 2): 3.0},
        n_qpus=3,
        capacity=1,
        comm_ports_per_qpu=2,
        sp=sp,
    )

    assert sorted(result.part) == [0, 1, 2]
    assert result.cut == 5.0
    assert diag.congestion_max == 3.0 * float(UNREACHABLE_DISTANCE)
    assert (
        diag.congestion_l2
        == (2.0 * float(UNREACHABLE_DISTANCE)) ** 2
        + (3.0 * float(UNREACHABLE_DISTANCE)) ** 2
    )
