# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="arg-type,list-item,dict-item,var-annotated"
# These tests intentionally pass malformed runtime values to verify validation errors.
import math
from typing import get_args

import pytest

from quport.config import InterTopology, MultiQPUConfig
from quport.network import (
    UNREACHABLE_DISTANCE,
    QpuShortestPaths,
    all_pairs_shortest_paths,
    build_qpu_graph,
    compute_boundary_counts,
    compute_traffic_matrix,
    congestion_metrics,
    route_link_loads,
    topology_metrics,
)

ALL_INTER_TOPOLOGIES: tuple[InterTopology, ...] = get_args(InterTopology)


def test_build_qpu_graph_degree_d_respects_requested_odd_degree_on_even_n() -> None:
    cfg = MultiQPUConfig(
        n_qpus=6,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=3,
    )
    adj = build_qpu_graph(cfg)
    assert all(len(nbrs) == 3 for nbrs in adj)


def test_build_qpu_graph_degree_d_caps_degree_for_small_networks() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=99,
    )
    adj = build_qpu_graph(cfg)
    assert adj == [[1, 2], [0, 2], [0, 1]]


def test_build_qpu_graph_degree_d_handles_odd_degree_on_odd_n_without_over_degree() -> (
    None
):
    cfg = MultiQPUConfig(
        n_qpus=5,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=3,
    )
    adj = build_qpu_graph(cfg)
    assert all(len(nbrs) == 2 for nbrs in adj)


@pytest.mark.parametrize("topo", ALL_INTER_TOPOLOGIES)
def test_build_qpu_graph_single_qpu_returns_empty_for_all_topologies(
    topo: InterTopology,
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=1,
        comm_qubits_per_qpu=2,
        inter_topology=topo,
        inter_degree=3,
    )
    assert build_qpu_graph(cfg) == [[]]


def test_build_qpu_graph_degree_d_complete_graph_fast_path() -> None:
    cfg = MultiQPUConfig(
        n_qpus=7,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=6,
    )
    adj = build_qpu_graph(cfg)
    for u, nbrs in enumerate(adj):
        assert len(nbrs) == 6
        assert u not in nbrs


@pytest.mark.parametrize("topo", ALL_INTER_TOPOLOGIES)
def test_build_qpu_graph_zero_comm_ports_returns_empty_adjacency_for_all_topologies(
    topo: InterTopology,
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        comm_qubits_per_qpu=0,
        inter_topology=topo,
        inter_degree=3,
    )
    assert build_qpu_graph(cfg) == [[], [], [], []]


@pytest.mark.parametrize("topo", ALL_INTER_TOPOLOGIES)
def test_build_qpu_graph_zero_qpus_returns_empty_for_all_topologies(
    topo: InterTopology,
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=0,
        comm_qubits_per_qpu=1,
        inter_topology=topo,
        inter_degree=3,
    )
    assert build_qpu_graph(cfg) == []


def test_build_qpu_graph_rejects_unknown_topology_even_for_zero_qpus() -> None:
    cfg = MultiQPUConfig(n_qpus=0, comm_qubits_per_qpu=1, inter_topology="switch")
    object.__setattr__(cfg, "inter_topology", "unknown")
    with pytest.raises(ValueError, match="Unknown inter_topology"):
        build_qpu_graph(cfg)


def test_build_qpu_graph_rejects_unknown_topology_even_for_zero_comm_ports() -> None:
    cfg = MultiQPUConfig(n_qpus=4, comm_qubits_per_qpu=0, inter_topology="switch")
    object.__setattr__(cfg, "inter_topology", "unknown")
    with pytest.raises(ValueError, match="Unknown inter_topology"):
        build_qpu_graph(cfg)


def test_build_qpu_graph_degree_d_allows_zero_degree() -> None:
    cfg = MultiQPUConfig(
        n_qpus=5,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=0,
    )
    assert build_qpu_graph(cfg) == [[], [], [], [], []]


def test_build_qpu_graph_degree_d_clamps_negative_degree_to_zero() -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=-3,
    )
    assert build_qpu_graph(cfg) == [[], [], [], []]


@pytest.mark.parametrize(
    ("field", "value", "msg"),
    [
        ("inter_degree", True, "inter_degree must be an integer"),
        ("inter_degree", "3", "inter_degree must be an integer"),
        ("inter_degree", 3.0, "inter_degree must be an integer"),
        ("n_qpus", True, "n_qpus must be an integer"),
        (
            "comm_qubits_per_qpu",
            True,
            "comm_qubits_per_qpu must be an integer",
        ),
    ],
)
def test_build_qpu_graph_rejects_invalid_integer_field_values(
    field: str, value: object, msg: str
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=3,
    )
    object.__setattr__(cfg, field, value)
    with pytest.raises(ValueError, match=msg):
        build_qpu_graph(cfg)


def test_build_qpu_graph_accepts_integer_subclass_values() -> None:
    class FancyInt(int):
        pass

    cfg = MultiQPUConfig(
        n_qpus=4,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=2,
    )
    object.__setattr__(cfg, "n_qpus", FancyInt(4))
    object.__setattr__(cfg, "comm_qubits_per_qpu", FancyInt(1))
    object.__setattr__(cfg, "inter_degree", FancyInt(2))
    adj = build_qpu_graph(cfg)
    assert all(len(row) == 2 for row in adj)


@pytest.mark.parametrize(
    ("field", "msg"),
    [
        ("n_qpus", "n_qpus must be non-negative"),
        ("comm_qubits_per_qpu", "comm_qubits_per_qpu must be non-negative"),
    ],
)
def test_build_qpu_graph_rejects_negative_size_fields(field: str, msg: str) -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        comm_qubits_per_qpu=1,
        inter_topology="switch",
    )
    object.__setattr__(cfg, field, -1)
    with pytest.raises(ValueError, match=msg):
        build_qpu_graph(cfg)


def test_partition_validation_rejects_out_of_range_assignment() -> None:
    weights = {(0, 1): 1.0}
    part = [0, 2]
    with pytest.raises(ValueError, match="out of range"):
        compute_traffic_matrix(weights, part, n_qpus=2)


def test_partition_validation_rejects_too_short_partition() -> None:
    weights = {(0, 2): 1.0}
    part = [0, 1]
    with pytest.raises(ValueError, match="partition length"):
        compute_boundary_counts(weights, part, n_qpus=3)


def test_partition_validation_rejects_non_positive_qpu_count() -> None:
    weights = {(0, 1): 1.0}
    part = [0, 0]
    with pytest.raises(ValueError, match="n_qpus must be positive"):
        compute_traffic_matrix(weights, part, n_qpus=0)


def test_partition_validation_rejects_negative_logical_index() -> None:
    weights = {(-1, 0): 1.0}
    part = [0]
    with pytest.raises(ValueError, match="negative logical index"):
        compute_boundary_counts(weights, part, n_qpus=1)


def test_partition_validation_rejects_non_tuple_weight_key() -> None:
    weights = {0: 1.0}
    part = [0, 0]
    with pytest.raises(ValueError, match="2-tuples"):
        compute_traffic_matrix(weights, part, n_qpus=1)


def test_partition_validation_rejects_bool_logical_index() -> None:
    weights = {(True, 1): 1.0}
    part = [0, 0]
    with pytest.raises(ValueError, match="integer logical indices"):
        compute_boundary_counts(weights, part, n_qpus=1)


def test_compute_traffic_matrix_merges_mirrored_edges_once() -> None:
    weights = {(0, 1): 1.0, (1, 0): 2.0}
    part = [0, 1]
    traffic = compute_traffic_matrix(weights, part, n_qpus=2)
    assert traffic == [[0.0, 3.0], [3.0, 0.0]]


def test_compute_traffic_matrix_rejects_non_finite_weight() -> None:
    weights = {(0, 1): math.nan}
    part = [0, 1]
    with pytest.raises(ValueError, match="finite"):
        compute_traffic_matrix(weights, part, n_qpus=2)


def test_compute_traffic_matrix_rejects_bool_weight() -> None:
    weights = {(0, 1): True}
    part = [0, 1]
    with pytest.raises(ValueError, match="not booleans"):
        compute_traffic_matrix(weights, part, n_qpus=2)


def test_compute_traffic_matrix_rejects_non_numeric_weight() -> None:
    weights = {(0, 1): "abc"}
    part = [0, 1]
    with pytest.raises(ValueError, match="weights must be numeric"):
        compute_traffic_matrix(weights, part, n_qpus=2)


def test_compute_traffic_matrix_rejects_weight_float_overflow() -> None:
    class FloatOverflow:
        def __float__(self) -> float:
            raise OverflowError("overflow")

    weights = {(0, 1): FloatOverflow()}
    part = [0, 1]
    with pytest.raises(ValueError, match="weights must be numeric"):
        compute_traffic_matrix(weights, part, n_qpus=2)


def test_compute_traffic_matrix_rejects_non_finite_self_loop_weight() -> None:
    weights = {(0, 0): math.nan}
    part = [0]
    with pytest.raises(ValueError, match="finite"):
        compute_traffic_matrix(weights, part, n_qpus=1)


def test_compute_traffic_matrix_empty_weights_returns_zero_matrix() -> None:
    part = [0, 1, 0]
    assert compute_traffic_matrix({}, part, n_qpus=3) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_compute_boundary_counts_rejects_non_finite_self_loop_weight() -> None:
    weights = {(0, 0): math.inf}
    part = [0]
    with pytest.raises(ValueError, match="finite"):
        compute_boundary_counts(weights, part, n_qpus=1)


def test_compute_boundary_counts_rejects_negative_weight() -> None:
    weights = {(0, 1): -1.0}
    part = [0, 1]
    with pytest.raises(ValueError, match="non-negative"):
        compute_boundary_counts(weights, part, n_qpus=2)


def test_compute_boundary_counts_rejects_non_numeric_weight() -> None:
    weights = {(0, 1): object()}
    part = [0, 1]
    with pytest.raises(ValueError, match="weights must be numeric"):
        compute_boundary_counts(weights, part, n_qpus=2)


def test_compute_boundary_counts_empty_weights_returns_zero_counts() -> None:
    assert compute_boundary_counts({}, [0, 1, 0], n_qpus=3) == [0, 0, 0]


def test_compute_boundary_counts_preserves_tiny_positive_remote_edge() -> None:
    tiny = 5e-13
    counts = compute_boundary_counts({(0, 1): tiny}, [0, 1], n_qpus=2)
    assert counts == [1, 1]


def test_compute_boundary_counts_deduplicates_mirrored_remote_edges() -> None:
    counts = compute_boundary_counts(
        {(0, 1): 1.0, (1, 0): 2.0, (0, 2): 3.0},
        [0, 1, 0],
        n_qpus=2,
    )
    assert counts == [1, 1]


def test_compute_boundary_counts_keeps_zero_for_uninvolved_qpus() -> None:
    counts = compute_boundary_counts({(0, 1): 1.0}, [0, 1], n_qpus=4)
    assert counts == [1, 1, 0, 0]


def test_compute_traffic_matrix_preserves_tiny_positive_weight() -> None:
    tiny = 5e-13
    traffic = compute_traffic_matrix({(0, 1): tiny}, [0, 1], n_qpus=2)
    assert traffic[0][1] == pytest.approx(tiny)
    assert traffic[1][0] == pytest.approx(tiny)


def test_partition_validation_rejects_bool_partition_assignment() -> None:
    weights = {}
    part = [True]
    with pytest.raises(ValueError, match="partition assignment must be an integer"):
        compute_traffic_matrix(weights, part, n_qpus=2)


def test_partition_validation_rejects_non_integer_partition_assignment() -> None:
    weights = {}
    part = ["0"]
    with pytest.raises(ValueError, match="partition assignment must be an integer"):
        compute_boundary_counts(weights, part, n_qpus=2)


def test_partition_validation_allows_empty_partition_without_weights() -> None:
    assert compute_traffic_matrix({}, [], n_qpus=2) == [[0.0, 0.0], [0.0, 0.0]]
    assert compute_boundary_counts({}, [], n_qpus=2) == [0, 0]


def test_partition_validation_rejects_empty_partition_with_weights_for_boundary_counts() -> (
    None
):
    with pytest.raises(ValueError, match="partition cannot be empty"):
        compute_boundary_counts({(0, 0): 1.0}, [], n_qpus=2)


def test_partition_validation_rejects_empty_partition_with_weights_for_traffic_matrix() -> (
    None
):
    with pytest.raises(ValueError, match="partition cannot be empty"):
        compute_traffic_matrix({(0, 0): 1.0}, [], n_qpus=2)


def test_route_link_loads_rejects_non_sequence_traffic_matrix() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="traffic matrix must be a sequence of rows"):
        route_link_loads(0, sp)


def test_route_link_loads_rejects_non_sequence_traffic_row() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="traffic matrix rows must be sequences"):
        route_link_loads([0, [0.0, 0.0]], sp)


def test_route_link_loads_rejects_non_sequence_shortest_path_table() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=0, next_hop=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="shortest-path must be a sequence of rows"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_non_sequence_shortest_path_row() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[0, [1, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="shortest-path rows must be sequences"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_negative_traffic() -> None:
    traffic = [[0.0, -1.0], [0.0, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="non-negative"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_disconnected_paths() -> None:
    traffic = [[0.0, 2.0], [2.0, 0.0]]
    sp = all_pairs_shortest_paths([[], []])
    with pytest.raises(ValueError, match="no path"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_shape_mismatch() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = all_pairs_shortest_paths([[1, 2], [0, 2], [0, 1]])
    with pytest.raises(ValueError, match="dimensions"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_asymmetric_traffic() -> None:
    traffic = [[0.0, 1.0], [2.0, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="symmetric"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_diagonal_traffic() -> None:
    traffic = [[1.0, 0.0], [0.0, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="diagonal"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_non_numeric_traffic_entries() -> None:
    traffic = [[0.0, "x"], ["x", 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="numeric values"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_bool_traffic_entries() -> None:
    traffic = [[0.0, True], [True, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="not booleans"):
        route_link_loads(traffic, sp)


@pytest.mark.parametrize(
    "value", ["1.0", b"1.0", bytearray(b"1.0"), memoryview(b"1.0")]
)
def test_route_link_loads_rejects_string_like_numeric_traffic_entries(
    value: object,
) -> None:
    traffic = [[0.0, value], [value, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="numeric values"):
        route_link_loads(traffic, sp)


class _FloatLike:
    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


class _IndexLike:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


@pytest.mark.parametrize(
    ("value", "expected"),
    [(_FloatLike(2.5), 2.5), (_IndexLike(3), 3.0)],
)
def test_route_link_loads_accepts_numeric_protocol_traffic_entries(
    value: object, expected: float
) -> None:
    traffic = [[0.0, value], [value, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    loads = route_link_loads(traffic, sp)
    assert loads == {(0, 1): pytest.approx(expected)}


def test_route_link_loads_rejects_traffic_entry_float_overflow() -> None:
    class FloatOverflow:
        def __float__(self) -> float:
            raise OverflowError("overflow")

    value = FloatOverflow()
    traffic = [[0.0, value], [value, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="numeric values"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_non_finite_traffic() -> None:
    traffic = [[0.0, math.nan], [math.nan, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="finite"):
        route_link_loads(traffic, sp)

    traffic = [[0.0, math.inf], [math.inf, 0.0]]
    with pytest.raises(ValueError, match="finite"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_cyclic_next_hop() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="cycle"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_invalid_next_hop_indices() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, 2], [1, 1]])
    with pytest.raises(ValueError, match="next_hop"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_non_integer_next_hop_entries() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, 1.0], [0, 1]])
    with pytest.raises(ValueError, match="must contain integer indices"):
        route_link_loads(traffic, sp)


def test_route_link_loads_rejects_non_integer_distance_entries() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1.0], [1, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="shortest-path distances must be an integer"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_bool_distance_entries() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, True], [1, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(
        ValueError, match="shortest-path distances must be an integer, not boolean"
    ):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_negative_distance_entries() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, -1], [1, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(
        ValueError, match="shortest-path distances must be non-negative"
    ):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_nonzero_distance_diagonal_in_ecmp() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[1, 1], [1, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="zero diagonal"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_zero_off_diagonal_distance_in_ecmp() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 0], [0, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="positive off-diagonal"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_overlong_finite_distance_in_ecmp() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, UNREACHABLE_DISTANCE - 1], [UNREACHABLE_DISTANCE - 1, 0]],
        next_hop=[[0, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="maximum finite graph distance"):
        route_link_loads(traffic, sp, mode="ecmp")


@pytest.mark.parametrize(
    "unreachable_like",
    [UNREACHABLE_DISTANCE, UNREACHABLE_DISTANCE + 1],
)
def test_route_link_loads_treats_unreachable_like_distances_as_disconnected_in_ecmp(
    unreachable_like: int,
) -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, unreachable_like], [unreachable_like, 0]],
        next_hop=[[0, 1], [0, 1]],
    )
    with pytest.raises(ValueError, match="no path between QPU 0 and 1"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_asymmetric_distance_matrix_in_ecmp() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1], [2, 0]], next_hop=[[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="distances must be symmetric"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_ecmp_adjacency_inconsistent_with_distance() -> None:
    traffic = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        next_hop=[[0, 1, 1], [0, 1, 2], [1, 1, 2]],
        adj=[[2], [], [0]],
    )
    with pytest.raises(ValueError, match="align with unit distances"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_ecmp_adjacency_missing_unit_distance_edge() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[[], []],
    )
    with pytest.raises(ValueError, match="align with unit distances"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_single_path_ignores_unused_distance_values() -> None:
    traffic = [[0.0, 2.0], [2.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, -1], [1.0, 0]], next_hop=[[0, 1], [0, 1]])
    loads = route_link_loads(traffic, sp, mode="single_path")
    assert loads == {(0, 1): 2.0}


def test_route_link_loads_ecmp_ignores_invalid_next_hop_table() -> None:
    traffic = [[0.0, 2.0], [2.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, "bad"], [None, 1]])
    loads = route_link_loads(traffic, sp, mode="ecmp")
    assert loads == {(0, 1): 2.0}


def test_route_link_loads_rejects_bool_next_hop_entries() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, True], [0, 1]])
    with pytest.raises(ValueError, match="must contain integer indices"):
        route_link_loads(traffic, sp)


def test_qpu_shortest_paths_path_rejects_invalid_src_dst_indices() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="valid QPU indices"):
        sp.path(-1, 0)
    with pytest.raises(ValueError, match="valid QPU indices"):
        sp.path(0, 2)


def test_qpu_shortest_paths_path_rejects_next_hop_cycle() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="contains a cycle"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_non_integer_src_dst_indices() -> None:
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="integer QPU indices"):
        sp.path(0.0, 1)
    with pytest.raises(ValueError, match="integer QPU indices"):
        sp.path(0, True)


def test_qpu_shortest_paths_path_rejects_inconsistent_next_hop_dimensions() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0], [0, 1]])
    with pytest.raises(ValueError, match="dimensions are inconsistent"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_non_sequence_next_hop_row() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[0, [0, 1]])
    with pytest.raises(ValueError, match="rows must be sequences"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_string_like_next_hop_row() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=["01", [0, 1]])
    with pytest.raises(ValueError, match="rows must be sequences"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_buffer_like_next_hop_row() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[bytearray(b"01"), [0, 1]])
    with pytest.raises(ValueError, match="rows must be sequences"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_non_sequence_next_hop_table() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=0)
    with pytest.raises(ValueError, match="must be a sequence of rows"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_string_like_next_hop_table() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop="01")
    with pytest.raises(ValueError, match="must be a sequence of rows"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_buffer_like_next_hop_table() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=bytearray(b"01"))
    with pytest.raises(ValueError, match="must be a sequence of rows"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_non_integer_next_hop_index() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, 1.0], [0, 1]])
    with pytest.raises(ValueError, match="must contain integer indices"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_rejects_invalid_next_hop_index() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0, 2], [0, 1]])
    with pytest.raises(ValueError, match="contains invalid indices"):
        sp.path(0, 1)


def test_qpu_shortest_paths_path_src_eq_dst_does_not_depend_on_row_shape() -> None:
    sp = QpuShortestPaths(dist=[[0, 1], [1, 0]], next_hop=[[0], [0, 1]])
    assert sp.path(1, 1) == [1]


def test_route_link_loads_ecmp_balances_parallel_paths() -> None:
    # 0-1-3 and 0-2-3 are two equal shortest paths between 0 and 3.
    adj = [[1, 2], [0, 3], [0, 3], [1, 2]]
    sp = all_pairs_shortest_paths(adj)
    traffic = [
        [0.0, 0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
    ]

    loads_single = route_link_loads(traffic, sp, mode="single_path")
    loads_ecmp = route_link_loads(traffic, sp, mode="ecmp")

    assert loads_single[(0, 1)] == pytest.approx(10.0)
    assert loads_single[(1, 3)] == pytest.approx(10.0)
    assert (0, 2) not in loads_single

    assert loads_ecmp[(0, 1)] == pytest.approx(5.0)
    assert loads_ecmp[(1, 3)] == pytest.approx(5.0)
    assert loads_ecmp[(0, 2)] == pytest.approx(5.0)
    assert loads_ecmp[(2, 3)] == pytest.approx(5.0)


def test_route_link_loads_rejects_invalid_routing_mode() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = all_pairs_shortest_paths([[1], [0]])
    with pytest.raises(ValueError, match="routing mode"):
        route_link_loads(traffic, sp, mode="invalid")


def test_route_link_loads_rejects_bad_ecmp_adjacency_shape() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[[1]],
    )
    with pytest.raises(ValueError, match="adjacency dimensions"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_all_pairs_shortest_paths_rejects_non_sequence_adjacency_table() -> None:
    with pytest.raises(ValueError, match="adjacency must be a sequence of rows"):
        all_pairs_shortest_paths(0)


def test_all_pairs_shortest_paths_rejects_string_like_adjacency_table() -> None:
    with pytest.raises(ValueError, match="adjacency must be a sequence of rows"):
        all_pairs_shortest_paths("01")


def test_route_link_loads_ecmp_deduplicates_adjacency_rows() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[[1, 1, 1], [0, 0]],
    )
    loads = route_link_loads(traffic, sp, mode="ecmp")
    assert loads[(0, 1)] == pytest.approx(1.0)


def test_route_link_loads_rejects_bad_ecmp_adjacency_indices() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[[2], [0]],
    )
    with pytest.raises(ValueError, match="adjacency contains invalid indices"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_asymmetric_ecmp_adjacency() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[[1], []],
    )
    with pytest.raises(ValueError, match="must be symmetric"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_non_sequence_ecmp_adjacency_row() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[1, [0]],
    )
    with pytest.raises(ValueError, match="rows must be sequences"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_route_link_loads_rejects_bool_ecmp_adjacency_indices() -> None:
    traffic = [[0.0, 1.0], [1.0, 0.0]]
    sp = QpuShortestPaths(
        dist=[[0, 1], [1, 0]],
        next_hop=[[0, 1], [0, 1]],
        adj=[[True], [0]],
    )
    with pytest.raises(ValueError, match="must contain integer indices"):
        route_link_loads(traffic, sp, mode="ecmp")


def test_all_pairs_shortest_paths_rejects_asymmetric_adjacency() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        all_pairs_shortest_paths([[1], []])


def test_all_pairs_shortest_paths_rejects_non_sequence_row() -> None:
    with pytest.raises(ValueError, match="rows must be sequences"):
        all_pairs_shortest_paths([1, [0]])


def test_all_pairs_shortest_paths_rejects_bool_index() -> None:
    with pytest.raises(ValueError, match="integer indices"):
        all_pairs_shortest_paths([[True], [0]])


def test_all_pairs_shortest_paths_deduplicates_and_strips_self_loops() -> None:
    sp = all_pairs_shortest_paths([[0, 1, 1], [0, 1]])
    assert sp.adj is not None
    assert sp.adj == [[1], [0]]


def test_route_link_loads_ecmp_conserves_source_outflow() -> None:
    adj = [[1, 2], [0, 3, 4], [0, 3, 4], [1, 2, 5], [1, 2, 5], [3, 4]]
    sp = all_pairs_shortest_paths(adj)
    traffic = [[0.0] * 6 for _ in range(6)]
    traffic[0][5] = 7.0
    traffic[5][0] = 7.0

    loads = route_link_loads(traffic, sp, mode="ecmp")
    src_outflow = sum(w for (u, v), w in loads.items() if u == 0 or v == 0)
    assert src_outflow == pytest.approx(7.0)


def test_build_qpu_graph_rejects_unhashable_topology_value() -> None:
    cfg = MultiQPUConfig(n_qpus=2, comm_qubits_per_qpu=1, inter_topology="ring")
    object.__setattr__(cfg, "inter_topology", ["ring"])
    with pytest.raises(ValueError, match="Unknown inter_topology"):
        build_qpu_graph(cfg)


def test_all_pairs_shortest_paths_prefers_first_bfs_parent_for_equal_paths() -> None:
    # Two equal shortest paths 0-1-3 and 0-2-3; normalized adjacency is sorted,
    # so BFS should pick 1 as deterministic first hop.
    sp = all_pairs_shortest_paths([[2, 1], [0, 3], [0, 3], [1, 2]])
    assert sp.path(0, 3) == [0, 1, 3]


def test_all_pairs_shortest_paths_marks_disconnected_pairs_unreachable() -> None:
    sp = all_pairs_shortest_paths([[1], [0], []])
    assert sp.dist[0][2] >= UNREACHABLE_DISTANCE
    assert sp.next_hop[0][2] == -1


def test_congestion_metrics_empty_loads_returns_zero_metrics() -> None:
    metrics = congestion_metrics({})
    assert metrics.max_load == 0.0
    assert metrics.l2_load == 0.0


def test_congestion_metrics_computes_max_and_l2_load() -> None:
    metrics = congestion_metrics({(0, 1): 1.5, (1, 2): 2.0, (0, 2): 0.5})
    assert metrics.max_load == pytest.approx(2.0)
    assert metrics.l2_load == pytest.approx(1.5 * 1.5 + 2.0 * 2.0 + 0.5 * 0.5)


def test_congestion_metrics_rejects_non_mapping_inputs() -> None:
    with pytest.raises(TypeError, match="provided as a mapping"):
        congestion_metrics([(0, 1, 1.0)])  # type: ignore[arg-type]


def test_congestion_metrics_accepts_non_dict_mapping() -> None:
    from collections import OrderedDict

    loads = OrderedDict([((0, 1), 2), ((1, 2), 3.5)])
    metrics = congestion_metrics(loads)
    assert metrics.max_load == pytest.approx(3.5)
    assert metrics.l2_load == pytest.approx(2.0 * 2.0 + 3.5 * 3.5)


def test_congestion_metrics_aggregates_reverse_oriented_edges() -> None:
    metrics = congestion_metrics({(0, 1): 1.25, (1, 0): 0.75, (1, 2): 2.0})
    assert metrics.max_load == pytest.approx(2.0)
    assert metrics.l2_load == pytest.approx(2.0 * 2.0 + 2.0 * 2.0)


@pytest.mark.parametrize(
    ("bad_load", "error"),
    [
        (True, "not booleans"),
        (-1.0, "non-negative"),
        (float("inf"), "finite"),
        (float("nan"), "finite"),
        ("bad", "numeric"),
    ],
)
def test_congestion_metrics_rejects_invalid_link_load_values(
    bad_load: object, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        congestion_metrics({(0, 1): bad_load})


@pytest.mark.parametrize(
    ("bad_edge", "error"),
    [
        (0, "2-tuples"),
        ((0, 1, 2), "2-tuples"),
        ((True, 1), "integer"),
        ((-1, 1), "non-negative"),
        ((1, 1), "self-loops"),
    ],
)
def test_congestion_metrics_rejects_invalid_edge_keys(
    bad_edge: object, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        congestion_metrics({bad_edge: 1.0})  # type: ignore[dict-item]


def test_topology_metrics_for_ring_reports_expected_structure() -> None:
    cfg = MultiQPUConfig(n_qpus=6, comm_qubits_per_qpu=1, inter_topology="ring")

    metrics = topology_metrics(build_qpu_graph(cfg))

    assert metrics.n_qpus == 6
    assert metrics.edges == 6
    assert metrics.min_degree == 2
    assert metrics.max_degree == 2
    assert metrics.average_degree == 2.0
    assert metrics.connected is True
    assert metrics.components == 1
    assert metrics.diameter == 3
    assert metrics.average_shortest_path == pytest.approx(1.8)
    assert metrics.unreachable_pairs == 0


def test_topology_metrics_reports_disconnected_zero_degree_topology() -> None:
    cfg = MultiQPUConfig(
        n_qpus=4,
        comm_qubits_per_qpu=1,
        inter_topology="degree_d",
        inter_degree=0,
    )

    metrics = topology_metrics(build_qpu_graph(cfg))

    assert metrics.connected is False
    assert metrics.components == 4
    assert metrics.edges == 0
    assert metrics.diameter == 0
    assert metrics.average_shortest_path == 0.0
    assert metrics.unreachable_pairs == 6


def test_topology_metrics_empty_graph_uses_zero_metrics() -> None:
    metrics = topology_metrics([])

    assert metrics.n_qpus == 0
    assert metrics.edges == 0
    assert metrics.min_degree == 0
    assert metrics.max_degree == 0
    assert metrics.average_degree == 0.0
    assert metrics.connected is True
    assert metrics.components == 0
    assert metrics.diameter == 0
    assert metrics.average_shortest_path == 0.0
    assert metrics.unreachable_pairs == 0


def test_topology_metrics_single_qpu_graph_is_connected_without_pairs() -> None:
    metrics = topology_metrics([[]])

    assert metrics.n_qpus == 1
    assert metrics.edges == 0
    assert metrics.min_degree == 0
    assert metrics.max_degree == 0
    assert metrics.average_degree == 0.0
    assert metrics.connected is True
    assert metrics.components == 1
    assert metrics.diameter == 0
    assert metrics.average_shortest_path == 0.0
    assert metrics.unreachable_pairs == 0


def test_topology_metrics_complete_graph_reports_unit_distances() -> None:
    cfg = MultiQPUConfig(n_qpus=5, comm_qubits_per_qpu=1, inter_topology="switch")

    metrics = topology_metrics(build_qpu_graph(cfg))

    assert metrics.edges == 10
    assert metrics.min_degree == 4
    assert metrics.max_degree == 4
    assert metrics.average_degree == 4.0
    assert metrics.connected is True
    assert metrics.components == 1
    assert metrics.diameter == 1
    assert metrics.average_shortest_path == 1.0
    assert metrics.unreachable_pairs == 0


def test_topology_metrics_normalizes_duplicate_edges_and_self_loops() -> None:
    metrics = topology_metrics([[0, 1, 1], [0, 1]])

    assert metrics.n_qpus == 2
    assert metrics.edges == 1
    assert metrics.min_degree == 1
    assert metrics.max_degree == 1
    assert metrics.diameter == 1
    assert metrics.average_shortest_path == 1.0


@pytest.mark.parametrize(
    ("adj", "error"),
    [
        ("01", "adjacency must be a sequence of rows"),
        ([[1], []], "adjacency must be symmetric"),
        ([[2], []], "adjacency contains invalid indices"),
        ([[True], []], "adjacency must contain integer indices"),
        ([1, []], "adjacency rows must be sequences"),
    ],
)
def test_topology_metrics_rejects_malformed_adjacency(adj: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        topology_metrics(adj)  # type: ignore[arg-type]
