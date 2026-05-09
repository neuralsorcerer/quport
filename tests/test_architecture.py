# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

pytest.importorskip("qiskit")

from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig


def test_bidirectional_edges_in_switch() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)
    cm = arch.build_coupling_map()
    edges = set(cm.get_edges())
    # pick a comm qubit in qpu0 and qpu1
    comm0 = arch.block_of_qpu(0).comm[0]
    comm1 = arch.block_of_qpu(1).comm[0]
    assert (comm0, comm1) in edges
    assert (comm1, comm0) in edges


def test_intra_clique_bidirectional() -> None:
    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    cm = arch.build_coupling_map()
    edges = set(cm.get_edges())
    nodes = arch.block_of_qpu(0).compute + arch.block_of_qpu(0).comm
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            assert (a, b) in edges
            assert (b, a) in edges


def _qpu_comm_edge_pairs(arch: MultiQPUArchitecture) -> set[tuple[int, int]]:
    inter_cm = arch.build_interconnect_coupling_map()
    qpu_pairs: set[tuple[int, int]] = set()
    for u, v in inter_cm.get_edges():
        if u < v:
            a = arch.qpu_of_phys(u)
            b = arch.qpu_of_phys(v)
            if a != b:
                qpu_pairs.add((a, b))
    return qpu_pairs


def _expected_qpu_pairs(arch: MultiQPUArchitecture) -> set[tuple[int, int]]:
    return {
        (a, b)
        for a, neighbors in enumerate(arch.qpu_graph())
        for b in neighbors
        if a < b
    }


def _undirected_comm_phys_edges(arch: MultiQPUArchitecture) -> set[tuple[int, int]]:
    inter_cm = arch.build_interconnect_coupling_map()
    edges: set[tuple[int, int]] = set()
    for u, v in inter_cm.get_edges():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if arch.qpu_of_phys(a) != arch.qpu_of_phys(b):
            edges.add((a, b))
    return edges


@pytest.mark.parametrize(
    ("n_qpus", "inter_degree"),
    [
        (6, 3),  # odd degree with even n: includes diametric links
        (5, 3),  # odd degree with odd n: no diametric matching available
        (4, 99),  # saturates to complete graph
        (2, 1),  # smallest non-trivial network
    ],
)
def test_degree_d_interconnect_matches_qpu_graph_across_edge_cases(
    n_qpus: int, inter_degree: int
) -> None:
    cfg = MultiQPUConfig(
        n_qpus=n_qpus,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=2,
        intra_topology="line",
        inter_topology="degree_d",
        inter_degree=inter_degree,
    )
    arch = MultiQPUArchitecture(cfg)
    assert _qpu_comm_edge_pairs(arch) == _expected_qpu_pairs(arch)


def test_degree_d_interconnect_honors_zero_degree() -> None:
    cfg = MultiQPUConfig(
        n_qpus=5,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="degree_d",
        inter_degree=0,
    )
    arch = MultiQPUArchitecture(cfg)
    assert _qpu_comm_edge_pairs(arch) == set()


def test_fat_tree_interconnect_matches_qpu_graph_adjacency() -> None:
    cfg = MultiQPUConfig(
        n_qpus=9,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=2,
        intra_topology="line",
        inter_topology="fat_tree",
    )
    arch = MultiQPUArchitecture(cfg)
    assert _qpu_comm_edge_pairs(arch) == _expected_qpu_pairs(arch)


def test_degree_d_interconnect_uses_all_comm_port_pairs_per_qpu_link() -> None:
    cfg = MultiQPUConfig(
        n_qpus=6,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=3,
        intra_topology="line",
        inter_topology="degree_d",
        inter_degree=3,
    )
    arch = MultiQPUArchitecture(cfg)
    comm_per_qpu = cfg.comm_qubits_per_qpu

    # For each adjacent QPU pair in qpu_graph, degree_d should realize a full
    # bipartite comm-port connectivity: comm_per_qpu^2 undirected phys links.
    expected_undirected_links = len(_expected_qpu_pairs(arch)) * (comm_per_qpu**2)
    assert len(_undirected_comm_phys_edges(arch)) == expected_undirected_links


def test_fat_tree_interconnect_uses_single_comm_port_per_qpu_link() -> None:
    cfg = MultiQPUConfig(
        n_qpus=9,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=3,
        intra_topology="line",
        inter_topology="fat_tree",
    )
    arch = MultiQPUArchitecture(cfg)

    # fat_tree intentionally models one representative comm link per QPU edge.
    assert len(_undirected_comm_phys_edges(arch)) == len(_expected_qpu_pairs(arch))


def test_degree_d_interconnect_clamps_negative_degree_to_zero() -> None:
    cfg = MultiQPUConfig(
        n_qpus=6,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=2,
        intra_topology="line",
        inter_topology="degree_d",
        inter_degree=-7,
    )
    arch = MultiQPUArchitecture(cfg)
    assert _qpu_comm_edge_pairs(arch) == set()
    assert _undirected_comm_phys_edges(arch) == set()


def test_global_coupling_map_preserves_isolated_physical_qubits() -> None:
    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    cm = arch.build_coupling_map()

    assert cm.physical_qubits == [0]
    assert cm.size() == cfg.total_physical_qubits()
    assert cm.get_edges() == []


def test_clos_single_comm_port_qpu_graph_matches_ring_fallback_edges() -> None:
    cfg = MultiQPUConfig(
        n_qpus=5,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="clos",
    )
    arch = MultiQPUArchitecture(cfg)

    assert _qpu_comm_edge_pairs(arch) == _expected_qpu_pairs(arch)
    assert _expected_qpu_pairs(arch) == {(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)}


def _expected_grid2d_undirected_edges(
    n: int, rows: int | None, cols: int | None
) -> set[tuple[int, int]]:
    """Expected row-major grid2d edges for tests, independent of implementation."""
    if n <= 1:
        return set()

    import math

    if rows is None and cols is None:
        resolved_rows = max(1, int(math.sqrt(n)))
        resolved_cols = max(1, math.ceil(n / resolved_rows))
    elif rows is None:
        assert cols is not None
        resolved_cols = cols
        resolved_rows = math.ceil(n / resolved_cols)
    elif cols is None:
        resolved_rows = rows
        resolved_cols = math.ceil(n / resolved_rows)
    else:
        resolved_rows = rows
        resolved_cols = cols

    occupied = {divmod(idx, resolved_cols): idx for idx in range(n)}
    edges: set[tuple[int, int]] = set()
    for (row, col), idx in occupied.items():
        for neighbor_coord in ((row, col + 1), (row + 1, col)):
            neighbor = occupied.get(neighbor_coord)
            if neighbor is None:
                continue
            a, b = sorted((idx, neighbor))
            edges.add((a, b))
    return edges


def _grid2d_undirected_edges(
    *, n_local: int, rows: int | None = None, cols: int | None = None
) -> set[tuple[int, int]]:
    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=n_local,
        comm_qubits_per_qpu=0,
        intra_topology="grid2d",
        grid_rows=rows,
        grid_cols=cols,
    )
    arch = MultiQPUArchitecture(cfg)
    edges: set[tuple[int, int]] = set()
    for u, v in arch.build_coupling_map().get_edges():
        if u < v:
            edges.add((u, v))
    return edges


@pytest.mark.parametrize(
    ("n_local", "rows", "cols"),
    [
        (0, None, None),
        (1, None, None),
        (2, None, None),
        (5, None, None),
        (5, 2, None),
        (5, None, 2),
        (5, 1, 10),
        (6, 10, None),
        (10, None, 2),
        (10, 5, 2),
    ],
)
def test_grid2d_edges_match_row_major_grid_for_inferred_and_explicit_dimensions(
    n_local: int, rows: int | None, cols: int | None
) -> None:
    assert _grid2d_undirected_edges(n_local=n_local, rows=rows, cols=cols) == (
        _expected_grid2d_undirected_edges(n_local, rows, cols)
    )


def test_grid2d_rejects_explicit_dimensions_that_do_not_cover_qubits() -> None:
    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=10,
        comm_qubits_per_qpu=0,
        intra_topology="grid2d",
        grid_rows=3,
        grid_cols=2,
    )
    arch = MultiQPUArchitecture(cfg)

    with pytest.raises(ValueError, match=r"grid_rows \* grid_cols must cover"):
        arch.build_coupling_map()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("grid_rows", 0, "grid_rows must be positive"),
        ("grid_rows", -1, "grid_rows must be positive"),
        ("grid_rows", True, "grid_rows must be an integer"),
        ("grid_rows", 1.5, "grid_rows must be an integer"),
        ("grid_cols", 0, "grid_cols must be positive"),
        ("grid_cols", -1, "grid_cols must be positive"),
        ("grid_cols", False, "grid_cols must be an integer"),
        ("grid_cols", "2", "grid_cols must be an integer"),
    ],
)
def test_grid2d_rejects_invalid_grid_dimensions(
    field: str, value: object, match: str
) -> None:
    if field == "grid_rows":
        cfg = MultiQPUConfig(
            n_qpus=1,
            compute_qubits_per_qpu=4,
            comm_qubits_per_qpu=0,
            intra_topology="grid2d",
            grid_rows=value,  # type: ignore[arg-type]
        )
    else:
        cfg = MultiQPUConfig(
            n_qpus=1,
            compute_qubits_per_qpu=4,
            comm_qubits_per_qpu=0,
            intra_topology="grid2d",
            grid_cols=value,  # type: ignore[arg-type]
        )
    arch = MultiQPUArchitecture(cfg)

    with pytest.raises(ValueError, match=match):
        arch.build_coupling_map()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_qpus": 0}, "n_qpus must be positive"),
        ({"n_qpus": -1}, "n_qpus must be positive"),
        ({"n_qpus": True}, "n_qpus must be an integer"),
        ({"compute_qubits_per_qpu": -1}, "compute_qubits_per_qpu must be non-negative"),
        ({"compute_qubits_per_qpu": 1.5}, "compute_qubits_per_qpu must be an integer"),
        ({"comm_qubits_per_qpu": -1}, "comm_qubits_per_qpu must be non-negative"),
        ({"comm_qubits_per_qpu": False}, "comm_qubits_per_qpu must be an integer"),
        ({"intra_topology": "bad"}, "Unknown intra_topology"),
        ({"inter_topology": "bad"}, "Unknown inter_topology"),
    ],
)
def test_architecture_rejects_invalid_structural_config(
    kwargs: dict[str, object], match: str
) -> None:
    cfg = MultiQPUConfig(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        MultiQPUArchitecture(cfg)


@pytest.mark.parametrize("phys", [-1, 4, True, 1.5, "1"])
def test_qpu_of_phys_rejects_invalid_physical_indices(phys: object) -> None:
    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    )

    with pytest.raises(ValueError, match="physical qubit index"):
        arch.qpu_of_phys(phys)  # type: ignore[arg-type]


def test_qpu_of_phys_maps_valid_physical_indices() -> None:
    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    )

    assert [arch.qpu_of_phys(i) for i in range(4)] == [0, 0, 1, 1]


@pytest.mark.parametrize("qpu_id", [-1, 2, True, 1.5, "1"])
def test_block_of_qpu_rejects_invalid_qpu_ids(qpu_id: object) -> None:
    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    )

    with pytest.raises(ValueError, match="qpu_id"):
        arch.block_of_qpu(qpu_id)  # type: ignore[arg-type]


def test_zero_qubit_qpu_preserves_empty_device_behavior() -> None:
    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=0, comm_qubits_per_qpu=0)
    )

    assert arch.n_phys == 0
    assert arch.block_of_qpu(0).compute == []
    assert arch.block_of_qpu(0).comm == []
    assert arch.build_coupling_map().size() == 0
    assert arch.build_intra_coupling_map(0).size() == 0
    assert arch.build_interconnect_coupling_map().size() == 0


@pytest.mark.parametrize("qpu_id", [-1, 2, True, 1.5, "1"])
def test_build_intra_coupling_map_reuses_qpu_id_validation(qpu_id: object) -> None:
    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    )

    with pytest.raises(ValueError, match="qpu_id"):
        arch.build_intra_coupling_map(qpu_id)  # type: ignore[arg-type]


def test_architecture_accepts_integral_subclasses_for_structural_config() -> None:
    class FancyInt(int):
        pass

    cfg = MultiQPUConfig(
        n_qpus=FancyInt(2),
        compute_qubits_per_qpu=FancyInt(1),
        comm_qubits_per_qpu=FancyInt(1),
    )
    arch = MultiQPUArchitecture(cfg)

    assert arch.n_phys == 4
    assert arch.block_of_qpu(FancyInt(1)).compute == [2]
    assert arch.qpu_of_phys(FancyInt(3)) == 1
