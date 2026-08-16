# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.hypergraph import (
    PacketDecomposition,
    build_distributable_packets,
    ebit_cost,
    ebit_objective,
    ebit_report,
    ebit_traffic_matrix,
)


def _packet_view(
    decomposition: PacketDecomposition,
) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    return [
        (packet.root, packet.partners, packet.gate_indices)
        for packet in decomposition.packets
    ]


def test_shared_control_collapses_into_one_packet() -> None:
    qc = QuantumCircuit(5)
    qc.h(0)
    for target in range(1, 5):
        qc.cx(0, target)

    decomposition = build_distributable_packets(qc)

    assert _packet_view(decomposition) == [(0, (1, 2, 3, 4), (1, 2, 3, 4))]
    assert decomposition.two_qubit_gates == 4
    assert decomposition.packed_gates() == 4
    assert decomposition.unpackable_gates == ()

    # All four targets on one remote QPU: one cat copy serves every gate.
    assert ebit_cost(decomposition, [0, 1, 1, 1, 1], 2) == 1
    # Spread across three QPUs: one cat copy per distinct destination.
    assert ebit_cost(decomposition, [0, 1, 1, 2, 2], 3) == 2
    # Everything local: no entanglement at all.
    assert ebit_cost(decomposition, [0, 0, 0, 0, 0], 2) == 0


def test_diagonal_rotations_on_the_root_keep_a_packet_open() -> None:
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.rz(0.3, 0)
    qc.t(0)
    qc.barrier(0)
    qc.cx(0, 2)

    decomposition = build_distributable_packets(qc)

    assert _packet_view(decomposition) == [(0, (1, 2), (0, 4))]
    assert ebit_cost(decomposition, [0, 1, 1], 2) == 1


@pytest.mark.parametrize("breaker", ["x", "h", "sx", "measure_like"])
def test_non_diagonal_operations_on_the_root_split_a_packet(breaker: str) -> None:
    qc = QuantumCircuit(3, 1)
    qc.cx(0, 1)
    if breaker == "x":
        qc.x(0)
    elif breaker == "h":
        qc.h(0)
    elif breaker == "sx":
        qc.sx(0)
    else:
        qc.measure(0, 0)
    qc.cx(0, 2)

    decomposition = build_distributable_packets(qc)

    assert len(decomposition.packets) == 2
    assert ebit_cost(decomposition, [0, 1, 1], 2) == 2


def test_target_side_of_a_cx_closes_the_target_packet() -> None:
    """A qubit used as a CX target cannot simultaneously root a live cat copy."""
    qc = QuantumCircuit(3)
    qc.cx(1, 2)  # opens a packet rooted at qubit 1
    qc.cx(0, 1)  # qubit 1 is the target here, so its packet must close
    qc.cx(1, 2)

    decomposition = build_distributable_packets(qc)

    roots = [packet.root for packet in decomposition.packets]
    assert roots.count(1) == 2
    # Qubit 1 needs a fresh cat copy on QPU 1 either side of the break, while
    # the packet rooted at qubit 0 stays local and costs nothing.
    assert ebit_cost(decomposition, [0, 0, 1], 2) == 2


def test_symmetric_gates_are_charged_to_exactly_one_root() -> None:
    qc = QuantumCircuit(3)
    qc.cz(0, 1)
    qc.cz(0, 2)

    decomposition = build_distributable_packets(qc)

    # Both operands of a CZ are diagonal; charging both would double count.
    assert sum(packet.size() for packet in decomposition.packets) == 2
    assert ebit_cost(decomposition, [0, 1, 1], 2) == 1


def test_greedy_symmetric_root_extends_an_open_packet() -> None:
    qc = QuantumCircuit(3)
    qc.cx(1, 0)  # opens a packet rooted at qubit 1
    qc.cz(0, 1)  # both operands diagonal; greedy should reuse qubit 1

    greedy = build_distributable_packets(qc, symmetric_root="greedy")
    lowest = build_distributable_packets(qc, symmetric_root="min_index")

    assert _packet_view(greedy) == [(1, (0,), (0, 1))]
    # Packets are ordered by first gate, so the CX packet comes before the CZ one.
    assert _packet_view(lowest) == [(1, (0,), (0,)), (0, (1,), (1,))]
    assert ebit_cost(greedy, [0, 1, 0], 2) == 1
    assert ebit_cost(lowest, [0, 1, 0], 2) == 2


def test_unpackable_two_qubit_gate_costs_two_ebits() -> None:
    qc = QuantumCircuit(2)
    qc.swap(0, 1)

    decomposition = build_distributable_packets(qc)

    assert decomposition.packets == ()
    assert len(decomposition.unpackable_gates) == 1
    assert ebit_cost(decomposition, [0, 1], 2) == 2
    assert ebit_cost(decomposition, [0, 0], 2) == 0


def test_multi_qubit_gate_spanning_three_qpus_costs_two_round_trips() -> None:
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    decomposition = build_distributable_packets(qc)

    assert len(decomposition.unpackable_gates) == 1
    assert ebit_cost(decomposition, [0, 1, 2], 3) == 4
    assert ebit_cost(decomposition, [0, 0, 1], 3) == 2
    assert ebit_cost(decomposition, [0, 0, 0], 3) == 0


def test_report_counts_baseline_and_reduction() -> None:
    qc = QuantumCircuit(4)
    for target in range(1, 4):
        qc.cx(0, target)

    decomposition = build_distributable_packets(qc)
    report = ebit_report(decomposition, [0, 1, 1, 1], 2)

    assert report.ebits == 1
    assert report.baseline_ebits == 3
    assert report.cut_gates == 3
    assert report.reduction == pytest.approx(1.0 - 1.0 / 3.0)
    assert report.active_packets == 1
    assert report.pair_ebits == (((0, 1), 1),)
    assert report.to_dict()["ebits"] == 1


def test_report_peak_cat_copies_counts_overlapping_windows() -> None:
    qc = QuantumCircuit(4)
    qc.cx(0, 2)  # copy of qubit 0 lands on QPU 1
    qc.cx(1, 3)  # copy of qubit 1 lands on QPU 1, overlapping the first
    qc.cx(0, 2)
    qc.cx(1, 3)

    decomposition = build_distributable_packets(qc)
    report = ebit_report(decomposition, [0, 0, 1, 1], 2)

    assert report.ebits == 2
    assert report.peak_cat_copies == (0, 2)


def test_hop_weighted_objective_scales_with_distance() -> None:
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)

    decomposition = build_distributable_packets(qc)
    part = [0, 1, 2]
    # A path graph 0 - 1 - 2: reaching QPU 2 from QPU 0 costs two hops.
    dist = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]

    count, weighted = ebit_objective(decomposition, part, 3, dist)

    assert count == 2
    assert weighted == pytest.approx(3.0)
    assert ebit_objective(decomposition, part, 3)[1] == pytest.approx(2.0)


def test_traffic_matrix_is_symmetric_and_matches_the_ebit_count() -> None:
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)

    decomposition = build_distributable_packets(qc)
    part = [0, 1, 1, 2]
    traffic = ebit_traffic_matrix(decomposition, part, 3)

    assert traffic == [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert sum(sum(row) for row in traffic) / 2 == ebit_cost(decomposition, part, 3)


def test_empty_circuit_produces_no_packets() -> None:
    decomposition = build_distributable_packets(QuantumCircuit(3))

    assert decomposition.packets == ()
    assert decomposition.unpackable_gates == ()
    assert ebit_cost(decomposition, [0, 0, 0], 1) == 0
    report = ebit_report(decomposition, [0, 0, 0], 1)
    assert report.reduction == 0.0


@pytest.mark.parametrize(
    ("part", "n_qpus", "message"),
    [
        ([0, 0], 2, "part length must match"),
        ([0, 0, 2], 2, "outside the valid QPU range"),
        ([0, 0, -1], 2, "outside the valid QPU range"),
        ([0, 0, True], 2, "must be an integer QPU index"),
    ],
)
def test_ebit_cost_rejects_malformed_partitions(
    part: list[object], n_qpus: int, message: str
) -> None:
    decomposition = build_distributable_packets(QuantumCircuit(3))

    with pytest.raises(ValueError, match=message):
        ebit_cost(decomposition, part, n_qpus)  # type: ignore[arg-type]


def test_ebit_cost_rejects_invalid_qpu_counts() -> None:
    decomposition = build_distributable_packets(QuantumCircuit(2))

    with pytest.raises(ValueError, match="n_qpus must be positive"):
        ebit_cost(decomposition, [0, 0], 0)
    with pytest.raises(ValueError, match="n_qpus must be an integer"):
        ebit_cost(decomposition, [0, 0], 1.5)  # type: ignore[arg-type]


def test_build_distributable_packets_validates_inputs() -> None:
    with pytest.raises(ValueError, match="qc must be a QuantumCircuit"):
        build_distributable_packets(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="symmetric_root must be one of"):
        build_distributable_packets(QuantumCircuit(1), symmetric_root="nope")  # type: ignore[arg-type]


def test_ebit_objective_rejects_malformed_distance_tables() -> None:
    decomposition = build_distributable_packets(QuantumCircuit(2))

    with pytest.raises(ValueError, match="dist dimensions do not match n_qpus"):
        ebit_objective(decomposition, [0, 1], 2, [[0, 1]])
    with pytest.raises(ValueError, match="dist rows must be sequences"):
        ebit_objective(decomposition, [0, 1], 2, [0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dist must contain integer distances"):
        ebit_objective(decomposition, [0, 1], 2, [[0, 1.5], [1.5, 0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-negative distances"):
        ebit_objective(decomposition, [0, 1], 2, [[0, -1], [-1, 0]])
