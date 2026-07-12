# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import inf, nan

import pytest
from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig
from quport.layout import (
    build_initial_layout,
    choose_comm_logicals,
    choose_comm_logicals_diverse,
    compute_layout_hints,
)

Chooser = Callable[
    [int, list[int], Mapping[tuple[int, int], float], int, int], set[int]
]


def _arch(
    *,
    n_qpus: int = 2,
    compute: int = 1,
    comm: int = 1,
) -> MultiQPUArchitecture:
    return MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=n_qpus,
            compute_qubits_per_qpu=compute,
            comm_qubits_per_qpu=comm,
        )
    )


@pytest.mark.parametrize(
    "chooser", [choose_comm_logicals, choose_comm_logicals_diverse]
)
def test_comm_selection_uses_deterministic_ties_and_fills_empty_ports(
    chooser: Chooser,
) -> None:
    assert chooser(4, [0, 0, 1, 1], {}, 2, 1) == {0, 2}


def test_topk_comm_selection_ranks_by_external_weight() -> None:
    comm = choose_comm_logicals(
        4,
        [0, 0, 1, 1],
        {(0, 2): 1.5, (1, 2): 4.0, (1, 3): 2.0},
        2,
        1,
    )

    assert comm == {1, 2}


def test_diverse_comm_selection_prefers_remote_qpu_coverage() -> None:
    comm = choose_comm_logicals_diverse(
        5,
        [0, 0, 0, 1, 2],
        {
            (0, 3): 10.0,
            (1, 3): 9.0,
            (2, 4): 8.0,
        },
        3,
        2,
    )

    assert {0, 2}.issubset(comm)


def test_build_initial_layout_places_comm_logicals_on_comm_ports() -> None:
    arch = _arch(n_qpus=2, compute=2, comm=1)

    layout = build_initial_layout(arch, 4, [0, 0, 0, 1], {2})

    assert layout == [0, 1, 2, 3]


def test_compute_layout_hints_validates_partition_before_indexing() -> None:
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match="outside the valid QPU range"):
        compute_layout_hints(qc, _arch(), [0, 2])


@pytest.mark.parametrize(
    "n_logical,qpu_of_logical,n_qpus,match",
    [
        (True, [0], 2, "n_logical must be an integer"),
        (-1, [], 2, "n_logical must be non-negative"),
        (2, [0], 2, "qpu_of_logical length"),
        (2, (0, 1), 2, "qpu_of_logical must be a list"),
        (2, [0, 2], 2, "outside the valid QPU range"),
        (2, [0, -1], 2, "outside the valid QPU range"),
        (2, [0, True], 2, "must be an integer QPU index"),
        (2, [0, 1.0], 2, "must be an integer QPU index"),
        (2, [0, 1], True, "n_qpus must be an integer"),
        (2, [0, 1], 0, "n_qpus must be positive"),
    ],
)
def test_build_initial_layout_rejects_invalid_partition_assignments(
    n_logical: object,
    qpu_of_logical: object,
    n_qpus: object,
    match: str,
) -> None:
    arch = _arch(n_qpus=2)
    if n_qpus == 0:
        arch = _arch(n_qpus=1)
        object.__setattr__(arch.cfg, "n_qpus", 0)
    elif n_qpus is True:
        arch = _arch(n_qpus=1)
        object.__setattr__(arch.cfg, "n_qpus", True)

    with pytest.raises(ValueError, match=match):
        build_initial_layout(arch, n_logical, qpu_of_logical, set())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "comm_logicals,match",
    [
        ([0], "comm_logicals must be a set"),
        ({2}, "out-of-range logical index"),
        ({-1}, "out-of-range logical index"),
        ({True}, "must contain integer logical indices"),
        ({1.0}, "must contain integer logical indices"),
    ],
)
def test_build_initial_layout_rejects_invalid_comm_logicals(
    comm_logicals: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_initial_layout(_arch(), 2, [0, 1], comm_logicals)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "chooser", [choose_comm_logicals, choose_comm_logicals_diverse]
)
def test_comm_logical_selection_rejects_invalid_partition(chooser: Chooser) -> None:
    with pytest.raises(ValueError, match="outside the valid QPU range"):
        chooser(2, [0, 2], {(0, 1): 1}, 2, 1)


@pytest.mark.parametrize(
    "chooser", [choose_comm_logicals, choose_comm_logicals_diverse]
)
@pytest.mark.parametrize(
    "slots,match",
    [
        (True, "comm_slots_per_qpu must be an integer"),
        (-1, "comm_slots_per_qpu must be non-negative"),
        (1.0, "comm_slots_per_qpu must be an integer"),
    ],
)
def test_comm_logical_selection_rejects_invalid_slot_counts(
    chooser: Chooser, slots: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        chooser(2, [0, 1], {(0, 1): 1.0}, 2, slots)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "penalty,match",
    [
        (True, "diversity_penalty must be numeric, not boolean"),
        (object(), "diversity_penalty must be numeric"),
        (-0.1, "diversity_penalty must be non-negative"),
        (nan, "diversity_penalty must be finite"),
        (inf, "diversity_penalty must be finite"),
    ],
)
def test_diverse_comm_selection_rejects_invalid_diversity_penalty(
    penalty: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        choose_comm_logicals_diverse(2, [0, 1], {(0, 1): 1.0}, 2, 1, penalty)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "chooser", [choose_comm_logicals, choose_comm_logicals_diverse]
)
@pytest.mark.parametrize(
    "weights,match",
    [
        ([], "weights must be a mapping"),
        ({0: 1.0}, "weights keys must be 2-tuples"),
        ({(0, 1, 2): 1.0}, "weights keys must be 2-tuples"),
        ({(0, True): 1.0}, "weights keys must contain integer"),
        ({(0, -1): 1.0}, "out-of-range logical indices"),
        ({(0, 2): 1.0}, "out-of-range logical indices"),
        ({(0, 1): True}, "not booleans"),
        ({(0, 1): object()}, "weights must be numeric"),
        ({(0, 1): 10**400}, "weights must be numeric"),
        ({(0, 1): nan}, "weights must be finite"),
        ({(0, 1): inf}, "weights must be finite"),
        ({(0, 1): -1.0}, "weights must be non-negative"),
    ],
)
def test_comm_selection_rejects_invalid_weights_even_with_zero_slots(
    chooser: Chooser,
    weights: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        chooser(2, [0, 1], weights, 2, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "chooser", [choose_comm_logicals, choose_comm_logicals_diverse]
)
def test_comm_selection_ignores_zero_weight_and_self_loop_edges(
    chooser: Chooser,
) -> None:
    comm = chooser(2, [0, 1], {(0, 0): 100.0, (0, 1): 0.0}, 2, 1)

    assert comm == {0, 1}


def test_build_initial_layout_reports_compute_overflow_after_valid_input() -> None:
    with pytest.raises(RuntimeError, match="QPU 0 overflow"):
        build_initial_layout(_arch(n_qpus=1, compute=1, comm=1), 2, [0, 0], set())


def test_build_initial_layout_accepts_empty_circuit() -> None:
    assert build_initial_layout(_arch(), 0, [], set()) == []


def test_compute_layout_hints_rejects_unknown_comm_mode() -> None:
    with pytest.raises(ValueError, match="comm_mode must be one of"):
        compute_layout_hints(QuantumCircuit(1), _arch(), [0], comm_mode="bad")


def test_compute_layout_hints_returns_consistent_layout_hints() -> None:
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(1, 2)
    arch = _arch(n_qpus=2, compute=2, comm=1)

    hints = compute_layout_hints(qc, arch, [0, 0, 1], comm_mode="topk")

    assert hints.qpu_of_logical == [0, 0, 1]
    assert hints.comm_logicals == {0, 2}
    assert hints.initial_layout == [2, 0, 5]
