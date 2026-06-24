# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.interaction import (
    cut_weight,
    degree,
    extract_temporal_twoq_weights,
    extract_twoq_weights,
    validate_temporal_decay,
)


class OverflowFloat:
    def __float__(self) -> float:
        raise OverflowError("boom")


class BadFloat:
    def __float__(self) -> float:
        raise ValueError("boom")


def test_extract_twoq_weights_counts_undirected_edges_and_ignores_non_2q_ops() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(2, 0)
    qc.cz(0, 2)
    qc.ccx(0, 1, 2)
    qc.swap(1, 2)
    qc.measure_all()

    assert extract_twoq_weights(qc) == {(0, 2): 2, (1, 2): 1}


def test_extract_temporal_twoq_weights_applies_decay_only_to_2q_sequence() -> None:
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cz(1, 2)
    qc.swap(2, 0)

    weights = extract_temporal_twoq_weights(qc, decay=0.5)

    assert weights == {(0, 1): 1.0, (1, 2): 0.5, (0, 2): 0.25}


def test_degree_and_cut_weight_accept_integral_and_float_like_valid_values() -> None:
    weights = {
        (0, 1): Decimal("1.25"),
        (1, 3): Decimal("1.5"),
    }

    assert degree(weights, 4) == [1.25, 2.75, 0.0, 1.5]
    assert cut_weight(weights, [0, 0, 1, 1]) == 1.5


@pytest.mark.parametrize(
    ("weights", "n", "match"),
    [
        ([], 2, "weights must be a mapping"),
        ({(0, 2): 1.0}, 2, "out-of-range"),
        ({(-1, 1): 1.0}, 2, "out-of-range"),
        ({(0, 1): -1.0}, 2, "non-negative"),
        ({(0, 1): math.inf}, 2, "finite"),
        ({(0, 1): math.nan}, 2, "finite"),
        ({(0, 1): True}, 2, "not booleans"),
        ({(0, 1): object()}, 2, "numeric"),
        ({(0, 1): OverflowFloat()}, 2, "numeric"),
        ({("0", 1): 1.0}, 2, "integer logical indices"),
        ({(False, 1): 1.0}, 2, "integer logical indices"),
        ({(0, 1, 2): 1.0}, 3, "2-tuples"),
        ({"bad": 1.0}, 3, "2-tuples"),
    ],
)
def test_degree_rejects_invalid_weight_graphs(
    weights: Mapping[tuple[Any, ...], object], n: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        degree(weights, n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n", "match"),
    [
        (-1, "n must be non-negative"),
        (True, "n must be a non-negative integer"),
        (1.5, "n must be a non-negative integer"),
    ],
)
def test_degree_rejects_invalid_node_count(n: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        degree({}, n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("part", "match"),
    [
        ((0, 1), "part must be a list"),
        ([0, True], r"part\[1\] must be an integer"),
        ([0, 1.5], r"part\[1\] must be an integer"),
        ([0, -1], r"part\[1\] must be non-negative"),
    ],
)
def test_cut_weight_rejects_invalid_partitions(part: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cut_weight({(0, 1): 1.0}, part)  # type: ignore[arg-type]


def test_cut_weight_validates_weight_indices_against_partition_length() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        cut_weight({(0, 2): 1.0}, [0, 1])


def test_degree_and_cut_weight_ignore_zero_and_self_loop_edges() -> None:
    weights = {(0, 0): 10.0, (0, 1): 0.0, (1, 2): 2.5}

    assert degree(weights, 3) == [0.0, 2.5, 2.5]
    assert cut_weight(weights, [0, 0, 1]) == 2.5


def test_degree_and_cut_weight_handle_empty_inputs() -> None:
    assert degree({}, 0) == []
    assert cut_weight({}, []) == 0.0


@pytest.mark.parametrize(
    ("decay", "match"),
    [
        (True, "numeric, not boolean"),
        (0.0, r"within \(0, 1\]"),
        (-0.1, r"within \(0, 1\]"),
        (1.01, r"within \(0, 1\]"),
        (math.nan, "finite"),
        (math.inf, "finite"),
        (object(), "numeric"),
        (BadFloat(), "numeric"),
        (OverflowFloat(), "numeric"),
    ],
)
def test_validate_temporal_decay_rejects_invalid_values(
    decay: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_temporal_decay(decay, label="temporal_decay")


@pytest.mark.parametrize("decay", [1, 1.0, "0.5", Decimal("0.25")])
def test_validate_temporal_decay_accepts_valid_values(decay: object) -> None:
    assert 0.0 < validate_temporal_decay(decay) <= 1.0
