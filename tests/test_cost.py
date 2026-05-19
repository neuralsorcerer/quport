# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any

import pytest

from quport.config import LatencyModel
from quport.cost import estimate_cost
from quport.metrics import CircuitMetrics

_VALID_LATENCY_KWARGS = {"n_1q": 1, "n_2q": 2, "swaps": 3, "remote_2q": 4}
_LATENCY_FIELDS = (
    "oneq",
    "twoq",
    "swap",
    "epr_gen",
    "classical_rtt",
    "remote_gate_overhead",
)
_COUNT_FIELDS = ("n_1q", "n_2q", "swaps", "remote_2q")
_METRIC_FIELDS = ("swaps", "depth", "size", "n_1q", "n_2q", "remote_2q")
_INVALID_COUNTS = (-1, True, 1.5, "1", None)
_INVALID_DEPTHS = (-1, True, 1.5, "1")
_INVALID_FLOATS = (-1.0, math.nan, math.inf, -math.inf, True, object())


def _metrics(**overrides: object) -> CircuitMetrics:
    values: dict[str, Any] = {
        "swaps": 3,
        "depth": 5,
        "size": 11,
        "n_1q": 7,
        "n_2q": 13,
        "remote_2q": 17,
    }
    values.update(overrides)
    return CircuitMetrics(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _COUNT_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_COUNTS)
def test_latency_model_rejects_invalid_operation_counts(
    field: str, bad_value: object
) -> None:
    kwargs: dict[str, Any] = dict(_VALID_LATENCY_KWARGS)
    kwargs[field] = bad_value

    with pytest.raises(ValueError):
        LatencyModel().estimate_latency(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_depth", _INVALID_DEPTHS)
def test_latency_model_rejects_invalid_depth(bad_depth: object) -> None:
    with pytest.raises(ValueError):
        LatencyModel().estimate_latency(**_VALID_LATENCY_KWARGS, depth=bad_depth)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _LATENCY_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_FLOATS)
def test_latency_model_rejects_invalid_coefficients(
    field: str, bad_value: object
) -> None:
    model = LatencyModel()
    object.__setattr__(model, field, bad_value)

    with pytest.raises(ValueError):
        model.estimate_latency(**_VALID_LATENCY_KWARGS)


def test_latency_model_rejects_non_finite_result() -> None:
    with pytest.raises(ValueError):
        LatencyModel(oneq=1e308).estimate_latency(10, 0, 0, 0)


def test_latency_model_computes_expected_latency() -> None:
    model = LatencyModel(
        oneq=1.0,
        twoq=10.0,
        swap=30.0,
        epr_gen=200.0,
        classical_rtt=20.0,
        remote_gate_overhead=50.0,
    )

    assert model.estimate_latency(1, 2, 3, 4, depth=5) == 1196.0


@pytest.mark.parametrize("field", _METRIC_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_COUNTS)
def test_estimate_cost_rejects_invalid_metrics(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError):
        estimate_cost(_metrics(**{field: bad_value}), LatencyModel())


@pytest.mark.parametrize("field", _LATENCY_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_FLOATS)
def test_estimate_cost_rejects_invalid_coefficients(
    field: str, bad_value: object
) -> None:
    model = LatencyModel()
    object.__setattr__(model, field, bad_value)

    with pytest.raises(ValueError):
        estimate_cost(_metrics(), model)


def test_estimate_cost_rejects_non_finite_components() -> None:
    with pytest.raises(ValueError, match="local cost"):
        estimate_cost(_metrics(n_1q=10), LatencyModel(oneq=1e308))

    with pytest.raises(ValueError, match="remote cost"):
        estimate_cost(_metrics(remote_2q=10), LatencyModel(epr_gen=1e308))

    with pytest.raises(ValueError, match="depth penalty"):
        estimate_cost(_metrics(depth=20, n_2q=0), LatencyModel(twoq=1e308))


def test_estimate_cost_matches_latency_model_formula() -> None:
    metrics = _metrics()
    model = LatencyModel()

    cost = estimate_cost(metrics, model)

    assert cost.local == 227.0
    assert cost.remote == 4590.0
    assert cost.depth_penalty == 5.0
    assert cost.total == 4822.0
    assert cost.total == model.estimate_latency(
        metrics.n_1q,
        metrics.n_2q,
        metrics.swaps,
        metrics.remote_2q,
        depth=metrics.depth,
    )
