# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from quport._validation import (
    validate_finite_result,
    validate_nonnegative_finite_float,
    validate_nonnegative_integral,
)
from quport.config import LatencyModel
from quport.metrics import CircuitMetrics


@dataclass(frozen=True)
class CostBreakdown:
    total: float
    local: float
    remote: float
    depth_penalty: float


def estimate_cost(metrics: CircuitMetrics, model: LatencyModel) -> CostBreakdown:
    n_1q = validate_nonnegative_integral(metrics.n_1q, label="n_1q")
    n_2q = validate_nonnegative_integral(metrics.n_2q, label="n_2q")
    swaps = validate_nonnegative_integral(metrics.swaps, label="swaps")
    remote_2q = validate_nonnegative_integral(metrics.remote_2q, label="remote_2q")
    depth = validate_nonnegative_integral(metrics.depth, label="depth")
    validate_nonnegative_integral(metrics.size, label="size")

    oneq = validate_nonnegative_finite_float(model.oneq, label="oneq")
    twoq = validate_nonnegative_finite_float(model.twoq, label="twoq")
    swap = validate_nonnegative_finite_float(model.swap, label="swap")
    epr_gen = validate_nonnegative_finite_float(model.epr_gen, label="epr_gen")
    classical_rtt = validate_nonnegative_finite_float(
        model.classical_rtt, label="classical_rtt"
    )
    remote_gate_overhead = validate_nonnegative_finite_float(
        model.remote_gate_overhead, label="remote_gate_overhead"
    )

    local = validate_finite_result(
        oneq * n_1q + twoq * n_2q + swap * swaps, label="local cost"
    )
    remote = validate_finite_result(
        remote_2q * (epr_gen + classical_rtt + remote_gate_overhead),
        label="remote cost",
    )
    depth_penalty = validate_finite_result(0.1 * depth * twoq, label="depth penalty")
    total = validate_finite_result(local + remote + depth_penalty, label="total cost")
    return CostBreakdown(
        total=total,
        local=local,
        remote=remote,
        depth_penalty=depth_penalty,
    )
