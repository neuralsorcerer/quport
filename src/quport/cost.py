# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from quport.config import LatencyModel
from quport.metrics import CircuitMetrics


@dataclass(frozen=True)
class CostBreakdown:
    total: float
    local: float
    remote: float
    depth_penalty: float


def estimate_cost(metrics: CircuitMetrics, model: LatencyModel) -> CostBreakdown:
    local = (
        model.oneq * metrics.n_1q
        + model.twoq * metrics.n_2q
        + model.swap * metrics.swaps
    )
    remote = metrics.remote_2q * (
        model.epr_gen + model.classical_rtt + model.remote_gate_overhead
    )
    depth_penalty = 0.1 * metrics.depth * model.twoq
    return CostBreakdown(
        total=local + remote + depth_penalty,
        local=local,
        remote=remote,
        depth_penalty=depth_penalty,
    )
