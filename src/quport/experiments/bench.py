# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from quport.config import LatencyModel, MultiQPUConfig
from quport.pipeline import benchmark_random_circuits


def run_bench(
    cfg: MultiQPUConfig,
    n_logical: int,
    depth: int,
    trials: int,
    seed: int = 0,
    out_csv: str = "results.csv",
) -> list[dict[str, float | str]]:
    """Programmatic API for running benchmarks."""
    return benchmark_random_circuits(
        cfg,
        n_logical,
        depth,
        trials,
        seed=seed,
        latency=LatencyModel(),
        out_csv=out_csv,
    )
