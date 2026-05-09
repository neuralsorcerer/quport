# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from quport.pipeline import sweep_topologies


def run_sweep(
    n_logical: int,
    depth: int,
    trials: int,
    seed: int,
    out_csv: str,
) -> None:
    """Programmatic API for sweeping topologies."""
    sweep_topologies(
        n_logical=n_logical, depth=depth, trials=trials, seed=seed, out_csv=out_csv
    )
