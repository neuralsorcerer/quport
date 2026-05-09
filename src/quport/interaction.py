# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit


def extract_twoq_weights(qc: QuantumCircuit) -> dict[tuple[int, int], int]:
    """Extract a weighted *undirected* interaction graph from a circuit.

    For each 2-qubit instruction acting on logical qubits (i, j), we increment
    weight[(min(i,j), max(i,j))].

    This works for random circuits and any circuits with 2Q ops (cx, cz, ecr, rxx, etc.).
    """
    weights: dict[tuple[int, int], int] = defaultdict(int)

    # Fast index lookup
    qindex = {q: i for i, q in enumerate(qc.qubits)}

    for inst in qc.data:
        if len(inst.qubits) != 2:
            continue
        i = qindex[inst.qubits[0]]
        j = qindex[inst.qubits[1]]
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        weights[(a, b)] += 1

    return dict(weights)


def degree(weights: Mapping[tuple[int, int], float], n: int) -> list[float]:
    """Weighted degree of each node."""
    deg = [0.0] * n
    for (i, j), w in weights.items():
        deg[i] += float(w)
        deg[j] += float(w)
    return deg


def cut_weight(weights: Mapping[tuple[int, int], float], part: list[int]) -> float:
    """Total weight of edges crossing partitions."""
    cut = 0.0
    for (i, j), w in weights.items():
        if part[i] != part[j]:
            cut += float(w)
    return cut


def validate_temporal_decay(decay: object, *, label: str = "decay") -> float:
    """Validate and normalize temporal 2Q-weight decay factors."""
    if type(decay) is bool:
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        decay_value = float(cast(SupportsFloat | SupportsIndex | str, decay))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(decay_value):
        raise ValueError(f"{label} must be finite")
    if decay_value <= 0.0 or decay_value > 1.0:
        raise ValueError(f"{label} must be within (0, 1]")
    return decay_value


def extract_temporal_twoq_weights(
    qc: QuantumCircuit,
    decay: object = 0.98,
) -> dict[tuple[int, int], float]:
    """Extract a *time-aware* 2Q interaction graph.

    Motivation
    ----------
    Some DQC compilation work observes that circuits often have non-uniform gate
    patterns across time ("hot" regions). A mapping that reduces remote ops
    early can reduce overall latency and decoherence sensitivity.

    This helper assigns higher weights to earlier 2Q interactions with an
    exponential decay:

        w_t = decay ** t

    where t is a monotonically increasing instruction index over 2Q ops.

    Parameters
    ----------
    decay:
        Exponential decay factor in (0,1]. Values closer to 1 approach the
        uniform weighting of :func:`extract_twoq_weights`.

    Returns
    -------
    dict[(i,j)] -> float
        Accumulated time-decayed weights.
    """
    decay_value = validate_temporal_decay(decay)

    weights: dict[tuple[int, int], float] = defaultdict(float)
    qindex = {q: i for i, q in enumerate(qc.qubits)}
    t = 0
    for inst in qc.data:
        if len(inst.qubits) != 2:
            continue
        i = qindex[inst.qubits[0]]
        j = qindex[inst.qubits[1]]
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        weights[(a, b)] += float(decay_value**t)
        t += 1
    return dict(weights)
