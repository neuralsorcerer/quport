# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from numbers import Integral
from typing import SupportsFloat, SupportsIndex, TypeAlias, cast

from qiskit import QuantumCircuit

from quport._validation import validate_nonnegative_integral

WeightValue: TypeAlias = SupportsFloat | SupportsIndex | str


def _iter_validated_weights(
    weights: Mapping[tuple[int, int], WeightValue], n: int
) -> Iterator[tuple[int, int, float]]:
    """Yield normalized positive weighted edges with clear public errors."""
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping of 2-tuples to numeric values")
    n_value = validate_nonnegative_integral(n, label="n")

    for edge, raw_weight in weights.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("weights keys must be 2-tuples of logical indices")
        i_raw, j_raw = edge
        if type(i_raw) is bool or not isinstance(i_raw, Integral):
            raise ValueError("weights keys must contain integer logical indices")
        if type(j_raw) is bool or not isinstance(j_raw, Integral):
            raise ValueError("weights keys must contain integer logical indices")
        i = int(i_raw)
        j = int(j_raw)
        if i < 0 or j < 0 or i >= n_value or j >= n_value:
            raise ValueError("weights contain out-of-range logical indices")
        if type(raw_weight) is bool:
            raise ValueError("weights must be numeric values, not booleans")
        try:
            weight = float(cast(SupportsFloat | SupportsIndex | str, raw_weight))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("weights must be numeric") from None
        if not math.isfinite(weight):
            raise ValueError("weights must be finite")
        if weight < 0.0:
            raise ValueError("weights must be non-negative")
        if i == j or weight == 0.0:
            continue
        yield i, j, weight


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


def degree(weights: Mapping[tuple[int, int], WeightValue], n: int) -> list[float]:
    """Weighted degree of each node.

    Public callers may pass weights assembled outside :func:`extract_twoq_weights`;
    validate them here so malformed graphs fail with deterministic ``ValueError``
    messages instead of accidental ``IndexError``/``TypeError`` exceptions.
    """
    n_value = validate_nonnegative_integral(n, label="n")
    deg = [0.0] * n_value
    for i, j, weight in _iter_validated_weights(weights, n_value):
        deg[i] += weight
        deg[j] += weight
    return deg


def cut_weight(
    weights: Mapping[tuple[int, int], WeightValue], part: list[int]
) -> float:
    """Total weight of edges crossing partitions."""
    if not isinstance(part, list):
        raise ValueError("part must be a list of integer QPU indices")
    for idx, qpu in enumerate(part):
        if type(qpu) is bool or not isinstance(qpu, Integral):
            raise ValueError(f"part[{idx}] must be an integer QPU index")
        if int(qpu) < 0:
            raise ValueError(f"part[{idx}] must be non-negative")

    cut = 0.0
    for i, j, weight in _iter_validated_weights(weights, len(part)):
        if int(part[i]) != int(part[j]):
            cut += weight
    return cut


def validate_temporal_decay(decay: object, *, label: str = "decay") -> float:
    """Validate and normalize temporal 2Q-weight decay factors."""
    if type(decay) is bool:
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        decay_value = float(cast(SupportsFloat | SupportsIndex | str, decay))
    except (TypeError, ValueError, OverflowError):
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
