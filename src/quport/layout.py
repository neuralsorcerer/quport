# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.interaction import extract_twoq_weights


def _validate_nonnegative_int(value: object, *, label: str) -> int:
    """Return a non-negative integer, rejecting booleans and non-integral values."""
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _validate_positive_int(value: object, *, label: str) -> int:
    """Return a positive integer, rejecting booleans and non-integral values."""
    out = _validate_nonnegative_int(value, label=label)
    if out <= 0:
        raise ValueError(f"{label} must be positive")
    return out


def _validate_nonnegative_float(value: object, *, label: str) -> float:
    """Return a finite non-negative float, rejecting booleans."""
    if type(value) is bool:
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        out = float(cast(SupportsFloat | SupportsIndex | str, value))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    if out < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _validate_layout_inputs(
    n_logical: int,
    qpu_of_logical: list[int],
    n_qpus: int,
    comm_logicals: set[int] | None = None,
) -> tuple[int, int]:
    """Validate logical-to-QPU assignments used for layout construction."""
    n_logical_value = _validate_nonnegative_int(n_logical, label="n_logical")
    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    if not isinstance(qpu_of_logical, list):
        raise ValueError("qpu_of_logical must be a list of integer QPU indices")
    if len(qpu_of_logical) != n_logical_value:
        raise ValueError("qpu_of_logical length must match n_logical")

    for logical_index, qpu in enumerate(qpu_of_logical):
        if type(qpu) is not int:
            raise ValueError(
                f"qpu_of_logical[{logical_index}] must be an integer QPU index"
            )
        if qpu < 0 or qpu >= n_qpus_value:
            raise ValueError(
                f"qpu_of_logical[{logical_index}] is outside the valid QPU range"
            )

    if comm_logicals is not None:
        if not isinstance(comm_logicals, set):
            raise ValueError("comm_logicals must be a set of logical indices")
        for logical_index in comm_logicals:
            if type(logical_index) is not int:
                raise ValueError("comm_logicals must contain integer logical indices")
            if logical_index < 0 or logical_index >= n_logical_value:
                raise ValueError("comm_logicals contains an out-of-range logical index")

    return n_logical_value, n_qpus_value


def _iter_validated_layout_weights(
    weights: Mapping[tuple[int, int], float], n_logical: int
) -> Iterator[tuple[int, int, float]]:
    """Yield validated, positive, non-self-loop interaction weights."""
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping of 2-tuples to numeric values")

    for edge, weight_raw in weights.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("weights keys must be 2-tuples of logical indices")
        i, j = edge
        if type(i) is not int or type(j) is not int:
            raise ValueError("weights keys must contain integer logical indices")
        if i < 0 or j < 0 or i >= n_logical or j >= n_logical:
            raise ValueError("weights contain out-of-range logical indices")
        if type(weight_raw) is bool:
            raise ValueError("weights must be numeric values, not booleans")
        try:
            weight = float(cast(SupportsFloat | SupportsIndex | str, weight_raw))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("weights must be numeric") from None
        if not math.isfinite(weight):
            raise ValueError("weights must be finite")
        if weight < 0.0:
            raise ValueError("weights must be non-negative")
        if i == j or weight == 0.0:
            continue
        yield i, j, weight


@dataclass(frozen=True)
class LayoutHints:
    qpu_of_logical: list[int]
    comm_logicals: set[int]
    initial_layout: list[int]


def choose_comm_logicals(
    n_logical: int,
    qpu_of_logical: list[int],
    weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    comm_slots_per_qpu: int,
) -> set[int]:
    """Choose logical qubits to place on communication qubits (ports).

    Heuristic: for each logical qubit, compute the total interaction weight to qubits
    assigned to *other* QPUs ("external score"). In each QPU, select top-K logicals.
    """
    n_logical_value, n_qpus_value = _validate_layout_inputs(
        n_logical, qpu_of_logical, n_qpus
    )
    slots = _validate_nonnegative_int(comm_slots_per_qpu, label="comm_slots_per_qpu")

    if slots == 0:
        # Still validate weights so invalid public inputs are rejected consistently.
        list(_iter_validated_layout_weights(weights, n_logical_value))
        return set()

    external = [0.0] * n_logical_value
    for i, j, weight in _iter_validated_layout_weights(weights, n_logical_value):
        if qpu_of_logical[i] != qpu_of_logical[j]:
            external[i] += weight
            external[j] += weight

    per_qpu: dict[int, list[int]] = defaultdict(list)
    for i, q in enumerate(qpu_of_logical):
        per_qpu[q].append(i)

    comm: set[int] = set()
    for q in range(n_qpus_value):
        nodes = per_qpu.get(q, [])
        nodes.sort(key=lambda i: (external[i], -i), reverse=True)
        for i in nodes[:slots]:
            comm.add(i)
    return comm


def choose_comm_logicals_diverse(
    n_logical: int,
    qpu_of_logical: list[int],
    weights: Mapping[tuple[int, int], float],
    n_qpus: int,
    comm_slots_per_qpu: int,
    diversity_penalty: float = 0.6,
) -> set[int]:
    """Choose comm/port logicals with *neighbor diversity*.

    Baseline :func:`choose_comm_logicals` picks the top-K boundary qubits by total
    external traffic. That can create a pathological situation where all selected
    comm logicals primarily talk to the *same* remote QPU, causing port contention.

    This diverse variant greedily selects comm logicals while encouraging coverage
    across *different* remote QPUs.

    Implementation
    --------------
    For each logical qubit i, compute external weights to each remote QPU:
        ext[i][q] = sum_w over edges (i,j) where qpu(j)==q and q!=qpu(i).

    Then for each local QPU, select comm_slots_per_qpu logical qubits by maximizing:
        score(i) = total_external(i) - diversity_penalty * overlap(i)

    where overlap(i) is the maximum ext[i][q] over remote QPUs already covered by
    previously selected comm logicals in the same QPU.

    This is still heuristic, but empirically improves robustness on random circuits.
    """
    n_logical_value, n_qpus_value = _validate_layout_inputs(
        n_logical, qpu_of_logical, n_qpus
    )
    slots = _validate_nonnegative_int(comm_slots_per_qpu, label="comm_slots_per_qpu")
    penalty = _validate_nonnegative_float(diversity_penalty, label="diversity_penalty")

    if slots == 0:
        # Still validate weights so invalid public inputs are rejected consistently.
        list(_iter_validated_layout_weights(weights, n_logical_value))
        return set()

    # ext_by_qpu[i][remote_qpu] = weight
    ext_by_qpu: list[dict[int, float]] = [
        defaultdict(float) for _ in range(n_logical_value)
    ]
    total_ext = [0.0] * n_logical_value

    for i, j, weight in _iter_validated_layout_weights(weights, n_logical_value):
        qi, qj = qpu_of_logical[i], qpu_of_logical[j]
        if qi != qj:
            ext_by_qpu[i][qj] += weight
            ext_by_qpu[j][qi] += weight
            total_ext[i] += weight
            total_ext[j] += weight

    per_qpu: dict[int, list[int]] = defaultdict(list)
    for i, q in enumerate(qpu_of_logical):
        per_qpu[q].append(i)

    comm: set[int] = set()
    for q in range(n_qpus_value):
        nodes = per_qpu.get(q, [])
        if not nodes:
            continue

        selected: list[int] = []
        covered_remote: set[int] = set()

        for _ in range(slots):
            best = None
            best_score = None
            for i in nodes:
                if i in selected:
                    continue
                if total_ext[i] <= 0.0:
                    continue

                overlap = 0.0
                if covered_remote:
                    overlap = max(
                        (ext_by_qpu[i].get(r, 0.0) for r in covered_remote),
                        default=0.0,
                    )

                score = total_ext[i] - penalty * overlap
                if best_score is None or score > best_score:
                    best_score = score
                    best = i
                elif (
                    best_score is not None and score == best_score and best is not None
                ):
                    if i < best:
                        best = i

            if best is None:
                break
            selected.append(best)
            # mark the strongest remote neighbor as covered, with deterministic ties
            best_logical = best
            if ext_by_qpu[best_logical]:
                remote_best = max(
                    ext_by_qpu[best_logical],
                    key=lambda r: (ext_by_qpu[best_logical][r], -r),
                )
                covered_remote.add(remote_best)

        if len(selected) < slots:
            # Communication ports are part of the physical capacity advertised by
            # MultiQPUConfig.capacity_per_qpu().  Even when there is no external
            # traffic (or all candidates have zero external score), fill the
            # remaining port slots deterministically so dense local partitions do
            # not overflow the compute-only region during initial-layout creation.
            remaining = [i for i in nodes if i not in selected]
            remaining.sort(key=lambda i: (total_ext[i], -i), reverse=True)
            selected.extend(remaining[: slots - len(selected)])

        for i in selected:
            comm.add(i)

    return comm


def build_initial_layout(
    arch: MultiQPUArchitecture,
    n_logical: int,
    qpu_of_logical: list[int],
    comm_logicals: set[int],
) -> list[int]:
    """Map logical qubits to physical qubits given partition + comm selections."""
    n_logical_value, _n_qpus = _validate_layout_inputs(
        n_logical, qpu_of_logical, arch.cfg.n_qpus, comm_logicals=comm_logicals
    )
    layout = [-1] * n_logical_value
    per_qpu: dict[int, list[int]] = defaultdict(list)
    for logical_index in range(n_logical_value):
        per_qpu[qpu_of_logical[logical_index]].append(logical_index)

    for qpu_id, logicals in per_qpu.items():
        blk = arch.block_of_qpu(qpu_id)
        comm_phys = blk.comm
        comp_phys = blk.compute

        comm_l = [logical for logical in logicals if logical in comm_logicals]
        comp_l = [logical for logical in logicals if logical not in comm_logicals]

        # truncate if needed
        comm_l = comm_l[: len(comm_phys)]
        # ensure the rest are compute logicals
        used_comm = set(comm_l)
        comp_l = [logical for logical in logicals if logical not in used_comm]

        if len(comp_l) > len(comp_phys):
            raise RuntimeError(
                f"QPU {qpu_id} overflow: {len(comp_l)} logicals > {len(comp_phys)} compute qubits"
            )

        for logical, physical in zip(comm_l, comm_phys, strict=False):
            layout[logical] = physical
        for logical, physical in zip(comp_l, comp_phys, strict=False):
            layout[logical] = physical

    if any(x < 0 for x in layout):
        raise RuntimeError("Initial layout incomplete")

    return layout


def naive_layout(n_logical: int) -> list[int]:
    """Trivial initial layout: logical i -> physical i."""
    return list(range(n_logical))


def compute_layout_hints(
    qc: QuantumCircuit,
    arch: MultiQPUArchitecture,
    qpu_of_logical: list[int],
    comm_mode: str = "topk",
) -> LayoutHints:
    """Compute partition-aware layout hints.

    Parameters
    ----------
    comm_mode:
        - "topk": pick comm logicals by total external traffic (baseline)
        - "diverse": diversity-aware comm selection to reduce port contention
    """
    weights = extract_twoq_weights(qc)

    if comm_mode == "diverse":
        comm = choose_comm_logicals_diverse(
            n_logical=qc.num_qubits,
            qpu_of_logical=qpu_of_logical,
            weights=weights,
            n_qpus=arch.cfg.n_qpus,
            comm_slots_per_qpu=arch.cfg.comm_qubits_per_qpu,
        )
    elif comm_mode == "topk":
        comm = choose_comm_logicals(
            n_logical=qc.num_qubits,
            qpu_of_logical=qpu_of_logical,
            weights=weights,
            n_qpus=arch.cfg.n_qpus,
            comm_slots_per_qpu=arch.cfg.comm_qubits_per_qpu,
        )
    else:
        raise ValueError("comm_mode must be one of: topk, diverse")

    init_layout = build_initial_layout(arch, qc.num_qubits, qpu_of_logical, comm)
    return LayoutHints(
        qpu_of_logical=qpu_of_logical, comm_logicals=comm, initial_layout=init_layout
    )
