# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from quport.architecture import MultiQPUArchitecture
from quport.interaction import cut_weight


@dataclass(frozen=True)
class CircuitMetrics:
    swaps: int
    depth: int
    size: int
    n_1q: int
    n_2q: int
    remote_2q: int


def count_ops(qc: QuantumCircuit) -> Counter[str]:
    c: Counter[str] = Counter()
    for inst in qc.data:
        c[inst.operation.name] += 1
    return c


def compute_metrics(qc: QuantumCircuit, arch: MultiQPUArchitecture) -> CircuitMetrics:
    depth = qc.depth()
    size = qc.size()

    swaps = 0
    n_1q = 0
    n_2q = 0
    remote_2q = 0
    qubit_index: dict[Qubit, int] | None = None
    phys_to_qpu: list[int] | None = None

    # physical qubit indices in the transpiled circuit are qc.qubits indices
    for inst in qc.data:
        if inst.operation.name == "swap":
            swaps += 1
        qubits = inst.qubits
        k = len(qubits)
        if k == 1:
            n_1q += 1
        elif k == 2:
            n_2q += 1
            if qubit_index is None:
                qubit_index = {qubit: idx for idx, qubit in enumerate(qc.qubits)}
                phys_to_qpu = [arch.qpu_of_phys(idx) for idx in range(len(qc.qubits))]
            assert phys_to_qpu is not None
            p0 = qubit_index[qubits[0]]
            p1 = qubit_index[qubits[1]]
            if phys_to_qpu[p0] != phys_to_qpu[p1]:
                remote_2q += 1

    return CircuitMetrics(
        swaps=swaps,
        depth=depth,
        size=size,
        n_1q=n_1q,
        n_2q=n_2q,
        remote_2q=remote_2q,
    )


def compute_cut(weights: Mapping[tuple[int, int], float], part: list[int]) -> float:
    return cut_weight(weights, part)
