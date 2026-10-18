# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, Qubit

from quport.architecture import MultiQPUArchitecture
from quport.interaction import WeightValue, cut_weight


@dataclass(frozen=True)
class CircuitMetrics:
    swaps: int
    depth: int
    size: int
    n_1q: int
    n_2q: int
    remote_2q: int


def count_ops(qc: QuantumCircuit) -> Counter[str]:
    """Tally ``qc``'s instructions by name, including those in control flow.

    A control-flow operation counts once itself, and every instruction in each
    of its bodies counts too, so a SWAP that routing placed inside a branch is
    tallied like one outside it.
    """
    c: Counter[str] = Counter()
    for inst in qc.data:
        c[inst.operation.name] += 1
        if isinstance(inst.operation, ControlFlowOp):
            for block in inst.operation.blocks:
                c.update(count_ops(block))
    return c


def compute_metrics(qc: QuantumCircuit, arch: MultiQPUArchitecture) -> CircuitMetrics:
    """Gate counts of a physical circuit, with ``depth`` and ``size`` from Qiskit.

    Instructions inside control-flow bodies are counted on the qubits the block
    applies them to -- once each, in every branch -- rather than the block
    itself, so wrapping gates in a ``box`` or a condition does not hide them.
    """
    depth = qc.depth()
    size = qc.size()

    swaps = 0
    n_1q = 0
    n_2q = 0
    remote_2q = 0
    qpu_at: dict[int, int] = {}

    def qpu_of(position: int) -> int:
        if position not in qpu_at:
            qpu_at[position] = arch.qpu_of_phys(position)
        return qpu_at[position]

    def walk(circuit: QuantumCircuit, positions: Sequence[int]) -> None:
        nonlocal swaps, n_1q, n_2q, remote_2q
        qubit_index: dict[Qubit, int] = dict(zip(circuit.qubits, positions))
        for inst in circuit.data:
            operation = inst.operation
            # Compiler directives (barriers) consume no runtime resources;
            # qiskit's own depth()/size() exclude them with the same check.
            if getattr(operation, "_directive", False):
                continue
            qubits = [qubit_index[qubit] for qubit in inst.qubits]
            if isinstance(operation, ControlFlowOp):
                for block in operation.blocks:
                    walk(block, qubits)
                continue
            if operation.name == "swap":
                swaps += 1
            k = len(qubits)
            if k == 1:
                n_1q += 1
            elif k == 2:
                n_2q += 1
                if qpu_of(qubits[0]) != qpu_of(qubits[1]):
                    remote_2q += 1

    # physical qubit indices in the transpiled circuit are qc.qubits indices
    walk(qc, range(len(qc.qubits)))

    return CircuitMetrics(
        swaps=swaps,
        depth=depth,
        size=size,
        n_1q=n_1q,
        n_2q=n_2q,
        remote_2q=remote_2q,
    )


def compute_cut(
    weights: Mapping[tuple[int, int], WeightValue], part: list[int]
) -> float:
    return cut_weight(weights, part)
