# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture


@dataclass(frozen=True)
class RemoteOp:
    """A remote (inter-QPU) 2-qubit operation placeholder."""

    name: str
    q0_phys: int
    q1_phys: int
    qpu0: int
    qpu1: int
    params: tuple[Any, ...]
    clbits: tuple[int, ...]
    index: int  # global instruction index


@dataclass
class DistributedProgram:
    """A decomposition of a mapped circuit into per-QPU local circuits plus remote ops."""

    local_circuits: dict[int, QuantumCircuit]
    remote_ops: list[RemoteOp]


def _group_qubits_by_qpu_in_operand_order(
    qubits: list[int], qpus: list[int]
) -> tuple[tuple[int, ...], dict[int, list[int]]]:
    """Return participating QPUs and per-QPU qubits preserving operand order."""
    if len(qubits) != len(qpus):
        raise ValueError("qubits and qpus must have the same length")
    qpu_order = tuple(dict.fromkeys(qpus))
    qpu_qubits: dict[int, list[int]] = {}
    for q, qpu in zip(qubits, qpus, strict=True):
        qpu_qubits.setdefault(qpu, []).append(q)
    return qpu_order, qpu_qubits


def _local_cargs_for_qpu(
    local: dict[int, QuantumCircuit], qpu: int, cargs_idx: list[int]
) -> list[Any]:
    """Map source-circuit clbit indices into the target local QPU circuit."""
    if not cargs_idx:
        return []
    qpu_clbits = local[qpu].clbits
    return [qpu_clbits[i] for i in cargs_idx]


def split_into_qpus(
    mapped: QuantumCircuit, arch: MultiQPUArchitecture
) -> DistributedProgram:
    """Split a *mapped* circuit (physical qubits) into per-QPU circuits.

    Notes
    -----
    - This does not *implement* teleportation/entanglement swapping; it produces a program
      representation where inter-QPU gates are extracted as `RemoteOp` events.
    - Local circuits include 1Q and intra-QPU 2Q operations.
    - For remote ops, local circuits will include barriers on the involved QPUs to make
      synchronization explicit for downstream schedulers.
    """
    n_qpus = arch.cfg.n_qpus
    local: dict[int, QuantumCircuit] = {}
    # Create per-QPU circuits with the full physical register for clarity.
    # (You may later shrink them to only used qubits.)
    for q in range(n_qpus):
        local[q] = QuantumCircuit(arch.n_phys, mapped.num_clbits)

    remote_ops: list[RemoteOp] = []

    qindex = {q: i for i, q in enumerate(mapped.qubits)}
    cindex = {c: i for i, c in enumerate(mapped.clbits)}

    for idx, inst in enumerate(mapped.data):
        cargs_idx = [cindex[c] for c in inst.clbits]
        op = inst.operation
        qs = [qindex[q] for q in inst.qubits]

        if not qs:
            if op.name == "barrier":
                for qpu in range(n_qpus):
                    local[qpu].barrier()
            else:
                for qpu in range(n_qpus):
                    local[qpu].append(
                        op, [], _local_cargs_for_qpu(local, qpu, cargs_idx)
                    )
            continue

        op_qpus = [arch.qpu_of_phys(q) for q in qs]

        if op.name == "barrier":
            qpu_order, qpu_qubits_barrier = _group_qubits_by_qpu_in_operand_order(
                qs, op_qpus
            )
            for qpu in qpu_order:
                local[qpu].barrier(*qpu_qubits_barrier[qpu])
            continue

        if len(qs) == 1:
            qpu = op_qpus[0]
            local[qpu].append(
                op,
                [local[qpu].qubits[qs[0]]],
                _local_cargs_for_qpu(local, qpu, cargs_idx),
            )

        elif len(qs) == 2:
            q0, q1 = qs
            qpu0, qpu1 = op_qpus
            if qpu0 == qpu1:
                local[qpu0].append(
                    op,
                    [local[qpu0].qubits[q0], local[qpu0].qubits[q1]],
                    _local_cargs_for_qpu(local, qpu0, cargs_idx),
                )
            else:
                remote_ops.append(
                    RemoteOp(
                        name=op.name,
                        q0_phys=q0,
                        q1_phys=q1,
                        qpu0=qpu0,
                        qpu1=qpu1,
                        params=tuple(getattr(op, "params", [])),
                        clbits=tuple(cargs_idx),
                        index=idx,
                    )
                )
                # add barriers to mark synchronization points
                local[qpu0].barrier(q0)
                local[qpu1].barrier(q1)

        else:
            # multi-qubit ops shouldn't appear if you translated to max_operands=2; keep safe.
            # We conservatively assign to QPU of first qubit if all in same QPU, else mark remote.
            qpu_order, qpu_qubits = _group_qubits_by_qpu_in_operand_order(qs, op_qpus)
            if len(qpu_order) == 1:
                qpu = qpu_order[0]
                local[qpu].append(
                    op,
                    [local[qpu].qubits[q] for q in qs],
                    _local_cargs_for_qpu(local, qpu, cargs_idx),
                )
            else:
                # treat as remote composite operation
                q0_phys = qs[0]
                qpu0 = op_qpus[0]

                remote_idx = next(
                    (i for i, qpu in enumerate(op_qpus[1:], start=1) if qpu != qpu0),
                    None,
                )
                if remote_idx is None:
                    raise ValueError(
                        "remote composite operation must involve another QPU"
                    )
                q1_phys = qs[remote_idx]
                qpu1 = op_qpus[remote_idx]

                remote_ops.append(
                    RemoteOp(
                        name=op.name,
                        q0_phys=q0_phys,
                        q1_phys=q1_phys,
                        qpu0=qpu0,
                        qpu1=qpu1,
                        params=tuple(getattr(op, "params", [])),
                        clbits=tuple(cargs_idx),
                        index=idx,
                    )
                )
                for qpu in qpu_order:
                    local[qpu].barrier(*qpu_qubits[qpu])

    return DistributedProgram(local_circuits=local, remote_ops=remote_ops)
