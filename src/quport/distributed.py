# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture


def _validate_manifest_int(value: object, *, label: str) -> int:
    """Return a non-negative integer manifest field, rejecting bools."""
    if type(value) is bool or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _validate_manifest_sequence(value: object, *, label: str) -> Sequence[object]:
    """Return a sequence manifest field, rejecting string-like containers."""
    if isinstance(value, str | bytes | bytearray | memoryview) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _json_safe_float(value: float) -> Any:
    """Represent finite floats directly and non-finite floats explicitly."""
    if math.isfinite(value):
        return value
    if math.isnan(value):
        label = "nan"
    elif value > 0:
        label = "inf"
    else:
        label = "-inf"
    return {"type": "float", "value": label}


def _enter_container(value: object, seen: set[int]) -> int:
    """Track containers so self-referential parameter structures fail clearly."""
    object_id = id(value)
    if object_id in seen:
        raise ValueError("remote operation parameters cannot contain cycles")
    seen.add(object_id)
    return object_id


def _json_safe_mapping(value: Mapping[object, object], seen: set[int]) -> Any:
    """Convert mappings while avoiding string-key collisions."""
    object_id = _enter_container(value, seen)
    try:
        converted: dict[str, Any] = {}
        entries: list[list[Any]] = []
        use_entries = False
        for key, item in value.items():
            safe_key = _json_safe_value(key, seen)
            safe_item = _json_safe_value(item, seen)
            entries.append([safe_key, safe_item])

            if isinstance(key, str) and key not in converted and not use_entries:
                converted[key] = safe_item
            else:
                use_entries = True

        if not use_entries:
            return converted
        return {"type": "mapping", "entries": entries}
    finally:
        seen.remove(object_id)


def _json_safe_sequence(value: Sequence[object], seen: set[int]) -> list[Any]:
    """Convert ordered containers to JSON arrays with cycle detection."""
    object_id = _enter_container(value, seen)
    try:
        return [_json_safe_value(item, seen) for item in value]
    finally:
        seen.remove(object_id)


def _json_safe_set(value: Set[object], seen: set[int]) -> Any:
    """Convert unordered containers deterministically without losing their type."""
    object_id = _enter_container(value, seen)
    try:
        items = [_json_safe_value(item, seen) for item in sorted(value, key=repr)]
        return {"type": type(value).__name__, "items": items}
    finally:
        seen.remove(object_id)


def _json_safe_value(value: object, seen: set[int] | None = None) -> Any:
    """Convert instruction metadata to a deterministic JSON-compatible value.

    Qiskit gate parameters are usually numbers or symbolic parameters, but custom
    instructions may carry richer metadata. This helper preserves JSON-native
    values, explicitly encodes non-finite/complex/bytes/container edge cases, and
    falls back to a typed string representation for opaque objects.
    """
    active_seen = set() if seen is None else seen

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return _json_safe_float(value)
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": _json_safe_float(value.real),
            "imag": _json_safe_float(value.imag),
        }
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        return {
            "type": type(value).__name__,
            "encoding": "base64",
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, active_seen)
    if isinstance(value, Set):
        return _json_safe_set(value, active_seen)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _json_safe_sequence(value, active_seen)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_value(item(), active_seen)
        except (TypeError, ValueError):
            pass

    return {"type": type(value).__name__, "repr": str(value)}


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

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-safe representation of this remote operation."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("remote operation name must be a non-empty string")

        q0_phys = _validate_manifest_int(self.q0_phys, label="remote operation q0_phys")
        q1_phys = _validate_manifest_int(self.q1_phys, label="remote operation q1_phys")
        qpu0 = _validate_manifest_int(self.qpu0, label="remote operation qpu0")
        qpu1 = _validate_manifest_int(self.qpu1, label="remote operation qpu1")
        if q0_phys == q1_phys:
            raise ValueError("remote operation physical qubits must be distinct")
        if qpu0 == qpu1:
            raise ValueError("remote operation QPUs must be distinct")

        params = _json_safe_sequence(
            _validate_manifest_sequence(self.params, label="remote operation params"),
            set(),
        )
        clbits = [
            _validate_manifest_int(clbit, label="remote operation clbit index")
            for clbit in _validate_manifest_sequence(
                self.clbits, label="remote operation clbits"
            )
        ]
        return {
            "name": self.name,
            "q0_phys": q0_phys,
            "q1_phys": q1_phys,
            "qpu0": qpu0,
            "qpu1": qpu1,
            "params": params,
            "clbits": clbits,
            "index": _validate_manifest_int(self.index, label="remote operation index"),
        }


@dataclass
class DistributedProgram:
    """A decomposition of a mapped circuit into per-QPU local circuits plus remote ops."""

    local_circuits: dict[int, QuantumCircuit]
    remote_ops: list[RemoteOp]

    def remote_ops_payload(self) -> list[dict[str, Any]]:
        """Return the ordered remote-operation manifest as JSON-safe dictionaries."""
        return [op.to_dict() for op in self.remote_ops]


def write_remote_ops_json(remote_ops: Iterable[RemoteOp], path: str | Path) -> None:
    """Write remote operations as a stable JSON manifest.

    The writer creates parent directories and uses ``allow_nan=False`` so the
    emitted manifest is standards-compliant JSON rather than Python's extended
    NaN/Infinity dialect.
    """
    payload: list[dict[str, Any]] = []
    for idx, op in enumerate(remote_ops):
        if not isinstance(op, RemoteOp):
            raise ValueError(f"remote_ops[{idx}] must be a RemoteOp")
        payload.append(op.to_dict())

    out_path = Path(path)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
