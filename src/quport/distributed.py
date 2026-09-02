# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import json
import math
import os
from collections import deque
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, TypeAlias

from qiskit import QuantumCircuit, QuantumRegister

from quport.architecture import MultiQPUArchitecture

_PathLike: TypeAlias = str | os.PathLike[str]


def _validate_manifest_int(value: object, *, label: str) -> int:
    """Return a non-negative integer manifest field, rejecting bools."""
    if type(value) is bool or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _validate_optional_manifest_int(value: object, *, label: str) -> int | None:
    """Return a non-negative integer manifest field, or None when unset."""
    if value is None:
        return None
    return _validate_manifest_int(value, label=label)


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
        except (TypeError, ValueError, OverflowError):
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

    # Position of this operation's marker barrier among all barriers of the
    # named QPU's local program.  Barriers are the only thing an emitted QASM
    # file carries to say *where* a remote operation belongs, and a routed
    # program may list them in a different order than the manifest -- barriers
    # on disjoint qubits commute, so rebuilding the circuit from its DAG can
    # reorder them.  Pairing by position is therefore wrong; these fields make
    # the pairing explicit.
    qpu0_marker: int | None = None
    qpu1_marker: int | None = None

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
            "qpu0_marker": _validate_optional_manifest_int(
                self.qpu0_marker, label="remote operation qpu0_marker"
            ),
            "qpu1_marker": _validate_optional_manifest_int(
                self.qpu1_marker, label="remote operation qpu1_marker"
            ),
        }


@dataclass
class DistributedProgram:
    """A decomposition of a mapped circuit into per-QPU local circuits plus remote ops."""

    local_circuits: dict[int, QuantumCircuit]
    remote_ops: list[RemoteOp]

    def remote_ops_payload(self) -> list[dict[str, Any]]:
        """Return the ordered remote-operation manifest as JSON-safe dictionaries."""
        return [op.to_dict() for op in self.remote_ops]


def _coerce_output_path(path: object, *, label: str) -> Path:
    """Return a filesystem path, rejecting ambiguous non-path objects."""
    if not isinstance(path, str | os.PathLike):
        raise ValueError(f"{label} must be a filesystem path")
    return Path(path)


def write_remote_ops_json(remote_ops: Iterable[RemoteOp], path: _PathLike) -> None:
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

    out_path = _coerce_output_path(path, label="path")
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_distributed_program(
    program: DistributedProgram,
    path: _PathLike,
    *,
    include_empty_circuits: bool = True,
) -> dict[str, Path]:
    """Write a distributed program bundle to a directory.

    The bundle contains one OpenQASM 3 file per local QPU circuit plus a
    standards-compliant ``remote_ops.json`` manifest.  The returned mapping uses
    stable artifact keys (``qpu_<id>`` and ``remote_ops``) so callers can report
    or post-process generated files without re-deriving filenames.
    """
    if not isinstance(program, DistributedProgram):
        raise ValueError("program must be a DistributedProgram")
    if not isinstance(include_empty_circuits, bool):
        raise ValueError("include_empty_circuits must be a boolean")

    from qiskit import qasm3

    out_dir = _coerce_output_path(path, label="path")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError("path must be a directory or a path that does not exist")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[int, QuantumCircuit]] = []
    for qpu, circuit in program.local_circuits.items():
        qpu_id = _validate_manifest_int(qpu, label="local circuit QPU id")
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError(f"local_circuits[{qpu!r}] must be a QuantumCircuit")
        entries.append((qpu_id, circuit))

    written: dict[str, Path] = {}
    for qpu_id, circuit in sorted(entries, key=lambda item: item[0]):
        if not include_empty_circuits and len(circuit.data) == 0:
            continue
        qpu_path = out_dir / f"qpu_{qpu_id}.qasm"
        qpu_path.write_text(qasm3.dumps(circuit), encoding="utf-8")
        written[f"qpu_{qpu_id}"] = qpu_path

    remote_path = out_dir / "remote_ops.json"
    write_remote_ops_json(program.remote_ops, remote_path)
    written["remote_ops"] = remote_path
    return written


#: Prefix of the label QuPort puts on the barriers that mark remote operations.
#:
#: Barrier labels are transpiler metadata: they survive routing, they are
#: remapped along with the qubit they sit on, and they are not emitted by the
#: OpenQASM writers. That makes them the right carrier for the one thing a
#: routed program cannot otherwise tell you -- which physical qubit a given
#: remote operation ended up on.
REMOTE_BARRIER_LABEL_PREFIX = "quport_remote_"


def remote_barrier_label(ordinal: int) -> str:
    """Return the barrier label marking remote operation ``ordinal``."""
    return f"{REMOTE_BARRIER_LABEL_PREFIX}{_validate_manifest_int(ordinal, label='remote operation ordinal')}"


def _remote_barrier_ordinal(operation: object) -> int | None:
    """Return the remote-op ordinal a barrier marks, or None if it marks none."""
    if not getattr(operation, "_directive", False):
        return None
    label = getattr(operation, "label", None)
    if not isinstance(label, str) or not label.startswith(REMOTE_BARRIER_LABEL_PREFIX):
        return None
    suffix = label[len(REMOTE_BARRIER_LABEL_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def remap_remote_ops_to_routed(
    remote_ops: Sequence[RemoteOp],
    local_routed: Mapping[int, QuantumCircuit],
) -> list[RemoteOp]:
    """Re-express remote operations in the labelling of *routed* local programs.

    Why this is needed
    ------------------
    :func:`split_into_qpus` records each remote operation's physical qubits as
    they stand in the circuit it split. Routing each local program afterwards
    can permute qubits inside a QPU -- it always does unless the intra-QPU
    topology is a clique -- so those indices stop pointing at the state they
    named. Shipping the original manifest next to routed programs would tell a
    consumer to wire the remote gate to the wrong qubit.

    The routed circuits already hold the answer: the transpiler remaps the
    synchronization barriers along with everything else, so the barrier marking
    a remote operation sits on exactly the qubit that operation must use. This
    function reads those barriers back, identifying them by the label
    :func:`remote_barrier_label` attached at split time.

    Returns
    -------
    list[RemoteOp]
        The same operations, in the same order, with ``q0_phys`` and ``q1_phys``
        replaced by their post-routing positions. Every other field, including
        the global instruction ``index``, is unchanged.

    Raises
    ------
    ValueError
        If a routed program is missing, or a remote operation's marker barrier
        cannot be found -- which would mean the routed programs and the manifest
        no longer describe the same computation.
    """
    if not isinstance(local_routed, Mapping):
        raise ValueError("local_routed must be a mapping of QPU id to circuit")

    # (ordinal, qpu) -> (routed physical qubits in operand order, barrier position)
    markers: dict[tuple[int, int], tuple[list[int], int]] = {}
    for qpu, circuit in local_routed.items():
        qpu_id = _validate_manifest_int(qpu, label="local circuit QPU id")
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError(f"local_routed[{qpu!r}] must be a QuantumCircuit")
        positions = {qubit: index for index, qubit in enumerate(circuit.qubits)}
        seen_barriers = 0
        for instruction in circuit.data:
            if not getattr(instruction.operation, "_directive", False):
                continue
            position = seen_barriers
            seen_barriers += 1
            ordinal = _remote_barrier_ordinal(instruction.operation)
            if ordinal is None:
                continue
            key = (ordinal, qpu_id)
            if key in markers:
                raise ValueError(
                    f"remote operation {ordinal} is marked twice on QPU {qpu_id}"
                )
            markers[key] = (
                [positions[qubit] for qubit in instruction.qubits],
                position,
            )

    out: list[RemoteOp] = []
    for ordinal, op in enumerate(remote_ops):
        if not isinstance(op, RemoteOp):
            raise ValueError(f"remote_ops[{ordinal}] must be a RemoteOp")
        # The marker's leading qubit is the operation's operand on that QPU:
        # split_into_qpus emits the barrier over that QPU's operands in operand
        # order, and both q0_phys and q1_phys are the first such operand.
        qubit0, marker0 = _routed_marker(markers, ordinal, op.qpu0)
        qubit1, marker1 = _routed_marker(markers, ordinal, op.qpu1)
        out.append(
            RemoteOp(
                name=op.name,
                q0_phys=qubit0,
                q1_phys=qubit1,
                qpu0=op.qpu0,
                qpu1=op.qpu1,
                params=op.params,
                clbits=op.clbits,
                index=op.index,
                qpu0_marker=marker0,
                qpu1_marker=marker1,
            )
        )
    return out


def _routed_marker(
    markers: Mapping[tuple[int, int], tuple[list[int], int]], ordinal: int, qpu: int
) -> tuple[int, int]:
    """Return the routed qubit and barrier position for one side of an operation."""
    entry = markers.get((ordinal, qpu))
    if entry is None or not entry[0]:
        raise ValueError(
            f"routed program for QPU {qpu} has no marker for remote operation "
            f"{ordinal}; the manifest and the routed circuits are out of step"
        )
    qubits, position = entry
    return qubits[0], position


def reassemble_distributed_program(
    mapped: QuantumCircuit,
    local_routed: Mapping[int, QuantumCircuit],
    remote_ops: Sequence[RemoteOp],
    arch: MultiQPUArchitecture,
    *,
    restore_layout: bool = True,
) -> QuantumCircuit:
    """Merge per-QPU programs and a remote-op manifest back into one circuit.

    This is the inverse of :func:`split_into_qpus` and the check that the pieces
    a distributed compile emits still describe the circuit they came from. The
    reassembled circuit is not meant to be executed -- the whole point of
    distributed compilation is that these programs run on separate devices --
    but it is directly comparable with the mapped circuit, which is what
    :func:`quport.protocol.verify_distributed_program` does with it.

    Ordering
    --------
    A distributed program is a **partial** order, not a linear one. Within a QPU
    the constraint is per *qubit*: two instructions touching disjoint qubits may
    run in either order, and routing routinely emits them in an order that
    differs from the manifest's. Merging therefore follows qubit dataflow --
    an instruction runs once it is first in line on every qubit it touches --
    and a remote operation runs once its marker leads on both sides.

    That is also the contract a consumer of the artifacts has to honour: reading
    each program strictly linearly can deadlock, because two QPUs can list the
    same pair of remote operations in opposite orders when they sit on disjoint
    qubits.

    Parameters
    ----------
    mapped:
        The circuit the programs were split from. Only its width and the
        operations named by ``remote_ops[k].index`` are used, which is how
        parameters and custom gates survive the round trip exactly.
    local_routed:
        Per-QPU programs, routed or not. Both are accepted; an unrouted program
        simply has an identity layout.
    remote_ops:
        The manifest matching ``local_routed``. Pass
        :attr:`~quport.compiler.DistributedCompileResult.routed_remote_ops` for
        routed programs and ``program.remote_ops`` for unrouted ones.
    restore_layout:
        Append the swaps that undo each QPU's routing permutation, so the result
        ends in the mapped circuit's qubit labelling. Without this the output is
        correct only up to that permutation.

    Raises
    ------
    ValueError
        If a marker is missing, or if the programs impose contradictory orders
        on the remote operations -- a genuine inconsistency rather than a
        scheduling choice.
    """
    if not isinstance(mapped, QuantumCircuit):
        raise ValueError("mapped must be a QuantumCircuit")
    if not isinstance(local_routed, Mapping):
        raise ValueError("local_routed must be a mapping of QPU id to circuit")
    if not isinstance(arch, MultiQPUArchitecture):
        raise ValueError("arch must be a MultiQPUArchitecture")

    width = max(arch.n_phys, len(mapped.qubits))
    out = QuantumCircuit(QuantumRegister(width, "q"))
    # Carry the classical side across, so measurements and conditioned
    # operations land on the bits they named in the circuit that was split.
    if mapped.clbits:
        out.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        out.add_register(creg)
    clbit_at = {clbit: index for index, clbit in enumerate(mapped.clbits)}

    qpus = sorted(local_routed)
    data: dict[int, list[Any]] = {}
    index_of: dict[int, dict[Any, int]] = {}
    queues: dict[int, dict[int, deque[int]]] = {}
    retired: dict[int, list[bool]] = {}
    marker_at: dict[tuple[int, int], int] = {}
    marker_qubits: dict[tuple[int, int], list[int]] = {}
    mapped_at = {qubit: index for index, qubit in enumerate(mapped.qubits)}

    def carried_clbits(instruction: Any) -> list[Any]:
        """Map an instruction's classical arguments onto the merged circuit."""
        if not instruction.clbits:
            return []
        try:
            return [out.clbits[clbit_at[clbit]] for clbit in instruction.clbits]
        except KeyError:  # pragma: no cover - defensive
            raise ValueError(
                "a per-QPU program uses classical bits the mapped circuit "
                "does not have"
            ) from None

    for qpu in qpus:
        circuit = local_routed[qpu]
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError(f"local_routed[{qpu!r}] must be a QuantumCircuit")
        instructions = list(circuit.data)
        positions = {qubit: index for index, qubit in enumerate(circuit.qubits)}
        per_qubit: dict[int, deque[int]] = {}
        for index, instruction in enumerate(instructions):
            ordinal = _remote_barrier_ordinal(instruction.operation)
            if ordinal is not None:
                marker_at[(ordinal, qpu)] = index
                marker_qubits[(ordinal, qpu)] = [
                    positions[qubit] for qubit in instruction.qubits
                ]
            for qubit in instruction.qubits:
                per_qubit.setdefault(positions[qubit], deque()).append(index)
        data[qpu] = instructions
        index_of[qpu] = positions
        queues[qpu] = per_qubit
        retired[qpu] = [False] * len(instructions)

    def leads(qpu: int, index: int) -> bool:
        """True when this instruction is first in line on every qubit it uses."""
        return all(
            queues[qpu][index_of[qpu][qubit]][0] == index
            for qubit in data[qpu][index].qubits
        )

    def retire(qpu: int, index: int) -> None:
        retired[qpu][index] = True
        for qubit in data[qpu][index].qubits:
            queues[qpu][index_of[qpu][qubit]].popleft()

    remaining = {qpu: sum(1 for inst in data[qpu] if inst.qubits) for qpu in qpus}
    pending = set(range(len(remote_ops)))

    while any(remaining.values()) or pending:
        progressed = False

        for qpu in qpus:
            for index, instruction in enumerate(data[qpu]):
                if retired[qpu][index] or not instruction.qubits:
                    continue
                if _remote_barrier_ordinal(instruction.operation) is not None:
                    continue
                if not leads(qpu, index):
                    continue
                if not getattr(instruction.operation, "_directive", False):
                    out.append(
                        instruction.operation,
                        [out.qubits[index_of[qpu][q]] for q in instruction.qubits],
                        carried_clbits(instruction),
                    )
                retire(qpu, index)
                remaining[qpu] -= 1
                progressed = True

        for ordinal in sorted(pending):
            op = remote_ops[ordinal]
            source = mapped.data[op.index]
            # An operation is charged to the QPUs its own operands sit on, which
            # is the set `split_into_qpus` emitted markers for. Reading it from
            # the operation rather than from the manifest's two endpoints is what
            # lets an operation on three or more qubits be rebuilt at all.
            source_qpus = [
                arch.qpu_of_phys(mapped_at[qubit]) for qubit in source.qubits
            ]
            sides: list[tuple[int, int]] = []
            for qpu in dict.fromkeys(source_qpus):
                marker = marker_at.get((ordinal, qpu))
                if marker is None:
                    raise ValueError(
                        f"program for QPU {qpu} has no marker for remote "
                        f"operation {ordinal}"
                    )
                sides.append((qpu, marker))
            if any(
                retired[qpu][marker] or not leads(qpu, marker) for qpu, marker in sides
            ):
                continue
            operands = _remote_operands(ordinal, op, source_qpus, marker_qubits)
            out.append(
                source.operation,
                [out.qubits[operand] for operand in operands],
                carried_clbits(source),
            )
            for qpu, marker in sides:
                retire(qpu, marker)
                remaining[qpu] -= 1
            pending.discard(ordinal)
            progressed = True

        if not progressed:
            raise ValueError(
                "per-QPU programs impose contradictory orders on their remote "
                "operations; the programs and the manifest are out of step"
            )

    if restore_layout:
        _append_layout_restoration(out, local_routed, arch, width)
    return out


def _remote_operands(
    ordinal: int,
    op: RemoteOp,
    source_qpus: Sequence[int],
    marker_qubits: Mapping[tuple[int, int], list[int]],
) -> list[int]:
    """Rebuild a remote operation's operands from the markers that name them.

    :func:`split_into_qpus` emits one marker per participating QPU, over that
    QPU's operands in operand order. Walking the operation's operand-to-QPU
    sequence and taking each QPU's marked qubits in turn therefore recovers the
    whole operand list -- including the third and later operands of a wide
    operation, which the two-endpoint manifest does not name.

    The markers are also the only record of where routing left each operand, so
    the manifest's own endpoints are checked against them: a manifest paired
    with programs it was not written for names the wrong qubits, and that is a
    mistake worth reporting rather than merging.
    """
    cursors: dict[int, int] = {}
    operands: list[int] = []
    for qpu in source_qpus:
        marked = marker_qubits[(ordinal, qpu)]
        cursor = cursors.get(qpu, 0)
        if cursor >= len(marked):
            raise ValueError(
                f"the marker for remote operation {ordinal} on QPU {qpu} names "
                f"{len(marked)} qubits, fewer than the operation uses there"
            )
        cursors[qpu] = cursor + 1
        operands.append(marked[cursor])

    for qpu, used in cursors.items():
        marked = marker_qubits[(ordinal, qpu)]
        # A marker names the operation's operands on that QPU, and may name one
        # more after them: the shared qubit that orders it against the QPU's
        # other markers, which is what keeps routing from reordering them.
        # Anything beyond that does not describe this operation.
        if len(marked) > used + 1:
            raise ValueError(
                f"the marker for remote operation {ordinal} on QPU {qpu} names "
                f"{len(marked)} qubits, more than the operation uses there"
            )

    leading: dict[int, int] = {}
    for position, qpu in enumerate(source_qpus):
        leading.setdefault(qpu, operands[position])
    for named_qpu, named_qubit, field in (
        (op.qpu0, op.q0_phys, "q0_phys"),
        (op.qpu1, op.q1_phys, "q1_phys"),
    ):
        marked_qubit = leading.get(named_qpu)
        if marked_qubit != named_qubit:
            raise ValueError(
                f"remote operation {ordinal} names {field}={named_qubit} on QPU "
                f"{named_qpu}, but the programs put that operand on "
                f"{marked_qubit}; the manifest does not match these programs"
            )

    return operands


def _append_layout_restoration(
    circuit: QuantumCircuit,
    local_routed: Mapping[int, QuantumCircuit],
    arch: MultiQPUArchitecture,
    width: int,
) -> None:
    """Append swaps undoing each QPU's routing permutation.

    ``final_index_layout()`` reads "input qubit ``i`` ended at position
    ``final[i]``". Undoing it needs the inverse, so the mapping is turned around
    before selection-sorting the qubits back into place.
    """
    final = list(range(width))
    for qpu, routed in local_routed.items():
        layout = getattr(routed, "layout", None)
        if layout is None:
            continue
        positions = layout.final_index_layout(filter_ancillas=False)
        block = arch.block_of_qpu(qpu)
        for phys in block.compute + block.comm:
            if phys < width and phys < len(positions):
                final[phys] = positions[phys]

    holder = [0] * width
    for source, destination in enumerate(final):
        holder[destination] = source
    for target in range(width):
        source = holder.index(target)
        if source != target:
            circuit.swap(target, source)
            holder[target], holder[source] = holder[source], holder[target]


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


def _new_local_circuit(mapped: QuantumCircuit, n_phys: int) -> QuantumCircuit:
    """Create an empty per-QPU circuit mirroring the source's classical layout.

    The circuit spans every physical qubit so physical indices stay usable
    directly as qubit positions.

    Its classical side reuses the *source* circuit's clbits and registers rather
    than allocating a fresh anonymous one.  Operations that carry a
    register-valued condition (``if_else``, ``while_loop``, ``switch_case``, and
    legacy ``c_if``) reference the registers of the circuit they came from, so a
    local circuit built with its own register leaves those references dangling
    and Qiskit cannot convert the result to a DAG.
    """
    circuit = QuantumCircuit(QuantumRegister(n_phys, "q"))
    if mapped.clbits:
        circuit.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        circuit.add_register(creg)
    return circuit


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
        local[q] = _new_local_circuit(mapped, arch.n_phys)

    remote_ops: list[RemoteOp] = []
    # Barriers emitted per QPU so far.  A remote operation records where its own
    # marker sits in this sequence, because an emitted QASM program carries no
    # labels and a routed program may list its barriers in a different order.
    barriers_emitted = [0] * n_qpus
    def emit_marker(qpu: int, operands: Sequence[int], label: str) -> None:
        """Mark a remote operation on ``qpu``, ordered against its other markers.

        A marker naming only the operation's own operand does not order itself
        against the QPU's *other* remote operations, so local routing is free to
        emit two of them in the opposite order -- and the local gates it then
        interleaves can tie those markers together in that order, leaving the
        per-QPU programs demanding an execution no schedule satisfies. Remote
        operations are synchronization points with other QPUs, so their program
        order is not the transpiler's to choose.

        Each marker therefore also names the qubit the QPU's previous marker
        sat on, which puts an edge between the two in the local DAG and, by
        transitivity, keeps every marker on this QPU in program order. Naming
        one extra qubit is the smallest constraint that does so; a marker over
        the whole block would order the QPU's local gates against every remote
        operation as well.

        The operation's own operands come first, in operand order, so a reader
        recovers them by position; the ordering qubit follows them.
        """
        marked = list(operands)
        previous = last_marker_qubit[qpu]
        if previous is not None and previous not in marked:
            marked.append(previous)
        local[qpu].barrier(*marked, label=label)
        barriers_emitted[qpu] += 1
        last_marker_qubit[qpu] = operands[0]

    # The qubit each QPU's most recent remote marker sat on, threaded through
    # the next one so the local DAG keeps them in order.
    last_marker_qubit: list[int | None] = [None] * n_qpus

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
                    barriers_emitted[qpu] += 1
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
                barriers_emitted[qpu] += 1
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
                        qpu0_marker=barriers_emitted[qpu0],
                        qpu1_marker=barriers_emitted[qpu1],
                    )
                )
                # Barriers mark the synchronization points.  They carry a
                # label naming the remote op so that the qubit each one lands
                # on can be recovered after local routing has permuted things;
                # see `remap_remote_ops_to_routed`.
                label = remote_barrier_label(len(remote_ops) - 1)
                emit_marker(qpu0, (q0,), label)
                emit_marker(qpu1, (q1,), label)

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
                        qpu0_marker=barriers_emitted[qpu0],
                        qpu1_marker=barriers_emitted[qpu1],
                    )
                )
                label = remote_barrier_label(len(remote_ops) - 1)
                for qpu in qpu_order:
                    emit_marker(qpu, qpu_qubits[qpu], label)

    return DistributedProgram(local_circuits=local, remote_ops=remote_ops)
