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
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from functools import cache
from numbers import Integral
from pathlib import Path
from typing import Any, TypeAlias

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Barrier, ControlFlowOp, Measure, Store

from quport.architecture import MultiQPUArchitecture
from quport.entanglement import is_directive

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
    """A remote (inter-QPU) operation placeholder: two-qubit, or wider."""

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

    # Every operand, in operand order, as (QPU, physical qubit). The endpoints
    # above name only the first operand and the first one on another QPU; an
    # operation on three or more qubits has more, and applying it takes them
    # all. Empty when not recorded.
    operands: tuple[tuple[int, int], ...] = ()
    # Every QPU the operation involves, as (QPU, marker position), in the order
    # the QPUs first appear among the operands. An operation on three or more
    # qubits can involve more QPUs than its two endpoints, and it runs only
    # once its marker leads on each of them. Empty when not recorded.
    markers: tuple[tuple[int, int], ...] = ()

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
        qpu0_marker = _validate_optional_manifest_int(
            self.qpu0_marker, label="remote operation qpu0_marker"
        )
        qpu1_marker = _validate_optional_manifest_int(
            self.qpu1_marker, label="remote operation qpu1_marker"
        )
        operands = _validate_manifest_pairs(self.operands, label="operands")
        if operands:
            if len({phys for _, phys in operands}) != len(operands):
                raise ValueError("remote operation operands must be distinct qubits")
            second = next(((q, p) for q, p in operands if q != qpu0), None)
            if operands[0] != (qpu0, q0_phys) or second != (qpu1, q1_phys):
                raise ValueError(
                    "remote operation operands must start with (qpu0, q0_phys) "
                    "and hold (qpu1, q1_phys) as their first operand on another QPU"
                )
        markers = _validate_manifest_pairs(self.markers, label="markers")
        if markers:
            marker_of = dict(markers)
            if len(marker_of) != len(markers):
                raise ValueError("remote operation markers must name each QPU once")
            if operands and list(marker_of) != list(
                dict.fromkeys(qpu for qpu, _ in operands)
            ):
                raise ValueError(
                    "remote operation markers must name the operands' QPUs, in "
                    "the order they first appear"
                )
            for qpu, marker, field in (
                (qpu0, qpu0_marker, "qpu0_marker"),
                (qpu1, qpu1_marker, "qpu1_marker"),
            ):
                if qpu not in marker_of or (
                    marker is not None and marker_of[qpu] != marker
                ):
                    raise ValueError(
                        f"remote operation markers must agree with {field}"
                    )
        return {
            "name": self.name,
            "q0_phys": q0_phys,
            "q1_phys": q1_phys,
            "qpu0": qpu0,
            "qpu1": qpu1,
            "params": params,
            "clbits": clbits,
            "index": _validate_manifest_int(self.index, label="remote operation index"),
            "qpu0_marker": qpu0_marker,
            "qpu1_marker": qpu1_marker,
            "operands": [{"qpu": qpu, "phys": phys} for qpu, phys in operands],
            "markers": [{"qpu": qpu, "marker": marker} for qpu, marker in markers],
        }


def _validate_manifest_pairs(value: object, *, label: str) -> list[tuple[int, int]]:
    """Return a manifest field holding pairs of non-negative integers."""
    pairs: list[tuple[int, int]] = []
    for pair in _validate_manifest_sequence(value, label=f"remote operation {label}"):
        items = _validate_manifest_sequence(
            pair, label=f"each remote operation {label} entry"
        )
        if len(items) != 2:
            raise ValueError(f"each remote operation {label} entry must be a pair")
        pairs.append(
            (
                _validate_manifest_int(items[0], label=f"remote operation {label} QPU"),
                _validate_manifest_int(
                    items[1], label=f"remote operation {label} index"
                ),
            )
        )
    return pairs


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
    text = _render_remote_ops_json(remote_ops)
    out_path = _coerce_output_path(path, label="path")
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _encodable(text: str, *, label: str) -> str:
    """Return ``text`` once it is known to encode as UTF-8.

    A lone surrogate survives every check on a ``str`` and fails only when the
    file is written -- after ``write_text`` has already truncated it -- so
    encoding is part of validating an artifact rather than of writing it.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{label} contains text that cannot be encoded as UTF-8"
        ) from exc
    return text


def _render_remote_ops_json(remote_ops: Iterable[RemoteOp]) -> str:
    """Validate remote operations and return their manifest's text."""
    payload: list[dict[str, Any]] = []
    for idx, op in enumerate(remote_ops):
        if not isinstance(op, RemoteOp):
            raise ValueError(f"remote_ops[{idx}] must be a RemoteOp")
        payload.append(op.to_dict())
    return _encodable(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        label="the remote-operation manifest",
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
    or post-process generated files without re-deriving filenames.  A
    ``qpu_<id>.qasm`` an earlier bundle left in the directory, for a QPU this
    one does not write, is removed: the directory holds one bundle.
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

    # Render every artifact, and check that it encodes, before writing any of
    # them: a bad manifest found after the programs were written would leave
    # new programs beside a stale manifest that no longer matches them.
    rendered: list[tuple[int, str]] = [
        (
            qpu_id,
            _encodable(qasm3.dumps(circuit), label=f"the program for QPU {qpu_id}"),
        )
        for qpu_id, circuit in sorted(entries, key=lambda item: item[0])
        if include_empty_circuits or len(circuit.data) > 0
    ]
    manifest = _render_remote_ops_json(program.remote_ops)

    written: dict[str, Path] = {}
    for qpu_id, text in rendered:
        qpu_path = out_dir / f"qpu_{qpu_id}.qasm"
        qpu_path.write_text(text, encoding="utf-8")
        written[f"qpu_{qpu_id}"] = qpu_path

    remote_path = out_dir / "remote_ops.json"
    remote_path.write_text(manifest, encoding="utf-8")
    written["remote_ops"] = remote_path
    _remove_stale_programs(out_dir, written.values())
    return written


def _remove_stale_programs(
    out_dir: _PathLike, written: Iterable[_PathLike], *, suffix: str = ""
) -> list[Path]:
    """Remove the per-QPU programs an earlier bundle left in ``out_dir``.

    A consumer finds a bundle's programs by their names, ``qpu_<id><suffix>.qasm``,
    so one left behind for a QPU the new bundle does not write -- after a run
    on fewer QPUs, or one that skipped empty programs -- would be read as part
    of it. Files named so and not among ``written`` are removed; nothing else
    is touched. Returns the paths removed.
    """
    directory = Path(out_dir)
    keep = {Path(path).resolve() for path in written}
    pattern = re.compile(rf"qpu_\d+{re.escape(suffix)}\.qasm")
    removed: list[Path] = []
    for candidate in sorted(directory.iterdir()):
        if (
            pattern.fullmatch(candidate.name)
            and candidate.is_file()
            and candidate.resolve() not in keep
        ):
            candidate.unlink()
            removed.append(candidate)
    return removed


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


def _is_barrier(operation: object) -> bool:
    """True for Qiskit's barrier, recognised by class rather than by name.

    A custom gate called ``barrier`` is a gate, and the transpiler keeps it as
    one; treating it as a barrier would drop it from every program.
    """
    return isinstance(operation, Barrier)


def _barrier_statements(operation: object) -> int:
    """Barrier statements ``operation`` puts in an emitted program.

    A barrier is one. A control-flow operation is emitted with its bodies
    inline, so it contributes every barrier inside them; anything else none.
    Marker indices count exactly these, so that a consumer counting barrier
    statements in the OpenQASM file lands on the same one.
    """
    if _is_barrier(operation):
        return 1
    if isinstance(operation, ControlFlowOp):
        return sum(
            _barrier_statements(instruction.operation)
            for block in operation.blocks
            for instruction in block.data
        )
    return 0


def _remote_barrier_ordinal(operation: object) -> int | None:
    """Return the remote-op ordinal a barrier marks, or None if it marks none."""
    if not _is_barrier(operation):
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
        The same operations, in the same order, with ``q0_phys``, ``q1_phys``
        and ``operands`` replaced by their post-routing positions, and
        ``qpu0_marker``, ``qpu1_marker`` and ``markers`` by where each marker
        now sits among the routed program's barriers -- routing can move a
        marker past another barrier. Every other field, including the global
        instruction ``index``, is unchanged.

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
            if not _is_barrier(instruction.operation):
                seen_barriers += _barrier_statements(instruction.operation)
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
        # split_into_qpus emits each marker over that QPU's operands in operand
        # order, so its leading qubits are where routing left them: q0_phys and
        # q1_phys are the first on their QPUs, and the operands are read off in
        # turn, QPU by QPU.
        involved = [qpu for qpu, _ in op.markers] or [op.qpu0, op.qpu1]
        found = {
            qpu: _routed_marker(markers, ordinal, qpu)
            for qpu in dict.fromkeys([op.qpu0, op.qpu1, *involved])
        }
        cursors: dict[int, int] = {}
        operands: list[tuple[int, int]] = []
        for qpu, _ in op.operands:
            marked = found.setdefault(qpu, _routed_marker(markers, ordinal, qpu))[0]
            cursor = cursors.get(qpu, 0)
            if cursor >= len(marked):
                raise ValueError(
                    f"the marker for remote operation {ordinal} on QPU {qpu} names "
                    f"{len(marked)} qubits, fewer than the operation uses there"
                )
            cursors[qpu] = cursor + 1
            operands.append((qpu, marked[cursor]))
        out.append(
            replace(
                op,
                q0_phys=found[op.qpu0][0][0],
                q1_phys=found[op.qpu1][0][0],
                qpu0_marker=found[op.qpu0][1],
                qpu1_marker=found[op.qpu1][1],
                operands=tuple(operands),
                markers=tuple((qpu, found[qpu][1]) for qpu, _ in op.markers),
            )
        )
    return out


def _routed_marker(
    markers: Mapping[tuple[int, int], tuple[list[int], int]], ordinal: int, qpu: int
) -> tuple[list[int], int]:
    """Return the routed qubits and barrier position of one QPU's marker."""
    entry = markers.get((ordinal, qpu))
    if entry is None or not entry[0]:
        raise ValueError(
            f"routed program for QPU {qpu} has no marker for remote operation "
            f"{ordinal}; the manifest and the routed circuits are out of step"
        )
    return entry


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
    and a remote operation runs once its marker leads on every QPU it involves.

    Classical bits are dataflow too, and they cross QPUs: a conditional on one
    QPU can read a bit measured on another. The per-QPU programs do not record
    that order -- each sees only its own accesses -- so it is taken from
    ``mapped``: an instruction that touches a classical bit runs only once every
    earlier access to that bit, on whichever QPU, has run.

    That is also the contract a consumer of the artifacts has to honour: reading
    each program strictly linearly can deadlock, because two QPUs can list the
    same pair of remote operations in opposite orders when they sit on disjoint
    qubits.

    Parameters
    ----------
    mapped:
        The circuit the programs were split from. Its width, its classical bits
        and registers, the order in which its instructions access each
        classical bit, and the operations named by ``remote_ops[k].index`` are
        used -- the last is how parameters and custom gates survive the round
        trip exactly. Its other operations are taken from the programs.
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
        on the remote operations or classical bits -- a genuine inconsistency
        rather than a scheduling choice.
    """
    if not isinstance(mapped, QuantumCircuit):
        raise ValueError("mapped must be a QuantumCircuit")
    if not isinstance(local_routed, Mapping):
        raise ValueError("local_routed must be a mapping of QPU id to circuit")
    if not isinstance(arch, MultiQPUArchitecture):
        raise ValueError("arch must be a MultiQPUArchitecture")
    if not isinstance(remote_ops, Sequence) or isinstance(remote_ops, str):
        raise ValueError("remote_ops must be a sequence of RemoteOp")
    # A manifest read back from JSON is a list of dicts, and one from other
    # programs can name instructions the circuit does not have; both would
    # otherwise fail deep inside the merge.
    named: set[int] = set()
    for entry, op in enumerate(remote_ops):
        if not isinstance(op, RemoteOp):
            raise ValueError(f"remote_ops[{entry}] must be a RemoteOp")
        index = _validate_manifest_int(op.index, label=f"remote_ops[{entry}].index")
        if index >= len(mapped.data):
            raise ValueError(
                f"remote_ops[{entry}] names instruction {index}, which the "
                "mapped circuit does not have"
            )
        if index in named:
            raise ValueError(
                f"remote_ops[{entry}] names instruction {index}, as an earlier "
                "entry does"
            )
        named.add(index)
    for qpu in local_routed:
        _validate_manifest_int(qpu, label="local circuit QPU id")

    reject_classical_storage(mapped, what="reassemble_distributed_program")

    width = max(arch.n_phys, len(mapped.qubits))
    name = _fresh_register_name("q", (creg.name for creg in mapped.cregs))
    out = QuantumCircuit(QuantumRegister(width, name))
    # Carry the classical side across, so measurements and conditioned
    # operations land on the bits they named in the circuit that was split.
    if mapped.clbits:
        out.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        out.add_register(creg)
    clbit_at = {clbit: index for index, clbit in enumerate(mapped.clbits)}

    def clbit_index(clbit: Any) -> int:
        """Return the merged circuit's index for a program's classical bit."""
        try:
            return clbit_at[clbit]
        except KeyError:  # pragma: no cover - defensive
            raise ValueError(
                "a per-QPU program uses classical bits the mapped circuit "
                "does not have"
            ) from None

    qpus = sorted(local_routed)
    data: dict[int, list[Any]] = {}
    index_of: dict[int, dict[Any, int]] = {}
    queues: dict[int, dict[int, deque[int]]] = {}
    # Per QPU, each classical bit's accesses in program order, like ``queues``
    # for qubits.
    clbit_queues: dict[int, dict[int, deque[int]]] = {}
    retired: dict[int, list[bool]] = {}
    marker_at: dict[tuple[int, int], int] = {}
    marker_qubits: dict[tuple[int, int], list[int]] = {}
    mapped_at = {qubit: index for index, qubit in enumerate(mapped.qubits)}

    def carried_clbits(instruction: Any) -> list[Any]:
        """Map an instruction's classical arguments onto the merged circuit."""
        return [out.clbits[clbit_index(clbit)] for clbit in instruction.clbits]

    # Who touches each classical bit, in the mapped circuit's order: a QPU for
    # a local instruction, a manifest ordinal for a remote one. An instruction
    # without qubits belongs to the QPU `split_into_qpus` gives it; one that
    # routing emptied of qubits stays with the QPU whose program holds it.
    #
    # A program routing decomposed a custom instruction in no longer names it,
    # and holds one access to each bit for every access the instruction's
    # definition makes -- two for a readout measuring twice into one bit, none
    # for a bit it only declares -- so such an instruction stands for that many
    # accesses here. Counting it once would let another QPU's access slip in
    # between the pieces, or leave the order stuck.
    program_names = {
        qpu: {instruction.operation.name for instruction in circuit.data}
        for qpu, circuit in local_routed.items()
        if isinstance(circuit, QuantumCircuit)
    }
    ordinal_at = {op.index: ordinal for ordinal, op in enumerate(remote_ops)}
    clbit_owners: list[list[tuple[str, int]]] = [[] for _ in mapped.clbits]
    for position, instruction in enumerate(mapped.data):
        if not instruction.clbits:
            continue
        ordinal = ordinal_at.get(position)
        accesses: list[int] | None = None
        if ordinal is None:
            qpu = (
                arch.qpu_of_phys(mapped_at[instruction.qubits[0]])
                if instruction.qubits
                else _QPU_WITHOUT_QUBITS
            )
            owner = ("local", qpu)
            if instruction.operation.name not in program_names.get(qpu, ()):
                accesses = _accesses_once_decomposed(instruction.operation)
        else:
            owner = ("remote", ordinal)
        for slot, clbit in enumerate(instruction.clbits):
            count = 1 if accesses is None else accesses[slot]
            clbit_owners[clbit_at[clbit]].extend([owner] * count)
    clbit_cursor = [0] * len(clbit_owners)
    # How many separate runs of accesses each QPU makes to each bit. Routing
    # can decompose one instruction into several that all access its bits, so
    # a QPU's program may hold more accesses than ``mapped`` lists for it; with
    # a single run they all belong to it, and the order moves on only once the
    # program has no access to the bit left.
    local_runs: list[dict[int, int]] = []
    for owners in clbit_owners:
        runs: dict[int, int] = {}
        previous: tuple[str, int] | None = None
        for owner in owners:
            if owner != previous and owner[0] == "local":
                runs[owner[1]] = runs.get(owner[1], 0) + 1
            previous = owner
        local_runs.append(runs)

    def owns_next(owner: tuple[str, int], clbits: Iterable[int]) -> bool:
        """True when ``owner`` makes the next access to each of these bits."""
        for clbit in clbits:
            owners = clbit_owners[clbit]
            cursor = clbit_cursor[clbit]
            if cursor >= len(owners) or owners[cursor] != owner:
                return False
        return True

    for qpu in qpus:
        circuit = local_routed[qpu]
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError(f"local_routed[{qpu!r}] must be a QuantumCircuit")
        # Routing can unroll a custom instruction into a store of its own.
        reject_classical_storage(circuit, what="reassemble_distributed_program")
        instructions = list(circuit.data)
        positions = {qubit: index for index, qubit in enumerate(circuit.qubits)}
        per_qubit: dict[int, deque[int]] = {}
        per_clbit: dict[int, deque[int]] = {}
        for index, instruction in enumerate(instructions):
            ordinal = _remote_barrier_ordinal(instruction.operation)
            if ordinal is not None:
                marker_at[(ordinal, qpu)] = index
                marker_qubits[(ordinal, qpu)] = [
                    positions[qubit] for qubit in instruction.qubits
                ]
            for qubit in instruction.qubits:
                per_qubit.setdefault(positions[qubit], deque()).append(index)
            for clbit in instruction.clbits:
                per_clbit.setdefault(clbit_index(clbit), deque()).append(index)
        data[qpu] = instructions
        index_of[qpu] = positions
        queues[qpu] = per_qubit
        clbit_queues[qpu] = per_clbit
        retired[qpu] = [False] * len(instructions)

    def leads(qpu: int, index: int) -> bool:
        """True when this instruction is first in line on every bit it uses.

        That is every qubit and classical bit in its own program, and, for a
        classical bit, across all programs as well.
        """
        instruction = data[qpu][index]
        if not all(
            queues[qpu][index_of[qpu][qubit]][0] == index
            for qubit in instruction.qubits
        ):
            return False
        clbits = [clbit_index(clbit) for clbit in instruction.clbits]
        return all(
            clbit_queues[qpu][clbit][0] == index for clbit in clbits
        ) and owns_next(("local", qpu), clbits)

    def advance(clbits: Iterable[int]) -> None:
        for clbit in clbits:
            clbit_cursor[clbit] += 1

    def advance_local(qpu: int, clbits: Iterable[int]) -> None:
        """Move past a local access, or past the QPU's whole single run."""
        owner = ("local", qpu)
        for clbit in clbits:
            if local_runs[clbit].get(qpu, 0) != 1:
                clbit_cursor[clbit] += 1
                continue
            if clbit_queues[qpu][clbit]:
                continue
            owners = clbit_owners[clbit]
            cursor = clbit_cursor[clbit]
            while cursor < len(owners) and owners[cursor] == owner:
                cursor += 1
            clbit_cursor[clbit] = cursor

    def retire(qpu: int, index: int) -> None:
        retired[qpu][index] = True
        instruction = data[qpu][index]
        for qubit in instruction.qubits:
            queues[qpu][index_of[qpu][qubit]].popleft()
        clbits = [clbit_index(clbit) for clbit in instruction.clbits]
        for clbit in clbits:
            clbit_queues[qpu][clbit].popleft()
        advance_local(qpu, clbits)

    def takes_part(instruction: Any) -> bool:
        """True unless the instruction is a directive acting on no qubit.

        An instruction without qubits still has its classical effect -- a
        condition whose body routing emptied of qubits still carries its body --
        so it is merged like any other, ordered by its classical bits alone.
        """
        return bool(instruction.qubits) or not is_directive(instruction.operation)

    remaining = {qpu: sum(1 for inst in data[qpu] if takes_part(inst)) for qpu in qpus}
    pending = set(range(len(remote_ops)))

    while any(remaining.values()) or pending:
        progressed = False

        for qpu in qpus:
            for index, instruction in enumerate(data[qpu]):
                if retired[qpu][index] or not takes_part(instruction):
                    continue
                if _remote_barrier_ordinal(instruction.operation) is not None:
                    continue
                if not leads(qpu, index):
                    continue
                if not is_directive(instruction.operation):
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
            source_clbits = [clbit_at[clbit] for clbit in source.clbits]
            if not owns_next(("remote", ordinal), source_clbits):
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
            advance(source_clbits)
            pending.discard(ordinal)
            progressed = True

        if not progressed:
            raise ValueError(
                "per-QPU programs impose contradictory orders on their remote "
                "operations or classical bits; the programs and the manifest "
                "are out of step"
            )

    if restore_layout:
        _append_layout_restoration(out, local_routed, arch, width)
    return out


def _accesses_once_decomposed(operation: Any) -> list[int] | None:
    """Accesses to each classical argument once ``operation`` is decomposed.

    Transpilation keeps a standard operation, a control-flow block, and an
    operation without a definition whole -- one access to each of its bits,
    reported as ``None`` -- and replaces any other by its definition, recursively.
    """
    if isinstance(operation, ControlFlowOp) or (
        operation.name in _standard_operation_names()
    ):
        return None
    definition = getattr(operation, "definition", None)
    if not isinstance(definition, QuantumCircuit):
        return None
    slot_of = {clbit: slot for slot, clbit in enumerate(definition.clbits)}
    counts = [0] * len(definition.clbits)
    for inner in definition.data:
        nested = _accesses_once_decomposed(inner.operation)
        for slot, clbit in enumerate(inner.clbits):
            counts[slot_of[clbit]] += 1 if nested is None else nested[slot]
    return counts


@cache
def _standard_operation_names() -> frozenset[str]:
    """Names of the operations Qiskit defines as standard."""
    from qiskit.circuit.library import get_standard_gate_name_mapping

    return frozenset(get_standard_gate_name_mapping())


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

    # A manifest rebuilt from JSON holds lists where split_into_qpus holds tuples.
    named = [tuple(pair) for pair in op.operands]
    if named and named != list(zip(source_qpus, operands)):
        raise ValueError(
            f"remote operation {ordinal} names operands {named}, but "
            f"the programs put them on {list(zip(source_qpus, operands))}; the "
            "manifest does not match these programs"
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
        try:
            positions = layout.final_index_layout(filter_ancillas=False)
        except (AttributeError, KeyError, TypeError, ValueError):
            # A program that did not come from the transpiler can still carry a
            # layout: a circuit read back from physical-qubit OpenQASM gets one
            # built from loose qubits, which Qiskit's own accessor cannot
            # resolve. Such a layout records no routing permutation -- the file
            # never held one -- so there is nothing to undo, and it is treated
            # exactly like the missing layout of an unrouted program.
            continue
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


def _fresh_register_name(preferred: str, taken: Iterable[str]) -> str:
    """Return ``preferred``, or the first ``preferred_<n>`` not in ``taken``.

    QuPort adds registers of its own beside a circuit's classical registers,
    and Qiskit refuses two registers of one name, so a user's register that
    happens to be called ``q`` or ``cat`` must not stop the build.
    """
    names = set(taken)
    if preferred not in names:
        return preferred
    suffix = 1
    while f"{preferred}_{suffix}" in names:
        suffix += 1
    return f"{preferred}_{suffix}"


def reject_classical_storage(circuit: QuantumCircuit, *, what: str) -> None:
    """Refuse classical variables, stretches, and ``store`` anywhere.

    Each per-QPU program would need its own declaration of a variable, and a
    store touches no qubit -- no QPU owns it, and nothing records the order it
    takes against the other accesses to its target -- so neither can be split
    and merged back faithfully. That holds inside a control-flow body too. A
    store is recognised by class: a gate merely named ``store`` is a gate.
    Saying so beats a Qiskit error from deep inside, or a store that silently
    disappears in the round trip.
    """
    if circuit.num_vars or getattr(circuit, "num_stretches", 0):
        raise ValueError(
            f"{what} does not support circuits with classical variables or stretches"
        )
    if _holds_store(circuit):
        raise ValueError(f"{what} does not support classical store instructions")


def _holds_store(circuit: QuantumCircuit) -> bool:
    """True when ``circuit`` stores to a classical target, at any depth."""
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, Store):
            return True
        if isinstance(operation, ControlFlowOp) and any(
            _holds_store(block) for block in operation.blocks
        ):
            return True
    return False


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
    name = _fresh_register_name("q", (creg.name for creg in mapped.cregs))
    circuit = QuantumCircuit(QuantumRegister(n_phys, name))
    if mapped.clbits:
        circuit.add_bits(mapped.clbits)
    for creg in mapped.cregs:
        circuit.add_register(creg)
    return circuit


#: The QPU whose program carries the instructions that act on no qubit.
_QPU_WITHOUT_QUBITS = 0


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
    - An instruction acting on no qubit, such as a condition whose body is empty, goes
      to QPU 0's program only, so it still runs once.
    """
    reject_classical_storage(mapped, what="split_into_qpus")
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

    def append_local(qpu: int, op: Any, qargs: list[Any], cargs: list[Any]) -> None:
        """Append a local operation, counting any barriers nested inside it."""
        local[qpu].append(op, qargs, cargs)
        barriers_emitted[qpu] += _barrier_statements(op)

    qindex = {q: i for i, q in enumerate(mapped.qubits)}
    cindex = {c: i for i, c in enumerate(mapped.clbits)}

    for idx, inst in enumerate(mapped.data):
        cargs_idx = [cindex[c] for c in inst.clbits]
        op = inst.operation
        qs = [qindex[q] for q in inst.qubits]

        if not qs:
            if _is_barrier(op):
                for qpu in range(n_qpus):
                    local[qpu].barrier()
                    barriers_emitted[qpu] += 1
            else:
                # No QPU owns an instruction without qubits, but it must still
                # run exactly once: a copy per program would apply it once per
                # QPU. QPU 0 takes it, as reassembly expects.
                append_local(
                    _QPU_WITHOUT_QUBITS,
                    op,
                    [],
                    _local_cargs_for_qpu(local, _QPU_WITHOUT_QUBITS, cargs_idx),
                )
            continue

        op_qpus = [arch.qpu_of_phys(q) for q in qs]

        if _is_barrier(op):
            qpu_order, qpu_qubits_barrier = _group_qubits_by_qpu_in_operand_order(
                qs, op_qpus
            )
            for qpu in qpu_order:
                local[qpu].barrier(*qpu_qubits_barrier[qpu])
                barriers_emitted[qpu] += 1
            continue

        if len(qs) == 1:
            qpu = op_qpus[0]
            append_local(
                qpu,
                op,
                [local[qpu].qubits[qs[0]]],
                _local_cargs_for_qpu(local, qpu, cargs_idx),
            )

        elif len(qs) == 2:
            q0, q1 = qs
            qpu0, qpu1 = op_qpus
            if qpu0 == qpu1:
                append_local(
                    qpu0,
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
                        operands=((qpu0, q0), (qpu1, q1)),
                        markers=(
                            (qpu0, barriers_emitted[qpu0]),
                            (qpu1, barriers_emitted[qpu1]),
                        ),
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
                append_local(
                    qpu,
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
                        operands=tuple(zip(op_qpus, qs)),
                        markers=tuple(
                            (qpu, barriers_emitted[qpu]) for qpu in qpu_order
                        ),
                    )
                )
                label = remote_barrier_label(len(remote_ops) - 1)
                for qpu in qpu_order:
                    emit_marker(qpu, qpu_qubits[qpu], label)

    return DistributedProgram(local_circuits=local, remote_ops=remote_ops)


#: Label of the barriers that hold events in order while a circuit is routed.
#: They exist only for the duration of routing.
_ORDER_BARRIER_LABEL = "quport_order"


def _shared_clbits(mapped: QuantumCircuit, arch: MultiQPUArchitecture) -> set[Any]:
    """Classical bits that more than one QPU, or a remote operation, accesses."""
    qindex = {qubit: index for index, qubit in enumerate(mapped.qubits)}
    owners: dict[Any, set[tuple[str, int]]] = {}
    for position, instruction in enumerate(mapped.data):
        if not instruction.clbits:
            continue
        qpus = {arch.qpu_of_phys(qindex[qubit]) for qubit in instruction.qubits}
        if not qpus:
            owner = ("local", _QPU_WITHOUT_QUBITS)
        elif len(qpus) == 1:
            owner = ("local", qpus.pop())
        else:
            owner = ("remote", position)
        for clbit in instruction.clbits:
            owners.setdefault(clbit, set()).add(owner)
    return {
        clbit
        for clbit, who in owners.items()
        if len(who) > 1 or any(kind == "remote" for kind, _ in who)
    }


def _hold_cross_qpu_order(
    program: DistributedProgram, mapped: QuantumCircuit, arch: MultiQPUArchitecture
) -> dict[int, QuantumCircuit]:
    """Copies of ``program``'s circuits whose cross-QPU events routing keeps in order.

    A program meets the others at its remote markers and at its accesses to
    classical bits another QPU or a remote operation also uses, and the order
    between programs is the mapped circuit's. Routing is free to emit two such
    events that share no wire in the opposite order, and the swaps it then adds
    can tie them together that way, leaving programs that no execution
    satisfies: a QPU waiting on a bit another QPU writes only after this QPU's
    later event. Each event is therefore tied to the one before it by a
    labelled barrier, which :func:`_release_order_barriers` takes out after
    routing. Events on bits no other program touches need no tie: their own
    wires order them. Measurements into one bit are tied as well; see
    :func:`_hold_measurement_order`.
    """
    shared = _shared_clbits(mapped, arch)

    def meets_others(instruction: Any) -> bool:
        return _remote_barrier_ordinal(instruction.operation) is not None or (
            not shared.isdisjoint(instruction.clbits)
        )

    return {
        qpu: _with_order_ties(circuit, meets_others)
        for qpu, circuit in program.local_circuits.items()
    }


def _hold_measurement_order(circuit: QuantumCircuit) -> QuantumCircuit:
    """A copy of ``circuit`` whose measurements into one bit routing keeps in order.

    Qiskit's routing stage moves the final measurements behind a barrier and
    puts them back in qubit order, honouring their qubit wires but not their
    classical ones, so two final measurements into the same bit can swap and
    leave the earlier outcome in the bit. Tying each measurement to the
    previous one into its bit with a labelled barrier over both makes their
    order a qubit dependency, which it keeps;
    :func:`_release_order_barriers` takes the barriers out after routing.
    """
    return _with_order_ties(circuit, lambda instruction: False)


def _with_order_ties(
    circuit: QuantumCircuit, is_event: Callable[[Any], bool]
) -> QuantumCircuit:
    """Copy ``circuit``, tying each event to the previous one, and each
    measurement to the previous measurement into the same bit.

    A tie is a labelled barrier spanning every qubit of both instructions, not
    one of each: routing drops the idle wires of a control-flow block, and
    unrolls a custom instruction into pieces on fewer qubits, either of which
    can cut a tie through a single qubit. Whatever survives of an instruction
    still touches the barrier, and one left with no qubit at all is ordered by
    its classical bits alone. Two instructions that share a qubit and that
    routing keeps whole -- markers and measurements -- are already ordered by it.
    """
    out = circuit.copy_empty_like()
    previous_event: Any = None
    last_access: dict[Any, Any] = {}
    for instruction in circuit.data:
        earlier: list[Any] = []
        if instruction.qubits and is_event(instruction):
            if previous_event is not None:
                earlier.append(previous_event)
            previous_event = instruction
        if isinstance(instruction.operation, Measure):
            for clbit in instruction.clbits:
                prior = last_access.get(clbit)
                if (
                    prior is not None
                    and isinstance(prior.operation, Measure)
                    and not any(prior is seen for seen in earlier)
                ):
                    earlier.append(prior)
        earlier = [
            prior
            for prior in earlier
            if not _ordered_by_a_shared_wire(prior, instruction)
        ]
        if earlier:
            tied = list(
                dict.fromkeys(
                    [qubit for prior in earlier for qubit in prior.qubits]
                    + list(instruction.qubits)
                )
            )
            out.append(Barrier(len(tied), label=_ORDER_BARRIER_LABEL), tied)
        for clbit in instruction.clbits:
            last_access[clbit] = instruction
        out.append(instruction)
    return out


def _ordered_by_a_shared_wire(first: Any, second: Any) -> bool:
    """True when routing cannot separate two instructions it keeps whole."""

    def whole(operation: Any) -> bool:
        return _is_barrier(operation) or isinstance(operation, Measure)

    return (
        whole(first.operation)
        and whole(second.operation)
        and not set(first.qubits).isdisjoint(second.qubits)
    )


def _release_order_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """Take the barriers the hold functions added out of ``circuit``."""
    circuit.data = [
        instruction
        for instruction in circuit.data
        if not (
            _is_barrier(instruction.operation)
            and getattr(instruction.operation, "label", None) == _ORDER_BARRIER_LABEL
        )
    ]
    return circuit
