# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Communication aggregation for mapped multi-QPU circuits.

A per-gate telegate compiler spends one EPR pair on every cross-QPU two-qubit
gate. That is wasteful: a single cat copy of a root qubit can serve *every*
later gate in which the root stays diagonal in the computational basis. This
module turns a mapped physical circuit into an ordered list of
:class:`RemoteBlock` objects, each of which is one entanglement transaction:

``cat``
    Distribute one EPR pair, build a cat copy of the root on the remote QPU,
    run every gate in the block against that copy, then disentangle. One e-bit,
    one comm port held on the remote QPU for the block's whole window.

``teleport``
    For gates no cat copy can serve -- ``swap``, ``ecr``, ``rxx`` and friends,
    whose operands are both non-diagonal, and operations on three or more
    qubits. The operand is teleported to the host QPU and back: two e-bits.
    QuPort does not merge consecutive teleports into one round trip, so each
    such gate is charged independently.

Where this sits
---------------
:mod:`quport.hypergraph` answers the same question *before* placement, as a
partitioning objective over logical qubits. This module answers it *after*
placement, on real physical qubits, and additionally respects the comm-port
budget: a QPU with ``P`` ports can host at most ``P`` cat copies at once, so a
plan that would exceed that evicts its least recently used copy and pays for a
fresh EPR pair when the evicted root is needed again. With an unbounded port
budget the two agree exactly, which
``tests/test_aggregation.py::test_unbounded_ports_match_hypergraph_ebits``
pins down.

The resulting plan feeds :func:`quport.schedule.estimate_entanglement_schedule`,
which is where port hold times and link capacity turn into a makespan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.entanglement import diagonal_positions, is_directive

__all__ = [
    "AggregationPlan",
    "BlockProtocol",
    "RemoteBlock",
    "aggregate_remote_operations",
]

BlockProtocol = Literal["cat", "teleport"]

#: E-bits consumed by one block of each protocol.
_EPR_PER_PROTOCOL: dict[str, int] = {"cat": 1, "teleport": 2}


@dataclass(frozen=True)
class RemoteBlock:
    """One entanglement transaction between two QPUs.

    Attributes
    ----------
    protocol:
        ``"cat"`` for a shared cat copy, ``"teleport"`` for a round-trip move.
    root_phys:
        Physical qubit whose state is copied (``cat``) or moved (``teleport``).
    root_qpu:
        QPU that owns ``root_phys``.
    remote_qpu:
        QPU that hosts the cat copy, or receives the teleported qubit. This is
        the QPU whose comm port is occupied for the block's window.
    gate_indices:
        Instruction indices of the mapped circuit served by this block, in
        circuit order. Always non-empty.
    epr_pairs:
        EPR pairs consumed: one for ``cat``, two for ``teleport``.
    """

    protocol: BlockProtocol
    root_phys: int
    root_qpu: int
    remote_qpu: int
    gate_indices: tuple[int, ...]
    epr_pairs: int

    @property
    def start_index(self) -> int:
        """Instruction index at which the entanglement must be available."""
        return self.gate_indices[0]

    @property
    def end_index(self) -> int:
        """Instruction index after which the entanglement can be released."""
        return self.gate_indices[-1]

    def size(self) -> int:
        """Number of cross-QPU gates this block serves."""
        return len(self.gate_indices)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation of this block."""
        return {
            "protocol": self.protocol,
            "root_phys": self.root_phys,
            "root_qpu": self.root_qpu,
            "remote_qpu": self.remote_qpu,
            "gate_indices": list(self.gate_indices),
            "start_index": self.start_index,
            "end_index": self.end_index,
            "epr_pairs": self.epr_pairs,
            "gates": self.size(),
        }


@dataclass(frozen=True)
class AggregationPlan:
    """An ordered communication plan for a mapped circuit.

    Attributes
    ----------
    blocks:
        Blocks sorted by ``(start_index, remote_qpu, root_phys)``.
    remote_gates:
        Cross-QPU operations found in the circuit, whether or not they could be
        served (``unschedulable_gates`` counts the ones that could not).
    epr_pairs:
        EPR pairs the plan consumes.
    baseline_epr_pairs:
        EPR pairs an un-aggregated, one-transaction-per-gate compiler would
        consume on the same circuit. Aggregation can only lower this.
    unschedulable_gates:
        Cross-QPU gates that no protocol can serve because the QPU that would
        have to host the entanglement has no comm ports at all.
    evictions:
        Times a live cat copy was released early to free a port. Each eviction
        costs one extra EPR pair if the same root is needed again.
    peak_cat_copies:
        Per QPU, the largest number of simultaneously live cat copies. By
        construction this never exceeds that QPU's port budget.
    """

    blocks: tuple[RemoteBlock, ...]
    remote_gates: int
    epr_pairs: int
    baseline_epr_pairs: int
    unschedulable_gates: int
    evictions: int
    peak_cat_copies: tuple[int, ...]

    @property
    def reduction(self) -> float:
        """Fraction of baseline EPR pairs saved by aggregation, in ``[0, 1]``."""
        if self.baseline_epr_pairs <= 0:
            return 0.0
        return 1.0 - (self.epr_pairs / self.baseline_epr_pairs)

    def blocks_by_gate_index(self) -> dict[int, tuple[RemoteBlock, ...]]:
        """Map every served instruction index to the blocks that serve it.

        Cross-QPU two-qubit gates always map to exactly one block. An operation
        on three or more qubits spanning ``k`` QPUs maps to the ``k - 1``
        teleport blocks that gather its operands, so the value is a tuple.
        """
        out: dict[int, list[RemoteBlock]] = {}
        for block in self.blocks:
            for index in block.gate_indices:
                out.setdefault(index, []).append(block)
        return {index: tuple(blocks) for index, blocks in out.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation of the plan."""
        return {
            "remote_gates": self.remote_gates,
            "epr_pairs": self.epr_pairs,
            "baseline_epr_pairs": self.baseline_epr_pairs,
            "reduction": self.reduction,
            "unschedulable_gates": self.unschedulable_gates,
            "evictions": self.evictions,
            "peak_cat_copies": list(self.peak_cat_copies),
            "blocks": [block.to_dict() for block in self.blocks],
        }


class _OpenBlock:
    """A cat block that is still accepting gates."""

    __slots__ = ("root_phys", "root_qpu", "remote_qpu", "gate_indices")

    def __init__(self, root_phys: int, root_qpu: int, remote_qpu: int) -> None:
        self.root_phys = root_phys
        self.root_qpu = root_qpu
        self.remote_qpu = remote_qpu
        self.gate_indices: list[int] = []

    def freeze(self) -> RemoteBlock:
        return RemoteBlock(
            protocol="cat",
            root_phys=self.root_phys,
            root_qpu=self.root_qpu,
            remote_qpu=self.remote_qpu,
            gate_indices=tuple(self.gate_indices),
            epr_pairs=_EPR_PER_PROTOCOL["cat"],
        )


def _validate_ports(
    ports_per_qpu: int | Sequence[int] | None,
    *,
    n_qpus: int,
    default: int,
) -> list[int]:
    """Normalize the per-QPU comm-port budget to one non-negative int per QPU."""
    if ports_per_qpu is None:
        value = default
        if type(value) is bool or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError("comm_qubits_per_qpu must be a non-negative integer")
        return [int(value)] * n_qpus

    if type(ports_per_qpu) is not bool and isinstance(ports_per_qpu, Integral):
        value = int(ports_per_qpu)
        if value < 0:
            raise ValueError("ports_per_qpu must be non-negative")
        return [value] * n_qpus

    if isinstance(ports_per_qpu, str | bytes | bytearray) or not isinstance(
        ports_per_qpu, Sequence
    ):
        raise ValueError("ports_per_qpu must be an integer or a sequence of integers")
    if len(ports_per_qpu) != n_qpus:
        raise ValueError("ports_per_qpu length must match n_qpus")

    out: list[int] = []
    for index, entry in enumerate(ports_per_qpu):
        if type(entry) is bool or not isinstance(entry, Integral):
            raise ValueError(f"ports_per_qpu[{index}] must be an integer")
        value = int(entry)
        if value < 0:
            raise ValueError(f"ports_per_qpu[{index}] must be non-negative")
        out.append(value)
    return out


def _validate_max_block_gates(max_block_gates: int | None) -> int | None:
    if max_block_gates is None:
        return None
    if type(max_block_gates) is bool or not isinstance(max_block_gates, Integral):
        raise ValueError("max_block_gates must be an integer")
    value = int(max_block_gates)
    if value <= 0:
        raise ValueError("max_block_gates must be positive")
    return value


def aggregate_remote_operations(
    mapped: QuantumCircuit,
    arch: MultiQPUArchitecture,
    *,
    ports_per_qpu: int | Sequence[int] | None = None,
    max_block_gates: int | None = None,
) -> AggregationPlan:
    """Aggregate a mapped circuit's cross-QPU gates into entanglement blocks.

    Parameters
    ----------
    mapped:
        A circuit whose qubit indices are physical indices of ``arch``.
    arch:
        The multi-QPU architecture that defines the physical-to-QPU mapping.
    ports_per_qpu:
        Comm ports available to host cat copies, either one budget for every QPU
        or one per QPU. Defaults to ``arch.cfg.comm_qubits_per_qpu``. Pass a
        large value to measure the port-unconstrained e-bit lower bound.
    max_block_gates:
        Optional cap on how many gates one block may serve. Useful for studying
        the trade-off between e-bit savings and how long a port stays pinned.

    Returns
    -------
    AggregationPlan

    Notes
    -----
    A block is closed as soon as its root qubit is touched non-diagonally,
    because at that instant the cat copy stops tracking the root. Barriers do
    not close blocks, and gates that are local to one QPU only close blocks
    whose root they disturb.
    """
    if not isinstance(mapped, QuantumCircuit):
        raise ValueError("mapped must be a QuantumCircuit")
    if not isinstance(arch, MultiQPUArchitecture):
        raise ValueError("arch must be a MultiQPUArchitecture")

    n_qpus = arch.cfg.n_qpus
    ports = _validate_ports(
        ports_per_qpu, n_qpus=n_qpus, default=arch.cfg.comm_qubits_per_qpu
    )
    gate_cap = _validate_max_block_gates(max_block_gates)

    qindex = {qubit: index for index, qubit in enumerate(mapped.qubits)}
    phys_to_qpu = [arch.qpu_of_phys(phys) for phys in range(len(mapped.qubits))]

    finished: list[RemoteBlock] = []
    open_blocks: dict[tuple[int, int], _OpenBlock] = {}
    roots_index: dict[int, set[tuple[int, int]]] = {}
    qpu_index: list[set[tuple[int, int]]] = [set() for _ in range(n_qpus)]
    live_peak = [0] * n_qpus

    remote_gates = 0
    baseline = 0
    unschedulable = 0
    evictions = 0

    def close(key: tuple[int, int]) -> None:
        block = open_blocks.pop(key, None)
        if block is None:
            return
        roots = roots_index.get(key[0])
        if roots is not None:
            roots.discard(key)
            if not roots:
                del roots_index[key[0]]
        qpu_index[block.remote_qpu].discard(key)
        finished.append(block.freeze())

    def close_root(root: int) -> None:
        keys = roots_index.get(root)
        if not keys:
            return
        for key in sorted(keys):
            close(key)

    def evict_lru(remote_qpu: int) -> None:
        """Release the least recently used cat copy hosted on ``remote_qpu``."""
        keys = qpu_index[remote_qpu]
        if not keys:
            return
        victim = min(
            keys,
            key=lambda key: (open_blocks[key].gate_indices[-1], key[0]),
        )
        close(victim)

    def open_block(root_phys: int, root_qpu: int, remote_qpu: int) -> _OpenBlock:
        block = _OpenBlock(root_phys, root_qpu, remote_qpu)
        key = (root_phys, remote_qpu)
        open_blocks[key] = block
        roots_index.setdefault(root_phys, set()).add(key)
        qpu_index[remote_qpu].add(key)
        live = len(qpu_index[remote_qpu])
        if live > live_peak[remote_qpu]:
            live_peak[remote_qpu] = live
        return block

    def can_extend(key: tuple[int, int]) -> bool:
        """Return True when an open block at ``key`` may take one more gate."""
        block = open_blocks.get(key)
        if block is None:
            return False
        return gate_cap is None or len(block.gate_indices) < gate_cap

    def ensure_free_ports(qpu: int, count: int = 1) -> bool:
        """Free ``count`` comm ports on ``qpu``, evicting cat copies as needed.

        Returns False only when the QPU's port budget is smaller than ``count``,
        which makes the entanglement transaction impossible however it is
        scheduled. Each eviction releases exactly one live copy, so the loop
        terminates after at most one pass over the copies hosted here.
        """
        nonlocal evictions
        if ports[qpu] < count:
            return False
        while ports[qpu] - len(qpu_index[qpu]) < count:
            evict_lru(qpu)
            evictions += 1
        return True

    def emit_teleport(root_phys: int, root_qpu: int, host_qpu: int, index: int) -> bool:
        """Record a teleport round trip; return False when a port is missing.

        Both ends are checked: the host must have a port to receive the
        teleported qubit, and the source must have one to hold its half of the
        EPR pair while it is consumed.
        """
        if not ensure_free_ports(host_qpu) or not ensure_free_ports(root_qpu):
            return False
        finished.append(
            RemoteBlock(
                protocol="teleport",
                root_phys=root_phys,
                root_qpu=root_qpu,
                remote_qpu=host_qpu,
                gate_indices=(index,),
                epr_pairs=_EPR_PER_PROTOCOL["teleport"],
            )
        )
        return True

    def placement_rank(
        position: int, qubits: list[int], qpus: list[int]
    ) -> tuple[int, int]:
        """Rank a candidate root: extend > open freely > force an eviction."""
        root = qubits[position]
        source = qpus[position]
        destination = qpus[1 - position]
        if ports[destination] <= 0 or ports[source] <= 0:
            return (4, root)
        if can_extend((root, destination)):
            # Reusing a live copy needs neither a new EPR pair nor a new port.
            return (0, root)
        free = (len(qpu_index[destination]) < ports[destination]) + (
            len(qpu_index[source]) < ports[source]
        )
        # Rank 1 when neither end has to evict, 2 when one does, 3 when both do.
        return (3 - free, root)

    for index, instruction in enumerate(mapped.data):
        operation = instruction.operation
        if is_directive(operation):
            continue

        qubits = [qindex[qubit] for qubit in instruction.qubits]
        arity = len(qubits)
        if arity == 0:
            continue

        diagonal = diagonal_positions(operation)

        if arity == 1:
            if 0 not in diagonal:
                close_root(qubits[0])
            continue

        qpus = [phys_to_qpu[qubit] for qubit in qubits]

        if arity > 2:
            for qubit in qubits:
                close_root(qubit)
            host = qpus[0]
            seen = {host}
            foreign: list[tuple[int, int]] = []
            for qubit, qpu in zip(qubits[1:], qpus[1:], strict=True):
                if qpu not in seen:
                    seen.add(qpu)
                    foreign.append((qubit, qpu))
            if foreign:
                remote_gates += 1
                # Every foreign operand has to sit on the host at the same time,
                # so the host needs that many free ports at once.
                if ensure_free_ports(host, len(foreign)):
                    for qubit, qpu in foreign:
                        if emit_teleport(qubit, qpu, host, index):
                            baseline += 2
                        else:
                            unschedulable += 1
                else:
                    unschedulable += len(foreign)
            continue

        first, second = qubits
        qpu_first, qpu_second = qpus

        # Any operand this operation disturbs loses its cat copies, whether or
        # not the operation itself crosses QPUs.
        for position, qubit in ((0, first), (1, second)):
            if position not in diagonal:
                close_root(qubit)

        if qpu_first == qpu_second:
            continue

        remote_gates += 1
        candidates = [position for position in (0, 1) if position in diagonal]

        if not candidates:
            # Neither operand is diagonal: teleport the second operand to the
            # first operand's QPU and back.
            if emit_teleport(second, qpu_second, qpu_first, index):
                baseline += 2
            else:
                unschedulable += 1
            continue

        root_position = min(
            candidates, key=lambda position: placement_rank(position, qubits, qpus)
        )
        root_phys = qubits[root_position]
        root_qpu = qpus[root_position]
        remote_qpu = qpus[1 - root_position]

        if ports[remote_qpu] <= 0 or ports[root_qpu] <= 0:
            # No port anywhere can host this gate's entanglement.
            unschedulable += 1
            continue

        baseline += 1
        key = (root_phys, remote_qpu)
        if can_extend(key):
            # Extending an existing cat copy needs no new EPR pair and no new port.
            open_blocks[key].gate_indices.append(index)
            continue

        # A block at its gate cap is retired so the next gate starts a fresh one.
        close(key)
        # The copy pins a port on the remote QPU for the whole block; the root's
        # QPU needs one only while the entangler runs, but it needs one *now*.
        ensure_free_ports(remote_qpu)
        ensure_free_ports(root_qpu)
        open_block(root_phys, root_qpu, remote_qpu).gate_indices.append(index)

    for key in sorted(open_blocks):
        close(key)

    finished.sort(
        key=lambda block: (block.start_index, block.remote_qpu, block.root_phys)
    )
    epr_pairs = sum(block.epr_pairs for block in finished)

    return AggregationPlan(
        blocks=tuple(finished),
        remote_gates=remote_gates,
        epr_pairs=epr_pairs,
        baseline_epr_pairs=baseline,
        unschedulable_gates=unschedulable,
        evictions=evictions,
        peak_cat_copies=tuple(live_peak),
    )
