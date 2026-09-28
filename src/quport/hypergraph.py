# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hypergraph e-bit model for distributed quantum circuits.

What this models
----------------
Cut weight -- the number of two-qubit gates whose operands land on different
QPUs -- is the classical objective for circuit partitioning, and it is the wrong
objective for a machine that uses cat-entanglement. One EPR pair (one *e-bit*)
distributed from a root qubit ``c`` to QPU ``B`` can serve **every** gate in
which ``c`` acts diagonally while the cat copy is live, not just one. Ten gates
from the same control into the same QPU cost ten units of cut weight but a
single e-bit.

The right objective is the *connectivity-minus-one* (lambda-1) metric of
hypergraph partitioning, which is what this module computes:

.. math::

    E(\\pi) = \\sum_{P \\in \\mathcal{P}}
             \\bigl| \\{\\pi(t) : t \\in \\mathrm{partners}(P)\\}
             \\setminus \\{\\pi(\\mathrm{root}(P))\\} \\bigr|

Each *distributable packet* ``P`` is a maximal run of gates over which one root
qubit stays diagonal in the computational basis, and it contributes one e-bit
per distinct remote QPU its partners occupy. This is exactly the number of
cat-entanglements a compiler must perform, which is why
:func:`quport.aggregation.aggregate_remote_operations` reproduces this count
exactly when comm ports are unconstrained.

Why packets are partition independent
-------------------------------------
Whether a root qubit stays diagonal depends only on the gate sequence, never on
where qubits are placed. Packets can therefore be built once per circuit and
re-evaluated in ``O(sum |partners|)`` for every candidate partition, which is
what makes :func:`ebit_cost` cheap enough to sit inside the TPCCAP annealing
loop.

References
----------
The hypergraph formulation of circuit distribution follows Andres-Martinez and
Mertens, *Automated distribution of quantum circuits via hypergraph
partitioning* (Phys. Rev. A 100, 032308, 2019); packing gates onto a shared cat
copy follows the communication-aggregation idea of Wu et al., *AutoComm*
(MICRO 2022).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

from qiskit import QuantumCircuit

from quport.entanglement import diagonal_positions, is_directive

__all__ = [
    "DistributablePacket",
    "EbitReport",
    "PacketDecomposition",
    "UnpackableGate",
    "build_distributable_packets",
    "ebit_cost",
    "ebit_objective",
    "ebit_report",
    "ebit_traffic_matrix",
]

SymmetricRootPolicy = Literal["greedy", "min_index", "first_operand"]

_SYMMETRIC_ROOT_POLICIES: frozenset[str] = frozenset(
    {"greedy", "min_index", "first_operand"}
)


@dataclass(frozen=True)
class DistributablePacket:
    """A maximal run of gates that one cat copy of ``root`` can serve.

    Attributes
    ----------
    root:
        Qubit index whose computational-basis label is mirrored by the cat copy.
    partners:
        The other operand of every gate in the packet, de-duplicated and kept in
        first-use order. One e-bit is needed per distinct QPU these occupy,
        excluding the root's own QPU.
    gate_indices:
        Instruction indices of the packet's gates, in circuit order.
    gate_partners:
        The partner qubit of each entry of ``gate_indices``, same length and
        order, so per-QPU cat-copy lifetimes can be recovered exactly.
    """

    root: int
    partners: tuple[int, ...]
    gate_indices: tuple[int, ...]
    gate_partners: tuple[int, ...]

    @property
    def start_index(self) -> int:
        """Instruction index of the packet's first gate."""
        return self.gate_indices[0]

    @property
    def end_index(self) -> int:
        """Instruction index of the packet's last gate."""
        return self.gate_indices[-1]

    def size(self) -> int:
        """Number of gates the packet covers."""
        return len(self.gate_indices)


@dataclass(frozen=True)
class UnpackableGate:
    """A gate that no single cat copy can serve.

    Two kinds of operation land here: two-qubit gates with no diagonal operand
    (``swap``, ``iswap``, ``ecr``, ``rxx``, ...), and operations on three or
    more qubits, which QuPort treats conservatively because a single bipartite
    cat copy cannot bring three QPUs together.

    When such a gate spans ``k`` QPUs it is costed at ``2 * (k - 1)`` e-bits:
    teleport every foreign operand to one host QPU and teleport it back, at one
    e-bit per direction. That is the standard cost of implementing an arbitrary
    non-local two-qubit unitary and an upper bound for wider gates.
    """

    index: int
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class PacketDecomposition:
    """All distributable packets of a circuit plus the gates that resist packing."""

    n_qubits: int
    packets: tuple[DistributablePacket, ...]
    unpackable_gates: tuple[UnpackableGate, ...]
    two_qubit_gates: int

    def packed_gates(self) -> int:
        """Number of two-qubit gates covered by some packet."""
        return sum(packet.size() for packet in self.packets)


def _validate_part(part: Sequence[int], *, n_qubits: int, n_qpus: int) -> list[int]:
    """Validate a logical-qubit-to-QPU assignment for e-bit evaluation."""
    if isinstance(part, str | bytes | bytearray) or not isinstance(part, Sequence):
        raise ValueError("part must be a sequence of integer QPU indices")
    if len(part) != n_qubits:
        raise ValueError("part length must match the decomposition's qubit count")
    out: list[int] = []
    for index, qpu in enumerate(part):
        if type(qpu) is bool or not isinstance(qpu, Integral):
            raise ValueError(f"part[{index}] must be an integer QPU index")
        value = int(qpu)
        if value < 0 or value >= n_qpus:
            raise ValueError(f"part[{index}] is outside the valid QPU range")
        out.append(value)
    return out


def _validate_n_qpus(n_qpus: int) -> int:
    if type(n_qpus) is bool or not isinstance(n_qpus, Integral):
        raise ValueError("n_qpus must be an integer")
    value = int(n_qpus)
    if value <= 0:
        raise ValueError("n_qpus must be positive")
    return value


class _OpenPacket:
    """Mutable builder for a packet that is still accepting gates."""

    __slots__ = ("root", "partners", "partner_set", "gate_indices", "gate_partners")

    def __init__(self, root: int) -> None:
        self.root = root
        self.partners: list[int] = []
        self.partner_set: set[int] = set()
        self.gate_indices: list[int] = []
        self.gate_partners: list[int] = []

    def add(self, partner: int, index: int) -> None:
        if partner not in self.partner_set:
            self.partner_set.add(partner)
            self.partners.append(partner)
        self.gate_indices.append(index)
        self.gate_partners.append(partner)

    def freeze(self) -> DistributablePacket:
        return DistributablePacket(
            root=self.root,
            partners=tuple(self.partners),
            gate_indices=tuple(self.gate_indices),
            gate_partners=tuple(self.gate_partners),
        )


def build_distributable_packets(
    qc: QuantumCircuit,
    *,
    symmetric_root: SymmetricRootPolicy = "greedy",
) -> PacketDecomposition:
    """Decompose a circuit into distributable packets.

    A packet rooted at qubit ``c`` starts at the first two-qubit gate in which
    ``c`` acts diagonally and is closed by the first later operation that acts on
    ``c`` non-diagonally (an ``X``, ``H``, ``SX``, a ``CX`` that uses ``c`` as the
    target, a measurement, a reset, or any operation QuPort cannot prove
    diagonal). Barriers do not close packets because they apply no unitary.

    Parameters
    ----------
    symmetric_root:
        Which operand becomes the root when *both* act diagonally, as they do for
        ``cz``, ``cp``, ``crz`` and ``rzz``:

        - ``"greedy"`` (default): reuse an operand that already roots an open
          packet, so runs of symmetric gates keep extending one cat copy;
          ties and gates with no open packet fall back to the lower qubit index.
        - ``"min_index"``: always the lower qubit index.
        - ``"first_operand"``: always operand 0, as written in the circuit.

        All three are deterministic. Because each gate is charged to exactly one
        root, the resulting e-bit count is exact for the chosen assignment and an
        upper bound over all assignments.

    Notes
    -----
    Packets are built from qubit indices of ``qc``. Call this on the *logical*
    circuit to drive partitioning, or on a mapped physical circuit to analyse an
    existing placement -- the construction is identical either way.
    """
    if not isinstance(qc, QuantumCircuit):
        raise ValueError("qc must be a QuantumCircuit")
    if symmetric_root not in _SYMMETRIC_ROOT_POLICIES:
        allowed = ", ".join(sorted(_SYMMETRIC_ROOT_POLICIES))
        raise ValueError(f"symmetric_root must be one of: {allowed}")

    qindex = {qubit: index for index, qubit in enumerate(qc.qubits)}
    open_packets: dict[int, _OpenPacket] = {}
    packets: list[DistributablePacket] = []
    unpackable: list[UnpackableGate] = []
    two_qubit_gates = 0

    def close(qubit: int) -> None:
        packet = open_packets.pop(qubit, None)
        if packet is not None:
            packets.append(packet.freeze())

    def extend(root: int, partner: int, index: int) -> None:
        packet = open_packets.get(root)
        if packet is None:
            packet = _OpenPacket(root)
            open_packets[root] = packet
        packet.add(partner, index)

    for index, instruction in enumerate(qc.data):
        operation = instruction.operation
        if is_directive(operation):
            continue

        qubits = [qindex[qubit] for qubit in instruction.qubits]
        arity = len(qubits)
        if arity == 0:
            # Classical-only operations leave every quantum register untouched.
            continue

        diagonal = diagonal_positions(operation)

        if arity == 1:
            if 0 not in diagonal:
                close(qubits[0])
            continue

        if arity > 2:
            # A bipartite cat copy cannot merge three or more QPUs, so treat the
            # whole operation as teleport-served and release every root it touches.
            for qubit in qubits:
                close(qubit)
            unpackable.append(
                UnpackableGate(index=index, qubits=tuple(dict.fromkeys(qubits)))
            )
            continue

        two_qubit_gates += 1
        first, second = qubits
        if first == second:
            # Degenerate operand pair: not an interaction, but conservatively
            # release the qubit's packet because the operation is unclassified.
            close(first)
            continue

        candidates = [position for position in (0, 1) if position in diagonal]

        # Release the packets of every operand this gate disturbs before the
        # gate is charged, so a closed packet never contains the gate that broke it.
        for position, qubit in ((0, first), (1, second)):
            if position not in candidates:
                close(qubit)

        if not candidates:
            unpackable.append(UnpackableGate(index=index, qubits=(first, second)))
            continue

        if len(candidates) == 1:
            root_position = candidates[0]
        elif symmetric_root == "first_operand":
            root_position = 0
        elif symmetric_root == "min_index":
            root_position = 0 if first <= second else 1
        else:  # "greedy"
            first_open = first in open_packets
            second_open = second in open_packets
            if first_open == second_open:
                root_position = 0 if first <= second else 1
            else:
                root_position = 0 if first_open else 1

        root = qubits[root_position]
        partner = qubits[1 - root_position]
        extend(root, partner, index)

    for qubit in sorted(open_packets):
        packets.append(open_packets[qubit].freeze())

    packets.sort(key=lambda packet: (packet.start_index, packet.root))
    return PacketDecomposition(
        n_qubits=qc.num_qubits,
        packets=tuple(packets),
        unpackable_gates=tuple(unpackable),
        two_qubit_gates=two_qubit_gates,
    )


def ebit_cost(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
) -> int:
    """Return the lambda-1 e-bit cost of ``part``.

    One e-bit per (packet, distinct remote QPU) pair, plus ``2 * (k - 1)`` for
    each unpackable gate spanning ``k`` QPUs. This is the exact number of EPR
    pairs a cat-entanglement compiler consumes when comm ports are unconstrained.
    """
    n_qpus_value = _validate_n_qpus(n_qpus)
    assignments = _validate_part(
        part, n_qubits=decomposition.n_qubits, n_qpus=n_qpus_value
    )
    return _ebit_cost_fast(decomposition, assignments, n_qpus_value)


def _ebit_cost_fast(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
) -> int:
    """Unvalidated e-bit evaluation for hot loops (partition search)."""
    return _ebit_objective_fast(decomposition, part, n_qpus, None)[0]


def ebit_objective(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
    dist: Sequence[Sequence[int]] | None = None,
) -> tuple[int, float]:
    """Return ``(ebits, hop_weighted_ebits)`` for ``part``.

    The hop-weighted figure charges each e-bit by the QPU-graph distance it must
    cross, which is the quantity a limited-degree interconnect actually pays:
    entanglement swapping consumes a link on every hop. On an all-to-all fabric
    every distance is 1 and the two figures coincide.

    Parameters
    ----------
    dist:
        All-pairs QPU distances, typically ``arch.qpu_shortest_paths().dist``.
        When omitted every pair is charged one hop.
    """
    n_qpus_value = _validate_n_qpus(n_qpus)
    assignments = _validate_part(
        part, n_qubits=decomposition.n_qubits, n_qpus=n_qpus_value
    )
    distances = _validate_distances(dist, n_qpus_value)
    return _ebit_objective_fast(decomposition, assignments, n_qpus_value, distances)


def _validate_distances(
    dist: Sequence[Sequence[int]] | None, n_qpus: int
) -> list[list[float]] | None:
    """Validate an all-pairs distance table used for hop weighting."""
    if dist is None:
        return None
    if isinstance(dist, str | bytes | bytearray) or not isinstance(dist, Sequence):
        raise ValueError("dist must be a sequence of rows")
    if len(dist) != n_qpus:
        raise ValueError("dist dimensions do not match n_qpus")
    out: list[list[float]] = []
    for row in dist:
        if isinstance(row, str | bytes | bytearray) or not isinstance(row, Sequence):
            raise ValueError("dist rows must be sequences")
        if len(row) != n_qpus:
            raise ValueError("dist dimensions do not match n_qpus")
        values: list[float] = []
        for entry in row:
            if type(entry) is bool or not isinstance(entry, Integral):
                raise ValueError("dist must contain integer distances")
            value = int(entry)
            if value < 0:
                raise ValueError("dist must contain non-negative distances")
            values.append(float(value))
        out.append(values)
    return out


def _ebit_objective_fast(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
    dist: Sequence[Sequence[float]] | None,
    traffic: list[list[float]] | None = None,
) -> tuple[int, float]:
    """Unvalidated e-bit count and hop-weighted cost for hot loops.

    When ``traffic`` is given -- a pre-zeroed ``n_qpus`` square matrix -- the
    same sweep also accumulates the symmetric per-pair EPR demand into it. The
    cost and the traffic therefore come from one traversal and one set of
    decisions, so a congestion term built from the matrix cannot describe a
    different plan from the one the cost prices.
    """
    seen_stamp = [-1] * n_qpus
    stamp = 0
    count = 0
    weighted = 0.0

    for packet in decomposition.packets:
        stamp += 1
        root_qpu = part[packet.root]
        row = None if dist is None else dist[root_qpu]
        for partner in packet.partners:
            qpu = part[partner]
            if qpu != root_qpu and seen_stamp[qpu] != stamp:
                seen_stamp[qpu] = stamp
                count += 1
                weighted += 1.0 if row is None else row[qpu]
                if traffic is not None:
                    traffic[root_qpu][qpu] += 1.0
                    traffic[qpu][root_qpu] += 1.0

    for gate in decomposition.unpackable_gates:
        host = part[gate.qubits[0]]
        row = None if dist is None else dist[host]
        stamp += 1
        seen_stamp[host] = stamp
        for qubit in gate.qubits[1:]:
            qpu = part[qubit]
            if seen_stamp[qpu] != stamp:
                seen_stamp[qpu] = stamp
                count += 2
                weighted += 2.0 * (1.0 if row is None else row[qpu])
                if traffic is not None:
                    traffic[host][qpu] += 2.0
                    traffic[qpu][host] += 2.0

    return count, weighted


def _teleport_host_and_foreign(
    gate: UnpackableGate, part: Sequence[int]
) -> tuple[int, tuple[int, ...]]:
    """Return the host QPU of an unpackable gate and the QPUs teleported into it.

    The host is the QPU of the gate's *first* operand, matching how
    :func:`quport.aggregation.aggregate_remote_operations` and the schedule
    estimators pick the leading QPU of a multi-QPU operation, so every view of a
    circuit charges the same links.
    """
    host = part[gate.qubits[0]]
    foreign: list[int] = []
    seen = {host}
    for qubit in gate.qubits[1:]:
        qpu = part[qubit]
        if qpu not in seen:
            seen.add(qpu)
            foreign.append(qpu)
    return host, tuple(foreign)


def ebit_traffic_matrix(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
) -> list[list[float]]:
    """Return the symmetric QPU-to-QPU EPR demand implied by ``part``.

    Unlike :func:`quport.network.compute_traffic_matrix`, which counts one unit
    per cut gate, this counts one unit per e-bit -- the quantity that actually
    crosses the interconnect once gates are aggregated onto shared cat copies.

    The matrix is filled by the same sweep that computes :func:`ebit_cost`, so
    its entries always sum to twice that count.
    """
    n_qpus_value = _validate_n_qpus(n_qpus)
    assignments = _validate_part(
        part, n_qubits=decomposition.n_qubits, n_qpus=n_qpus_value
    )

    traffic = [[0.0] * n_qpus_value for _ in range(n_qpus_value)]
    _ebit_objective_fast(decomposition, assignments, n_qpus_value, None, traffic)
    return traffic


@dataclass(frozen=True)
class EbitReport:
    """Diagnostics for the e-bit cost of a partition.

    Attributes
    ----------
    ebits:
        Total EPR pairs required with aggregation (the lambda-1 cost).
    baseline_ebits:
        EPR pairs required without aggregation: one per cut two-qubit gate that
        has a diagonal operand, and ``2 * (k - 1)`` per unpackable gate spanning
        ``k`` QPUs. This is what a per-gate telegate compiler consumes.
    reduction:
        ``1 - ebits / baseline_ebits``, or ``0.0`` when no e-bits are needed.
    peak_cat_copies:
        Per QPU, the largest number of cat copies that are simultaneously live.
        A value above the QPU's comm-port count means the unconstrained plan is
        not directly realisable and ports will serialise it.
    pair_ebits:
        E-bits per unordered QPU pair, sorted by pair.
    """

    ebits: int
    baseline_ebits: int
    reduction: float
    packets: int
    active_packets: int
    packed_gates: int
    cut_gates: int
    unpackable_ebits: int
    peak_cat_copies: tuple[int, ...]
    pair_ebits: tuple[tuple[tuple[int, int], int], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready representation of this report."""
        return {
            "ebits": self.ebits,
            "baseline_ebits": self.baseline_ebits,
            "reduction": self.reduction,
            "packets": self.packets,
            "active_packets": self.active_packets,
            "packed_gates": self.packed_gates,
            "cut_gates": self.cut_gates,
            "unpackable_ebits": self.unpackable_ebits,
            "peak_cat_copies": list(self.peak_cat_copies),
            "pair_ebits": [
                {"qpus": [pair[0], pair[1]], "ebits": count}
                for pair, count in self.pair_ebits
            ],
        }


def ebit_report(
    decomposition: PacketDecomposition,
    part: Sequence[int],
    n_qpus: int,
) -> EbitReport:
    """Compute a full e-bit diagnostic for ``part``."""
    n_qpus_value = _validate_n_qpus(n_qpus)
    assignments = _validate_part(
        part, n_qubits=decomposition.n_qubits, n_qpus=n_qpus_value
    )

    ebits = 0
    baseline = 0
    cut_gates = 0
    packed_gates = 0
    active_packets = 0
    unpackable_ebits = 0
    pair_counts: dict[tuple[int, int], int] = {}
    # (index, +1/-1) events per QPU for the concurrent cat-copy sweep.
    events: list[list[tuple[int, int, int]]] = [[] for _ in range(n_qpus_value)]

    for packet in decomposition.packets:
        root_qpu = assignments[packet.root]
        windows: dict[int, tuple[int, int]] = {}
        for gate_index, partner in zip(
            packet.gate_indices, packet.gate_partners, strict=True
        ):
            qpu = assignments[partner]
            if qpu == root_qpu:
                continue
            cut_gates += 1
            baseline += 1
            window = windows.get(qpu)
            if window is None:
                windows[qpu] = (gate_index, gate_index)
            else:
                windows[qpu] = (window[0], gate_index)

        if windows:
            active_packets += 1
        ebits += len(windows)
        packed_gates += packet.size()
        for qpu, (start, end) in windows.items():
            pair = (root_qpu, qpu) if root_qpu < qpu else (qpu, root_qpu)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            # The cat copy lives on the remote QPU for the whole window; the
            # root QPU only needs a port while the EPR pair is being consumed.
            events[qpu].append((start, 0, 1))
            events[qpu].append((end, 1, -1))

    for gate in decomposition.unpackable_gates:
        host, foreign = _teleport_host_and_foreign(gate, assignments)
        if not foreign:
            continue
        cost = 2 * len(foreign)
        ebits += cost
        baseline += cost
        unpackable_ebits += cost
        for qpu in foreign:
            pair = (host, qpu) if host < qpu else (qpu, host)
            pair_counts[pair] = pair_counts.get(pair, 0) + 2
            # The teleported operand occupies a slot on the host for the gate.
            events[host].append((gate.index, 0, 1))
            events[host].append((gate.index, 1, -1))

    peak: list[int] = []
    for qpu_events in events:
        # Sort so that a copy opening at index i is counted alongside one closing
        # at the same index: both are live while that instruction executes.
        qpu_events.sort()
        live = 0
        best = 0
        for _index, _order, delta in qpu_events:
            live += delta
            if live > best:
                best = live
        peak.append(best)

    reduction = 1.0 - (ebits / baseline) if baseline > 0 else 0.0

    return EbitReport(
        ebits=ebits,
        baseline_ebits=baseline,
        reduction=reduction,
        packets=len(decomposition.packets),
        active_packets=active_packets,
        packed_gates=packed_gates,
        cut_gates=cut_gates,
        unpackable_ebits=unpackable_ebits,
        peak_cat_copies=tuple(peak),
        pair_ebits=tuple(sorted(pair_counts.items())),
    )
