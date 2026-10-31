# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import csv
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, TypeAlias, TypeVar, cast

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    AnnotatedOperation,
    BreakLoopOp,
    ContinueLoopOp,
    ControlFlowOp,
    Gate,
)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins

from quport._validation import validate_nonnegative_integral
from quport.architecture import MultiQPUArchitecture
from quport.config import InterTopology, IntraTopology, LatencyModel, MultiQPUConfig
from quport.cost import CostBreakdown, estimate_cost
from quport.distributed import (
    _hold_measurement_order,
    _release_order_barriers,
    _route_keeping_order,
)
from quport.entanglement import is_standard_gate, is_standard_namesake
from quport.hypergraph import build_distributable_packets
from quport.interaction import (
    extract_temporal_twoq_weights,
    extract_twoq_weights,
    validate_temporal_decay,
)
from quport.layout import compute_layout_hints, naive_layout
from quport.metrics import CircuitMetrics, compute_metrics
from quport.partition import (
    PartitionDiagnostics,
    PartitionResult,
    balanced_greedy_partition,
    ebit_partition,
    heavy_edge_clustering_partition,
    tpccap_partition,
    tpccap_sa_partition,
)

BenchmarkRow: TypeAlias = dict[str, float | str]
SweepSummaryRow: TypeAlias = dict[str, float | str]
_StringT = TypeVar("_StringT", bound=str)
_BENCHMARK_METHOD_IDS = {
    "baseline": 0.0,
    "balanced": 1.0,
    "tpccap": 2.0,
    "tpccap_sa": 3.0,
    "cluster": 4.0,
    "ebit": 5.0,
}
_BENCHMARK_METHOD_LABELS = {
    method_id: strategy for strategy, method_id in _BENCHMARK_METHOD_IDS.items()
}


def _validate_out_csv(out_csv: object, *, label: str = "out_csv") -> str:
    """Check an output path before a run spends minutes on the results.

    The file is only opened once every trial has run, so a path that can never
    be written -- not a path at all, empty, holding a NUL, an existing
    directory, or under a file where its directory should be -- would
    otherwise throw the whole run away at the end.
    """
    if not isinstance(out_csv, str | os.PathLike):
        raise ValueError(f"{label} must be a filesystem path")
    path = os.fspath(out_csv)
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} must be a non-empty path")
    if "\x00" in path:
        raise ValueError(f"{label} must not contain a NUL character")
    # Path() drops a trailing separator and a final ".", so look at the text:
    # either names a directory however the rest of the path turns out.
    if os.path.basename(path) in ("", ".", ".."):
        raise ValueError(f"{label} {path!r} names a directory, not a file")
    target = Path(path)
    try:
        if target.is_dir():
            raise ValueError(f"{label} {path!r} names a directory, not a file")
        # The directory is created when the file is written, which works only
        # if the nearest part of the path that already exists is a directory.
        # A symbolic link counts as existing even when broken or looping: the
        # directory cannot be created in its place.
        for ancestor in target.parents:
            if ancestor.is_symlink() or ancestor.exists():
                if not ancestor.is_dir():
                    raise ValueError(
                        f"{label} {path!r} cannot be created: {str(ancestor)!r} "
                        "is not a directory"
                    )
                break
        if target.is_symlink() and not target.exists():
            # Writing follows the link, so its own target must be creatable.
            resolved = Path(os.path.realpath(path))
            if resolved.is_symlink() or not resolved.parent.is_dir():
                raise ValueError(
                    f"{label} {path!r} cannot be created: it is a broken "
                    "symbolic link"
                )
    except OSError as exc:  # a name too long for the filesystem, for one
        raise ValueError(f"{label} {path!r} cannot be used: {exc}") from None
    return path


def _open_csv_for_write(out_csv: str) -> TextIO:
    """Open ``out_csv`` for writing, creating the directory it names.

    A sweep can run for minutes before it reaches this line. Leaving the
    directory to the caller meant the whole run was lost to a FileNotFoundError
    once it finished, so the parent is created here as the bundle writers and
    the ``--out-dir`` commands already do.
    """
    parent = Path(out_csv).parent
    if parent != Path(""):
        parent.mkdir(parents=True, exist_ok=True)
    return open(out_csv, "w", newline="", encoding="utf-8")


def benchmark_method_labels() -> dict[float, str]:
    """Return a copy of the stable numeric benchmark-method label map."""
    return dict(_BENCHMARK_METHOD_LABELS)


def _validate_positive_int(value: object, *, label: str) -> int:
    """Return a positive integer, rejecting bools and non-integral values."""
    out = validate_nonnegative_integral(value, label=label)
    if out <= 0:
        raise ValueError(f"{label} must be positive")
    return out


def _validate_string_sequence(
    values: Sequence[_StringT],
    *,
    label: str,
) -> tuple[_StringT, ...]:
    """Validate an API sequence of strings without treating a bare string as a sequence."""
    if isinstance(values, str | bytes | bytearray):
        raise ValueError(f"{label} must be a sequence of strings, not a string")
    if not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of strings")
    out = tuple(values)
    for idx, value in enumerate(out):
        if not isinstance(value, str):
            raise ValueError(f"{label}[{idx}] must be a string")
    return out


def _validate_strategy_sequence(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """Validate benchmark/sweep strategy names, preserving caller order."""
    selected = _validate_string_sequence(values, label=label)
    if not selected:
        raise ValueError(f"{label} must contain at least one strategy")

    seen: set[str] = set()
    duplicates: list[str] = []
    for strategy in selected:
        if strategy in seen and strategy not in duplicates:
            duplicates.append(strategy)
        seen.add(strategy)
    if duplicates:
        duplicate_list = ", ".join(duplicates)
        raise ValueError(f"{label} contains duplicate strategies: {duplicate_list}")

    unknown = sorted(set(selected) - set(_BENCHMARK_METHOD_IDS))
    if unknown:
        allowed = ", ".join(_BENCHMARK_METHOD_IDS)
        raise ValueError(
            f"Unknown {label}: {', '.join(unknown)}. Use any of: {allowed}."
        )
    return selected


def _validate_nonnegative_int_sequence(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    """Validate a non-string sequence of non-negative integers."""
    if isinstance(values, str | bytes | bytearray) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of non-negative integers")
    return tuple(
        validate_nonnegative_integral(value, label=f"{label}[{idx}]")
        for idx, value in enumerate(values)
    )


@dataclass(frozen=True)
class MapResult:
    """Outputs of a mapping+transpilation run."""

    mapped_circuit: QuantumCircuit
    cfg: MultiQPUConfig
    partition: list[int]
    partition_cut: float
    strategy: str
    partition_diagnostics: PartitionDiagnostics | None
    mapping_time_s: float
    transpile_time_s: float
    metrics: CircuitMetrics
    cost: CostBreakdown


def _translate_to_basis(
    qc: QuantumCircuit, basis_gates: Sequence[str], seed: int | None
) -> QuantumCircuit:
    """Translate the circuit into a safe basis before routing.

    This avoids direction-fixing errors for gates that cannot be auto-flipped by the transpiler.
    """
    return transpile(
        _expand_standard_namesakes(qc),
        basis_gates=list(basis_gates),
        optimization_level=0,
        seed_transpiler=seed,
    )


def _refuse_classical_register_named_q(qc: QuantumCircuit) -> None:
    """Refuse a classical register named ``q`` before laying the circuit out.

    Qiskit's layout stage always names the physical register ``q``, and a
    circuit cannot hold two registers of one name, so such a circuit fails
    inside Qiskit once a layout is applied. Renaming it is not an option: a
    condition on the register refers to it as it is.
    """
    if any(creg.name == "q" for creg in qc.cregs):
        raise ValueError(
            "the circuit has a classical register named 'q', which collides with "
            "the physical register Qiskit's layout creates; rename it"
        )


def _expand_standard_namesakes(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace each gate named like a Qiskit standard gate, but not one, by its definition.

    Qiskit's basis translation finds a gate's decomposition by its name, so a
    custom ``swap`` defined as a single CX is translated as a real SWAP, and a
    parameterless custom ``rzz`` as Qiskit's ``rzz(theta)`` with ``theta``
    left unbound -- neither computes the circuit. Inlining the definition
    first keeps what the circuit says; an OpenQASM 2 program that defines its
    own ``swap`` loads as exactly such a gate. Circuits without one are
    returned unchanged.
    """
    if not _has_standard_namesake(circuit):
        return circuit
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        _append_expanded(
            out,
            instruction.operation,
            list(instruction.qubits),
            list(instruction.clbits),
        )
    return out


def _has_standard_namesake(circuit: QuantumCircuit) -> bool:
    return any(_holds_namesake(i.operation) for i in circuit.data)


def _holds_namesake(operation: Any) -> bool:
    """True when ``operation`` is a namesake or contains one, at any depth.

    Qiskit unrolls a custom gate before translating what it contains, so a
    namesake used inside another custom gate reaches the name-keyed lookup
    just as surely as one written at the top level.
    """
    if is_standard_namesake(operation):
        return True
    if isinstance(operation, ControlFlowOp):
        return any(_has_standard_namesake(block) for block in operation.blocks)
    if isinstance(operation, AnnotatedOperation):
        # Synthesis builds the controlled or powered base from its name too.
        return _holds_namesake(operation.base_op)
    definition = _user_definition(operation)
    return definition is not None and _has_standard_namesake(definition)


def _user_definition(operation: Any) -> QuantumCircuit | None:
    """The definition of a user-defined operation, or ``None``.

    Qiskit's own gates -- standard ones and the library's -- are built from
    Qiskit's gates and never contain a namesake, so their definitions, which
    can be costly to synthesise, are not opened.
    """
    if is_standard_gate(operation) or isinstance(operation, ControlFlowOp):
        return None
    # Qiskit's singleton classes are made at runtime and carry no module.
    module = type(operation).__module__ or ""
    if module.startswith("qiskit.circuit.library"):
        return None
    definition = getattr(operation, "definition", None)
    return definition if isinstance(definition, QuantumCircuit) else None


def _append_expanded(
    out: QuantumCircuit, operation: Any, qubits: list[Any], clbits: list[Any]
) -> None:
    """Append ``operation``, inlining it -- recursively -- if it holds a namesake."""
    if isinstance(operation, ControlFlowOp):
        operation = operation.replace_blocks(
            [_expand_standard_namesakes(block) for block in operation.blocks]
        )
    elif isinstance(operation, AnnotatedOperation):
        if _holds_namesake(operation):
            operation = _without_namesakes(operation)
    elif _holds_namesake(operation):
        definition = _definition_to_expand(operation)
        qubit_of = dict(zip(definition.qubits, qubits))
        clbit_of = dict(zip(definition.clbits, clbits))
        out.global_phase += definition.global_phase
        for inner in definition.data:
            _append_expanded(
                out,
                inner.operation,
                [qubit_of[qubit] for qubit in inner.qubits],
                [clbit_of[clbit] for clbit in inner.clbits],
            )
        return
    out.append(operation, qubits, clbits)


def _definition_to_expand(operation: Any) -> QuantumCircuit:
    """The definition a namesake, or a gate holding one, is replaced by."""
    definition = operation.definition
    if definition is None:
        raise ValueError(
            f"gate {operation.name!r} shares a Qiskit standard gate's name "
            "without being that gate, and has no definition to translate "
            "it by; rename it"
        )
    return definition


def _without_namesakes(operation: Any) -> Any:
    """An equivalent of ``operation`` that holds no namesake, kept whole.

    The base of an annotated operation cannot be inlined -- the modifiers
    apply to it as one gate -- so a base holding a namesake is replaced by a
    gate built from its expanded definition, under a name no standard gate
    has, which synthesis can only build from that definition.
    """
    if isinstance(operation, AnnotatedOperation):
        return AnnotatedOperation(
            _without_namesakes(operation.base_op), operation.modifiers
        )
    # A copy: the expansion hands back a definition with nothing to expand.
    expanded = _expand_standard_namesakes(_definition_to_expand(operation)).copy(
        name=f"{operation.name}_expanded"
    )
    if isinstance(operation, Gate):
        return expanded.to_gate()
    return expanded.to_instruction()


def validate_transpiler_settings(cfg: MultiQPUConfig) -> None:
    """Check the config's transpiler settings name things Qiskit has.

    ``MultiQPUConfig`` checks only that they are non-empty strings; Qiskit
    rejects an unknown layout or routing plugin, a basis gate it does not
    define, or a basis gate on other than one or two qubits next to a coupling
    map -- which every QuPort pipeline routes on -- but only once ``transpile``
    runs, after partitioning. This applies the same rules up front, read from
    Qiskit's own registries, so what passes here is what ``transpile`` accepts.
    """
    for stage, method in (
        ("layout", cfg.layout_method),
        ("routing", cfg.routing_method),
    ):
        available = list_stage_plugins(stage)
        if method not in available:
            raise ValueError(
                f"{stage}_method {method!r} is not a Qiskit {stage} plugin; "
                f"choose one of: {', '.join(available)}"
            )
    standard = get_standard_gate_name_mapping()
    for gate in cfg.basis_gates:
        if gate in CONTROL_FLOW_OP_NAMES:
            continue
        operation = standard.get(gate)
        if operation is None:
            raise ValueError(
                f"basis_gates entry {gate!r} is not a standard Qiskit gate"
            )
        if operation.num_qubits not in (1, 2):
            raise ValueError(
                f"basis_gates entry {gate!r} acts on {operation.num_qubits} qubits; "
                "Qiskit routes on a coupling map only with gates on one or two"
            )


def _route_globally(
    qc: QuantumCircuit,
    cfg: MultiQPUConfig,
    arch: MultiQPUArchitecture,
    coupling: CouplingMap,
    initial_layout: Sequence[int],
    seed: int | None,
) -> QuantumCircuit:
    """Route ``qc`` on the whole machine, explaining a disconnected one.

    Global routing can only move state along coupling-map edges, so a gate
    whose operands sit in parts of the machine no path joins -- with no comm
    ports, or an interconnect that leaves QPUs unreachable -- cannot be routed,
    and Qiskit fails deep inside routing. Qiskit is still the judge of that:
    a gate it optimizes away, or a control-flow block that only applies
    single-qubit gates on both sides, routes fine. Only when routing fails
    because operands sit in disconnected components is the reason looked for,
    and reported as a ``ValueError`` naming the QPUs; any other failure, such
    as a routing method that cannot handle control flow, is Qiskit's to report.
    The distributed compiler handles the same machine: it schedules such
    operations as unschedulable rather than routing them.
    """
    # Qiskit's routing stage can swap two final measurements into one bit.
    held = _hold_measurement_order(qc)
    try:
        routed = _route_keeping_order(
            held,
            coupling_map=coupling,
            initial_layout=list(initial_layout),
            basis_gates=list(cfg.basis_gates),
            optimization_level=cfg.optimization_level,
            layout_method=cfg.layout_method,
            routing_method=cfg.routing_method,
            seed_transpiler=seed,
        )
    except BaseException as exc:
        # Qiskit reports disconnected operands as a TranspilerError, except
        # that its SABRE router panics instead on a control-flow block whose
        # gates span the gap; either way the reason is looked for.
        disconnected = isinstance(exc, TranspilerError) and "disconnected" in str(exc)
        if not (disconnected or _is_rust_panic(exc)):
            raise
        qpus = _unreachable_interaction(qc, initial_layout, arch, coupling)
        if qpus is None:
            raise
        raise ValueError(
            f"global routing cannot connect QPUs {qpus}: no inter-QPU path "
            "joins them (comm_qubits_per_qpu=0 or a disconnected "
            "inter_topology); compile_distributed schedules such operations "
            "as unschedulable instead"
        ) from exc
    return _release_order_barriers(routed)


def _is_rust_panic(exc: BaseException) -> bool:
    """True for the exception a panic in Qiskit's Rust core surfaces as."""
    kind = type(exc)
    return kind.__name__ == "PanicException" and kind.__module__ == "pyo3_runtime"


def _unreachable_interaction(
    qc: QuantumCircuit,
    layout: Sequence[int],
    arch: MultiQPUArchitecture,
    coupling: CouplingMap,
) -> list[int] | None:
    """The QPUs of a gate whose operands no coupling-map path joins, if any.

    Gates inside control-flow blocks count too, on the qubits their block
    is applied to; a block itself is not an interaction.
    """
    parent = list(range(arch.n_phys))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for u, v in coupling.get_edges():
        parent[find(u)] = find(v)

    def visit(
        circuit: QuantumCircuit, physical_of: Mapping[Any, int]
    ) -> list[int] | None:
        for instruction in circuit.data:
            operation = instruction.operation
            if isinstance(operation, ControlFlowOp):
                for block in operation.blocks:
                    inner = {
                        block_qubit: physical_of[qubit]
                        for block_qubit, qubit in zip(block.qubits, instruction.qubits)
                    }
                    found = visit(block, inner)
                    if found is not None:
                        return found
                continue
            # A loop's break or continue spans the loop's qubits but couples none.
            if (
                len(instruction.qubits) < 2
                or getattr(operation, "_directive", False)
                or isinstance(operation, (BreakLoopOp, ContinueLoopOp))
            ):
                continue
            physical = [physical_of[qubit] for qubit in instruction.qubits]
            if len({find(p) for p in physical}) > 1:
                return sorted({arch.qpu_of_phys(p) for p in physical})
        return None

    return visit(qc, dict(zip(qc.qubits, layout)))


def map_and_transpile(
    qc: QuantumCircuit,
    cfg: MultiQPUConfig,
    latency: LatencyModel | None = None,
    seed: int | None = None,
    strategy: str = "balanced",
    temporal_decay: float | None = None,
) -> MapResult:
    """End-to-end pipeline: partition -> initial layout -> transpile -> metrics -> cost.

    Parameters
    ----------
    qc:
        Input circuit (logical qubits).
    cfg:
        Multi-QPU architecture + transpiler configuration.
    latency:
        Cost model for latency estimation.
    seed:
        Random seed for reproducibility (partition refinement, SABRE).
    strategy:
        Partitioning strategy:
        - "balanced": balanced greedy partition + local refinement (recommended baseline)
        - "cluster" : heavy-edge clustering baseline
        - "tpccap"    : topology+port+congestion aware partitioner (novel)
        - "tpccap_sa" : TPCCAP + simulated annealing refinement (best)
        - "ebit"      : TPCCAP-SA minimising hop-scaled e-bit demand instead of
          cut weight, i.e. the EPR pairs left after communication aggregation
    temporal_decay:
        Interaction weighting for the topology-aware strategies, matching the
        argument of the same name on :func:`quport.compiler.compile_distributed`.
        A value in (0, 1) emphasises earlier two-qubit gates; ``1.0`` is plain
        interaction counts. It applies to ``tpccap``, ``tpccap_sa`` and ``ebit``
        alike, so a run isolates the annealing rather than also changing the
        objective's inputs.

        Leaving it as ``None`` keeps the historical per-strategy behaviour, in
        which ``tpccap`` uses uniform counts while ``tpccap_sa`` and ``ebit``
        use a decay of 0.98. That default is preserved so existing benchmark
        numbers do not move, but it means the strategies differ in more than the
        search; pass an explicit value to compare them on equal terms.

    Returns
    -------
    MapResult
    """
    latency = latency or LatencyModel()
    if seed is not None:
        seed = validate_nonnegative_integral(seed, label="seed")

    _refuse_classical_register_named_q(qc)
    # Building the architecture validates the config's integer fields, which
    # the capacity check below would otherwise trip over as a TypeError.
    arch = MultiQPUArchitecture(cfg)
    if qc.num_qubits > cfg.total_physical_qubits():
        raise ValueError(
            f"Logical qubits={qc.num_qubits} exceed physical qubits={cfg.total_physical_qubits()} in config."
        )

    coupling = arch.build_coupling_map()

    # Step 0: translate into basis (removes CY, etc.)
    qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)

    # Step 1: extract interaction weights for partitioning
    t0 = time.perf_counter()
    weights = extract_twoq_weights(qc_basis)
    capacity = cfg.capacity_per_qpu()

    def topology_weights(
        default_decay: float | None,
    ) -> Mapping[tuple[int, int], float]:
        """Interaction weights for a topology-aware strategy.

        ``temporal_decay=None`` keeps each strategy's default -- uniform counts
        for tpccap, a decay of 0.98 for tpccap_sa and ebit; an explicit value is
        applied to all three so they can be compared on equal terms.

        Validation happens here rather than up front so that, exactly as in
        :func:`quport.compiler.compile_distributed`, a decay supplied alongside a
        strategy that does not use it is ignored rather than rejected.
        """
        effective = default_decay
        if temporal_decay is not None:
            effective = validate_temporal_decay(temporal_decay, label="temporal_decay")
        # decay == 1.0 yields exactly the uniform counts, so skip recomputing them.
        if effective is None or effective >= 1.0:
            return weights
        return extract_temporal_twoq_weights(qc_basis, decay=effective)

    partition_diag: PartitionDiagnostics | None = None
    comm_mode = "topk"

    if strategy == "balanced":
        pres: PartitionResult = balanced_greedy_partition(
            n=qc_basis.num_qubits,
            weights=weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        comm_mode = "topk"

    elif strategy == "tpccap":
        sp = arch.qpu_shortest_paths()
        pres, diag = tpccap_partition(
            n=qc_basis.num_qubits,
            weights=topology_weights(None),
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        partition_diag = diag
        comm_mode = "diverse"

    elif strategy == "tpccap_sa":
        sp = arch.qpu_shortest_paths()
        # time-decayed weights often improve early remote behavior on random circuits
        pres, diag, _anneal = tpccap_sa_partition(
            n=qc_basis.num_qubits,
            weights=topology_weights(0.98),
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        partition_diag = diag
        comm_mode = "diverse"

    elif strategy == "ebit":
        sp = arch.qpu_shortest_paths()
        # The same objective `compile_distributed` uses for this strategy,
        # penalty rescaling included, so `quport map/bench/sweep --strategy ebit`
        # measure the strategy the rest of QuPort calls by that name.
        pres, diag, _anneal = ebit_partition(
            n=qc_basis.num_qubits,
            weights=topology_weights(0.98),
            n_qpus=cfg.n_qpus,
            capacity=capacity,
            comm_ports_per_qpu=max(0, cfg.comm_qubits_per_qpu),
            sp=sp,
            packets=build_distributable_packets(qc_basis),
            seed=seed,
        )
        part = pres.part
        cut = pres.cut
        partition_diag = diag
        comm_mode = "diverse"

    elif strategy == "cluster":
        part = heavy_edge_clustering_partition(
            n=qc_basis.num_qubits,
            weights=weights,
            n_qpus=cfg.n_qpus,
            capacity=capacity,
        )
        cut = 0
        for (i, j), w in weights.items():
            if part[i] != part[j]:
                cut += w
        comm_mode = "topk"

    else:
        raise ValueError(
            "Unknown strategy. Use 'balanced', 'cluster', 'ebit', 'tpccap', "
            "or 'tpccap_sa'."
        )

    hints = compute_layout_hints(qc_basis, arch, part, comm_mode=comm_mode)
    mapping_time = time.perf_counter() - t0

    # Step 2: transpile on the global coupling map using SABRE
    t1 = time.perf_counter()
    mapped = _route_globally(qc_basis, cfg, arch, coupling, hints.initial_layout, seed)
    transpile_time = time.perf_counter() - t1

    metrics = compute_metrics(mapped, arch)
    cost = estimate_cost(metrics, latency)

    return MapResult(
        mapped_circuit=mapped,
        cfg=cfg,
        partition=part,
        partition_cut=cut,
        strategy=strategy,
        partition_diagnostics=partition_diag,
        mapping_time_s=mapping_time,
        transpile_time_s=transpile_time,
        metrics=metrics,
        cost=cost,
    )


def transpile_baseline(
    qc: QuantumCircuit,
    cfg: MultiQPUConfig,
    latency: LatencyModel | None = None,
    seed: int | None = None,
) -> MapResult:
    """Baseline: translate to basis, place logical qubit ``i`` on physical qubit
    ``i``, and let the transpiler route (no partition hints).

    The identity layout is deliberate -- it is deterministic and exposes the
    inter-QPU penalty an unpartitioned placement pays -- so ``cfg.layout_method``
    does not apply: Qiskit skips its layout stage when given an initial layout.
    """
    latency = latency or LatencyModel()
    if seed is not None:
        seed = validate_nonnegative_integral(seed, label="seed")

    _refuse_classical_register_named_q(qc)
    # Building the architecture validates the config's integer fields, which
    # the capacity check below would otherwise trip over as a TypeError.
    arch = MultiQPUArchitecture(cfg)
    if qc.num_qubits > cfg.total_physical_qubits():
        raise ValueError(
            f"Logical qubits={qc.num_qubits} exceed physical qubits={cfg.total_physical_qubits()} in config."
        )

    coupling = arch.build_coupling_map()
    qc_basis = _translate_to_basis(qc, cfg.basis_gates, seed)

    # Optional: initial layout = identity (logical i -> physical i)
    # This baseline is deterministic and exposes inter-QPU penalties.
    init_layout = naive_layout(qc_basis.num_qubits)

    t0 = time.perf_counter()
    mapped = _route_globally(qc_basis, cfg, arch, coupling, init_layout, seed)
    t1 = time.perf_counter()

    metrics = compute_metrics(mapped, arch)
    cost = estimate_cost(metrics, latency)

    return MapResult(
        mapped_circuit=mapped,
        cfg=cfg,
        partition=[arch.qpu_of_phys(physical) for physical in init_layout],
        partition_cut=-1.0,
        strategy="baseline",
        partition_diagnostics=None,
        mapping_time_s=0.0,
        transpile_time_s=t1 - t0,
        metrics=metrics,
        cost=cost,
    )


def random_benchmark_circuit(
    n_logical: int,
    depth: int,
    seed: int,
) -> QuantumCircuit:
    """Random circuit generator suitable for mapping benchmarks."""
    n_logical_value = validate_nonnegative_integral(n_logical, label="n_logical")
    depth_value = validate_nonnegative_integral(depth, label="depth")
    seed_value = validate_nonnegative_integral(seed, label="seed")
    if n_logical_value == 0:
        # Qiskit's generator rejects zero qubits with an unrelated error about
        # its operand distribution; the empty circuit is what was asked for,
        # and every pipeline handles it.
        return QuantumCircuit(0)
    return random_circuit(
        num_qubits=n_logical_value,
        depth=depth_value,
        max_operands=2,
        measure=False,
        seed=seed_value,
    )


def benchmark_random_circuits(
    cfg: MultiQPUConfig,
    n_logical: int,
    depth: int,
    trials: int,
    seed: int = 0,
    latency: LatencyModel | None = None,
    out_csv: str | None = None,
    strategies: Sequence[str] = ("baseline", "balanced", "tpccap"),
) -> list[BenchmarkRow]:
    """Run a benchmark over multiple random circuits and write results to CSV.

    Parameters
    ----------
    strategies:
        Any subset of:
        - "baseline": identity-ish initial layout + SABRE routing
        - "balanced": QuPort baseline partitioner
        - "cluster": heavy-edge clustering partitioner
        - "tpccap": QuPort novel partitioner (topology+port+congestion aware)
        - "tpccap_sa": TPCCAP plus simulated-annealing refinement
        - "ebit": TPCCAP-SA driven by e-bit demand after aggregation

    Notes
    -----
    The CSV is deliberately numeric-friendly. The column `method` encodes:
        baseline=0, balanced=1, tpccap=2, tpccap_sa=3, cluster=4, ebit=5
    and the column `strategy` stores the string name for readability.
    """
    latency = latency or LatencyModel()
    rows: list[BenchmarkRow] = []
    if out_csv:
        out_csv = _validate_out_csv(out_csv)

    n_logical_value = validate_nonnegative_integral(n_logical, label="n_logical")
    depth_value = validate_nonnegative_integral(depth, label="depth")
    trials_value = validate_nonnegative_integral(trials, label="trials")
    seed_value = validate_nonnegative_integral(seed, label="seed")

    selected_strategies = _validate_strategy_sequence(
        strategies, label="benchmark strategies"
    )

    fieldnames = [
        "trial",
        "seed",
        "method",
        "strategy",
        "swaps",
        "remote_2q",
        "depth",
        "size",
        "cost_total",
        "cost_local",
        "cost_remote",
        "mapping_time_s",
        "transpile_time_s",
    ]

    for t in range(trials_value):
        s = seed_value + t
        qc = random_benchmark_circuit(n_logical_value, depth_value, s)

        results: list[tuple[str, MapResult]] = []
        for strategy_name in selected_strategies:
            if strategy_name == "baseline":
                result = transpile_baseline(qc, cfg, latency=latency, seed=s)
            else:
                result = map_and_transpile(
                    qc, cfg, latency=latency, seed=s, strategy=strategy_name
                )
            results.append((strategy_name, result))

        for strat, r in results:
            rows.append(
                {
                    "trial": float(t),
                    "seed": float(s),
                    "method": _BENCHMARK_METHOD_IDS[strat],
                    "strategy": strat,
                    "swaps": float(r.metrics.swaps),
                    "remote_2q": float(r.metrics.remote_2q),
                    "depth": float(r.metrics.depth),
                    "size": float(r.metrics.size),
                    "cost_total": float(r.cost.total),
                    "cost_local": float(r.cost.local),
                    "cost_remote": float(r.cost.remote),
                    "mapping_time_s": float(r.mapping_time_s),
                    "transpile_time_s": float(r.transpile_time_s),
                }
            )

    if out_csv:
        with _open_csv_for_write(out_csv) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def sweep_topologies(
    n_logical: int,
    depth: int,
    trials: int,
    seed: int,
    out_csv: str,
    intra_topologies: Sequence[IntraTopology] = ("clique", "line", "ring"),
    inter_topologies: Sequence[InterTopology] = ("switch", "ring", "degree_d", "clos"),
    comm_ports: Sequence[int] = (1, 2),
    compute_per_qpu: int = 8,
    n_qpus: int = 10,
    inter_degree: int = 2,
    strategies: Sequence[str] = ("baseline", "balanced", "tpccap"),
) -> None:
    """Sweep multiple topology settings; write summary CSV."""
    out_csv = _validate_out_csv(out_csv)
    n_logical_value = validate_nonnegative_integral(n_logical, label="n_logical")
    depth_value = validate_nonnegative_integral(depth, label="depth")
    trials_value = validate_nonnegative_integral(trials, label="trials")
    seed_value = validate_nonnegative_integral(seed, label="seed")
    compute_per_qpu_value = validate_nonnegative_integral(
        compute_per_qpu, label="compute_per_qpu"
    )
    n_qpus_value = _validate_positive_int(n_qpus, label="n_qpus")
    inter_degree_value = validate_nonnegative_integral(
        inter_degree, label="inter_degree"
    )
    intra_values = _validate_string_sequence(intra_topologies, label="intra_topologies")
    inter_values = _validate_string_sequence(inter_topologies, label="inter_topologies")
    comm_port_values = _validate_nonnegative_int_sequence(
        comm_ports, label="comm_ports"
    )
    selected_strategies = _validate_strategy_sequence(
        strategies, label="sweep strategies"
    )

    latency = LatencyModel()
    summary: list[SweepSummaryRow] = []

    for intra in intra_values:
        for inter in inter_values:
            for ports in comm_port_values:
                cfg = MultiQPUConfig(
                    n_qpus=n_qpus_value,
                    compute_qubits_per_qpu=compute_per_qpu_value,
                    comm_qubits_per_qpu=ports,
                    intra_topology=intra,
                    inter_topology=inter,
                    inter_degree=inter_degree_value,
                )
                if n_logical_value > cfg.total_physical_qubits():
                    continue

                try:
                    rows = benchmark_random_circuits(
                        cfg=cfg,
                        n_logical=n_logical_value,
                        depth=depth_value,
                        trials=trials_value,
                        seed=seed_value,
                        latency=latency,
                        out_csv=None,
                        strategies=selected_strategies,
                    )
                except ValueError as exc:
                    # Most often a machine global routing cannot connect --
                    # with no comm ports, say -- which only a circuit that
                    # needs a path between its QPUs runs into.
                    raise ValueError(
                        f"sweep setting (intra={intra}, inter={inter}, "
                        f"ports={ports}): {exc}"
                    ) from exc

                rows_snapshot = list(rows)
                intra_id = intra
                inter_id = inter
                ports_count = ports

                def agg(
                    strategy_name: str,
                    *,
                    rows_snapshot: list[BenchmarkRow] = rows_snapshot,
                    intra_id: str = intra_id,
                    inter_id: str = inter_id,
                    ports_count: int = ports_count,
                ) -> SweepSummaryRow:
                    method = _BENCHMARK_METHOD_IDS[strategy_name]
                    rs = [r for r in rows_snapshot if r["strategy"] == strategy_name]

                    def mean(k: str) -> float:
                        return sum(float(r[k]) for r in rs) / max(1, len(rs))

                    def median(k: str) -> float:
                        """Median alongside the mean, because cost is heavily skewed.

                        A handful of hard random circuits can move the mean far
                        enough to reverse which strategy looks better while the
                        median points the other way; README.md quotes a measured
                        case. Reporting only the mean hides that.
                        """
                        values = sorted(float(r[k]) for r in rs)
                        if not values:
                            return 0.0
                        mid = len(values) // 2
                        if len(values) % 2:
                            return values[mid]
                        return (values[mid - 1] + values[mid]) / 2.0

                    return {
                        "intra": intra_id,
                        "inter": inter_id,
                        "ports": float(ports_count),
                        "method": method,
                        "swaps_mean": mean("swaps"),
                        "remote_2q_mean": mean("remote_2q"),
                        "depth_mean": mean("depth"),
                        "cost_mean": mean("cost_total"),
                        "cost_median": median("cost_total"),
                        "transpile_time_mean": mean("transpile_time_s"),
                    }

                # With no trials there is nothing to summarise: a row of zeros
                # would read as a measured zero cost. benchmark_random_circuits
                # writes only its header in the same case.
                if trials_value == 0:
                    continue
                for strategy_name in selected_strategies:
                    summary.append(agg(strategy_name))

    sweep_fieldnames = [
        "intra",
        "inter",
        "ports",
        "method",
        "swaps_mean",
        "remote_2q_mean",
        "depth_mean",
        "cost_mean",
        "cost_median",
        "transpile_time_mean",
    ]
    with _open_csv_for_write(out_csv) as f:
        writer = csv.DictWriter(f, fieldnames=sweep_fieldnames)
        writer.writeheader()
        writer.writerows(summary)
