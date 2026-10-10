"""Reproduce every measured figure quoted in README.md and docs/entanglement.md.

Each section prints the numbers one documented claim rests on, from a setup
spelled out in full below -- instance sizes, depth, seeds, comm ports,
interconnect, and how results are aggregated -- so a figure in the docs is a
checked claim rather than a remembered one. Run every section, or name some:

    python examples/reproduce_readme_figures.py
    python examples/reproduce_readme_figures.py calibration temporal

Three sections (``measured_effect``, ``rescaling`` and ``calibration``) also run
the ``"ebit"`` strategy under other objectives -- the one it used before its
penalties were rescaled, or e-bits alone. They patch
``quport.compiler.ebit_partition`` for the duration of that one comparison, so
the rest of the pipeline -- layout, splitting, aggregation, scheduling -- is
exactly what ships.

A few private helpers are imported on purpose: ``annealing`` has to score a
partition with exactly the objective the TPCCAP seed was built for, and
translate circuits exactly as the pipelines do.

Everything is deterministic for the seeds below. Expect a few minutes for a
full run; ``annealing`` is the slowest section.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
import statistics
import sys
from collections.abc import Callable, Iterator, Sequence
from unittest import mock

from qiskit import QuantumCircuit

import quport.compiler
from quport import (
    LatencyModel,
    MultiQPUArchitecture,
    MultiQPUConfig,
    compile_distributed,
)
from quport.aggregation import aggregate_remote_operations
from quport.exact import optimal_partition
from quport.hypergraph import build_distributable_packets, ebit_cost
from quport.interaction import (
    cut_weight,
    extract_temporal_twoq_weights,
    extract_twoq_weights,
)
from quport.network import compute_boundary_counts
from quport.partition import (
    _objective_tpccap,
    _validate_and_normalize_partition_inputs,
    tpccap_partition,
    tpccap_sa_partition,
)
from quport.pipeline import (
    _translate_to_basis,
    benchmark_random_circuits,
    random_benchmark_circuit,
)
from quport.schedule import (
    audit_entanglement_schedule,
    estimate_entanglement_schedule,
)
from quport.temporal import (
    TemporalPartition,
    optimize_temporal_partition,
    split_windows,
    temporal_ebit_cost,
)

SEEDS = range(6)


def _config(
    n_logical: int, n_qpus: int, comm: int, topology: str = "switch"
) -> MultiQPUConfig:
    """A machine whose capacity is exactly ``n_logical / n_qpus`` per QPU."""
    capacity = n_logical // n_qpus
    assert capacity * n_qpus == n_logical, "instances use exactly-full machines"
    return MultiQPUConfig(
        n_qpus=n_qpus,
        compute_qubits_per_qpu=capacity - comm,
        comm_qubits_per_qpu=comm,
        inter_topology=topology,  # type: ignore[arg-type]
    )


def _pre_rescaling_ebit(
    n: int,
    weights: object,
    n_qpus: int,
    capacity: int,
    comm_ports_per_qpu: int,
    sp: object,
    packets: object,
    seed: int | None = None,
) -> object:
    """The ``"ebit"`` objective before rescaling: only the volume term swapped."""
    return tpccap_sa_partition(
        n=n,
        weights=weights,  # type: ignore[arg-type]
        n_qpus=n_qpus,
        capacity=capacity,
        comm_ports_per_qpu=comm_ports_per_qpu,
        sp=sp,  # type: ignore[arg-type]
        seed=seed,
        w_dist=0.0,
        packets=packets,  # type: ignore[arg-type]
        w_ebit=1.0,
    )


def _ebit_objective_alone(
    n: int,
    weights: object,
    n_qpus: int,
    capacity: int,
    comm_ports_per_qpu: int,
    sp: object,
    packets: object,
    seed: int | None = None,
) -> object:
    """The same search priced by hop-scaled e-bits and nothing else."""
    return tpccap_sa_partition(
        n=n,
        weights=weights,  # type: ignore[arg-type]
        n_qpus=n_qpus,
        capacity=capacity,
        comm_ports_per_qpu=comm_ports_per_qpu,
        sp=sp,  # type: ignore[arg-type]
        seed=seed,
        w_dist=0.0,
        w_port=0.0,
        w_cong=0.0,
        anneal_w_cong=None,
        packets=packets,  # type: ignore[arg-type]
        w_ebit=1.0,
    )


@contextlib.contextmanager
def _ebit_strategy(partitioner: Callable[..., object]) -> Iterator[None]:
    """Run ``strategy="ebit"`` with a different partitioner, and only here."""
    with mock.patch.object(quport.compiler, "ebit_partition", partitioner):
        yield


def _qft(n_qubits: int) -> QuantumCircuit:
    """The controlled-phase ladder without the terminating swaps."""
    circuit = QuantumCircuit(n_qubits)
    for control in range(n_qubits):
        circuit.h(control)
        for target in range(control + 1, n_qubits):
            circuit.cp(math.pi / 2 ** (target - control), target, control)
    return circuit


def _ghz(n_qubits: int) -> QuantumCircuit:
    """One ``h`` followed by a ``cx`` fan-out from qubit 0."""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for target in range(1, n_qubits):
        circuit.cx(0, target)
    return circuit


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# README "Measured effect", and the pre-rescaling counts quoted under it
# ---------------------------------------------------------------------------


def measured_effect() -> None:
    """4 QPUs x (4 compute + 2 comm), switch, optimization level 0."""
    cfg = MultiQPUConfig(
        n_qpus=4,
        compute_qubits_per_qpu=4,
        comm_qubits_per_qpu=2,
        inter_topology="switch",
        optimization_level=0,
    )
    rows = [
        ("16-qubit QFT, ebit, seed 0", [(_qft(16), 0)], "ebit"),
        ("16-qubit GHZ fan-out, ebit, seed 0", [(_ghz(16), 0)], "ebit"),
        (
            "16-qubit random depth-20, tpccap_sa, seeds 0..9",
            [(random_benchmark_circuit(16, 20, seed), seed) for seed in range(10)],
            "tpccap_sa",
        ),
    ]
    print("| Circuit | Cross-QPU gates | EPR pairs, per gate | aggregated | Saved |")
    for label, instances, strategy in rows:
        plans = [
            compile_distributed(qc, cfg, seed=seed, strategy=strategy).aggregation
            for qc, seed in instances
        ]
        remote = sum(plan.remote_gates for plan in plans)
        baseline = sum(plan.baseline_epr_pairs for plan in plans)
        pairs = sum(plan.epr_pairs for plan in plans)
        print(
            f"| {label} | {remote} | {baseline} | {pairs} | {1 - pairs / baseline:.1%} |"
        )

    print("\nThe same two circuits under the pre-rescaling ebit objective:")
    with _ebit_strategy(_pre_rescaling_ebit):
        for label, qc in (("QFT", _qft(16)), ("GHZ", _ghz(16))):
            plan = compile_distributed(qc, cfg, seed=0, strategy="ebit").aggregation
            print(
                f"  {label}: {plan.remote_gates} cross-QPU gates, {plan.epr_pairs} EPR pairs"
            )


# ---------------------------------------------------------------------------
# README "Rescaling the rest of the objective"
# ---------------------------------------------------------------------------

RESCALING_SIZES = ((9, 3), (12, 3), (16, 4), (20, 5))
RESCALING_TOPOLOGIES = ("ring", "switch", "mesh")
RESCALING_DEPTH = 20
RESCALING_COMM = 2


def rescaling() -> None:
    """Pre-rescaling vs. current ebit objective, on the compiled plan.

    Sizes x topologies x seeds, depth 20, two comm ports per QPU, capacity
    exactly n/q. Reported as means over every instance.
    """
    metrics: dict[str, list[list[float]]] = {"before": [], "after": []}
    for (n, n_qpus), topology, seed in (
        (size, topology, seed)
        for size in RESCALING_SIZES
        for topology in RESCALING_TOPOLOGIES
        for seed in SEEDS
    ):
        cfg = _config(n, n_qpus, RESCALING_COMM, topology)
        qc = random_benchmark_circuit(n, RESCALING_DEPTH, seed)
        with _ebit_strategy(_pre_rescaling_ebit):
            before = compile_distributed(qc, cfg, seed=seed, strategy="ebit")
        after = compile_distributed(qc, cfg, seed=seed, strategy="ebit")
        for key, res in (("before", before), ("after", after)):
            schedule = res.entanglement_schedule
            metrics[key].append(
                [
                    res.aggregation.epr_pairs,
                    res.aggregation.evictions,
                    schedule.makespan,
                    max((busy for _edge, busy in schedule.link_busy_time), default=0.0),
                ]
            )
    count = len(metrics["after"])
    print(f"Means over {count} instances (before -> after):")
    for column, label in enumerate(
        ("EPR pairs spent", "port evictions", "entanglement makespan", "peak link busy")
    ):
        first = _mean([row[column] for row in metrics["before"]])
        second = _mean([row[column] for row in metrics["after"]])
        print(f"  {label}: {first:.2f} -> {second:.2f}")


# ---------------------------------------------------------------------------
# README "Calibrating the heuristics against the exact optimum"
# ---------------------------------------------------------------------------

CALIBRATION_FAMILIES = ((8, 2), (9, 3), (12, 3), (12, 4))
CALIBRATION_DEPTH = 10


def calibration() -> None:
    """Mean relative gap to the proved optimum, under both objectives.

    Families x seeds, depth 10, one comm port per QPU, switch (all-to-all),
    capacity exactly n/q. The cut objective scores uniform interaction counts
    of the basis-translated circuit; every optimum is proved and non-zero.
    """
    strategies = ("tpccap", "cluster", "tpccap_sa", "balanced", "ebit")
    variants = {
        "ebit, pre-rescaling": _pre_rescaling_ebit,
        "e-bit objective alone": _ebit_objective_alone,
    }
    gaps: dict[str, dict[str, list[float]]] = {
        name: {"ebits": [], "cut": []} for name in (*strategies, *variants)
    }
    for n, n_qpus in CALIBRATION_FAMILIES:
        cfg = _config(n, n_qpus, comm=1)
        capacity = cfg.capacity_per_qpu()
        for seed in SEEDS:
            qc = random_benchmark_circuit(n, CALIBRATION_DEPTH, seed)
            results = {
                strategy: compile_distributed(qc, cfg, seed=seed, strategy=strategy)
                for strategy in strategies
            }
            partitions = {name: res.partition for name, res in results.items()}
            for name, partitioner in variants.items():
                with _ebit_strategy(partitioner):
                    partitions[name] = compile_distributed(
                        qc, cfg, seed=seed, strategy="ebit"
                    ).partition
            # Every strategy translates the circuit identically, so any result's
            # packets and basis circuit describe the instance they all solved.
            packets = results["ebit"].packets
            weights = extract_twoq_weights(results["ebit"].basis_circuit)
            optimum = {
                "ebits": optimal_partition(
                    n, n_qpus, capacity, objective="ebits", packets=packets
                ),
                "cut": optimal_partition(
                    n, n_qpus, capacity, objective="cut", weights=weights
                ),
            }
            for exact in optimum.values():
                assert exact.proved_optimal and exact.objective > 0
            for name, part in partitions.items():
                scores = {
                    "ebits": float(ebit_cost(packets, part, n_qpus)),
                    "cut": cut_weight(weights, part),
                }
                for objective, score in scores.items():
                    best = optimum[objective].objective
                    gaps[name][objective].append((score - best) / best)
    count = len(gaps["ebit"]["ebits"])
    print(f"Mean gap over {count} instances:")
    print("| Strategy | gap vs. optimal e-bits | gap vs. optimal cut |")
    for name, by_objective in gaps.items():
        print(
            f"| {name} | {_mean(by_objective['ebits']):.1%} "
            f"| {_mean(by_objective['cut']):.1%} |"
        )


# ---------------------------------------------------------------------------
# README "Letting qubits move between QPUs"
# ---------------------------------------------------------------------------

TEMPORAL_SIZES = ((9, 3), (12, 3), (16, 4), (20, 5))
TEMPORAL_DEPTH = 20
TEMPORAL_WINDOWS = (2, 3, 4)


def temporal() -> None:
    """Static re-placement and migration, against the ebit strategy's partition.

    Sizes x seeds, depth 20, one comm port per QPU, switch, capacity exactly
    n/q, 2 to 4 windows. "Better static placement" is the mean seed-to-
    stationary saving; the migration columns give the range, over window
    counts, of the per-window-count mean.
    """
    print(
        "| Instance | Better static placement | Migration, on top | Migrations used |"
    )
    for n, n_qpus in TEMPORAL_SIZES:
        cfg = _config(n, n_qpus, comm=1)
        capacity = cfg.capacity_per_qpu()
        static_gain: list[float] = []
        migration = {count: [] for count in TEMPORAL_WINDOWS}
        moves = {count: [] for count in TEMPORAL_WINDOWS}
        for seed in SEEDS:
            res = compile_distributed(
                random_benchmark_circuit(n, TEMPORAL_DEPTH, seed),
                cfg,
                seed=seed,
                strategy="ebit",
            )
            for count in TEMPORAL_WINDOWS:
                windows = split_windows(res.packets, count)
                result = optimize_temporal_partition(
                    res.packets, res.partition, n_qpus, capacity, windows, seed=seed
                )
                if count == TEMPORAL_WINDOWS[0]:
                    static_gain.append(
                        (result.static_cost - result.stationary_cost)
                        / result.static_cost
                    )
                migration[count].append(result.migration_reduction)
                moves[count].append(result.cost.moves)
        mig = [_mean(values) for values in migration.values()]
        mv = [_mean(values) for values in moves.values()]
        print(
            f"| {n}q / {n_qpus} QPUs | {_mean(static_gain):.1%} "
            f"| {min(mig):.1%} - {max(mig):.1%} | {min(mv):.1f} - {max(mv):.1f} |"
        )


NEIGHBOURHOOD_SIZE = (16, 4)
NEIGHBOURHOOD_WINDOWS = 4


def _single_window_descent(
    packets: object,
    start: list[int],
    windows: Sequence[object],
    n_qpus: int,
    capacity: int,
    seed: int,
    max_passes: int = 8,
) -> int:
    """First-improvement descent whose moves each change one window only.

    The interval search's own neighbourhood restricted to intervals of length
    one: a single-window move of one qubit, or a single-window swap of two.
    Seeded, like the library's temporal phase, from the stationary optimum.
    """
    rng = random.Random(seed)
    assignments = [list(start) for _ in windows]

    def cost() -> int:
        partition = TemporalPartition(
            windows=tuple(windows),  # type: ignore[arg-type]
            assignments=tuple(tuple(a) for a in assignments),
        )
        return temporal_ebit_cost(packets, partition, n_qpus).total  # type: ignore[arg-type]

    best = cost()
    n_qubits = len(start)
    order = [(w, q) for w in range(len(windows)) for q in range(n_qubits)]
    for _ in range(max_passes):
        improved = False
        rng.shuffle(order)
        for window, qubit in order:
            row = assignments[window]
            origin = row[qubit]
            # Like the interval search: take the first improving move, then
            # the first improving swap, for this qubit and window.
            for target in range(n_qpus):
                if target == origin or row.count(target) >= capacity:
                    continue
                row[qubit] = target
                candidate = cost()
                if candidate < best:
                    best, improved = candidate, True
                    break
                row[qubit] = origin
            for other in range(n_qubits):
                if row[other] == row[qubit]:
                    continue
                row[qubit], row[other] = row[other], row[qubit]
                candidate = cost()
                if candidate < best:
                    best, improved = candidate, True
                    break
                row[qubit], row[other] = row[other], row[qubit]
        if not improved:
            break
    return best


def neighbourhood() -> None:
    """Saving over the seed: single-window moves vs. interval moves.

    16 qubits on 4 QPUs, depth 20, seeds, 4 windows, one comm port, switch.
    Both searches start from the same stationary optimum, so the difference
    is the neighbourhood alone.
    """
    n, n_qpus = NEIGHBOURHOOD_SIZE
    cfg = _config(n, n_qpus, comm=1)
    capacity = cfg.capacity_per_qpu()
    single: list[float] = []
    interval: list[float] = []
    for seed in SEEDS:
        res = compile_distributed(
            random_benchmark_circuit(n, TEMPORAL_DEPTH, seed),
            cfg,
            seed=seed,
            strategy="ebit",
        )
        windows = split_windows(res.packets, NEIGHBOURHOOD_WINDOWS)
        result = optimize_temporal_partition(
            res.packets, res.partition, n_qpus, capacity, windows, seed=seed
        )
        stationary = optimize_temporal_partition(
            res.packets,
            res.partition,
            n_qpus,
            capacity,
            split_windows(res.packets, 1),
            seed=seed,
        )
        start = list(stationary.partition.assignments[0])
        single_cost = _single_window_descent(
            res.packets, start, windows, n_qpus, capacity, seed
        )
        single.append((result.static_cost - single_cost) / result.static_cost)
        interval.append(result.reduction)
    print(
        f"{n}-qubit instances: single-window moves save {_mean(single):.1%}, "
        f"interval moves save {_mean(interval):.1%}"
    )


# ---------------------------------------------------------------------------
# README "Output artifacts": entanglement schedule properties
# ---------------------------------------------------------------------------

SCHEDULE_TOPOLOGIES = ("ring", "switch", "mesh", "degree_d")
SCHEDULE_INSTANCES = 28


def schedules() -> None:
    """Audit and monotonicity over one fixed plan per instance.

    9 qubits on 3 QPUs (3 compute + 1 comm), depth 10, tpccap_sa, for each
    topology and 28 seeds. Each instance is scheduled six ways -- as built,
    with ports +1 and x4, with one more link channel, with epr_gen doubled and
    with the success probability halved -- and every schedule is audited.
    """
    checked = 0
    violations = 0
    for topology in SCHEDULE_TOPOLOGIES:
        cfg = MultiQPUConfig(
            n_qpus=3,
            compute_qubits_per_qpu=3,
            comm_qubits_per_qpu=1,
            inter_topology=topology,  # type: ignore[arg-type]
            optimization_level=0,
        )
        for seed in range(SCHEDULE_INSTANCES):
            res = compile_distributed(
                random_benchmark_circuit(9, 10, seed),
                cfg,
                seed=seed,
                strategy="tpccap_sa",
            )
            mapped = res.physical_circuit
            arch = MultiQPUArchitecture(cfg)
            plan = aggregate_remote_operations(mapped, arch, ports_per_qpu=1)
            model = LatencyModel(epr_success_prob=0.5)
            wider_links = MultiQPUArchitecture(
                dataclasses.replace(cfg, link_capacity=cfg.link_capacity + 1)
            )
            cases = {
                "base": (arch, model, 1),
                "ports+1": (arch, model, 2),
                "ports*4": (arch, model, 4),
                "links+1": (wider_links, model, 1),
                "slower": (
                    arch,
                    dataclasses.replace(model, epr_gen=model.epr_gen * 2),
                    1,
                ),
                "lossier": (arch, dataclasses.replace(model, epr_success_prob=0.25), 1),
            }
            makespans = {}
            for name, (machine, latency, ports) in cases.items():
                summary = estimate_entanglement_schedule(
                    mapped, machine, latency, plan=plan, ports_per_qpu=ports
                )
                problems = audit_entanglement_schedule(
                    summary, mapped, machine, latency, plan=plan, ports_per_qpu=ports
                )
                violations += bool(problems)
                makespans[name] = summary.makespan
                checked += 1
            base = makespans["base"]
            tolerance = 1e-9 * max(1.0, base)
            violations += any(
                makespans[name] > base + tolerance
                for name in ("ports+1", "ports*4", "links+1")
            )
            violations += any(
                makespans[name] < base - tolerance for name in ("slower", "lossier")
            )
    print(
        f"{checked} schedules over {len(SCHEDULE_TOPOLOGIES)} topologies, "
        f"{violations} property violations"
    )


# ---------------------------------------------------------------------------
# README "tpccap_sa": how often annealing loses ground under the seed's weighting
# ---------------------------------------------------------------------------


def _loses_to_its_seed(n: int, weights: object, cfg: MultiQPUConfig, seed: int) -> bool:
    sp = MultiQPUArchitecture(cfg).qpu_shortest_paths()
    capacity = cfg.capacity_per_qpu()
    comm = cfg.comm_qubits_per_qpu
    args = (n, weights, cfg.n_qpus, capacity, comm, sp)
    start = tpccap_partition(*args, seed=seed)[0].part  # type: ignore[arg-type]
    annealed = tpccap_sa_partition(*args, seed=seed)[0].part  # type: ignore[arg-type]
    normalized = _validate_and_normalize_partition_inputs(
        n, weights, cfg.n_qpus, capacity  # type: ignore[arg-type]
    )

    def score(part: list[int]) -> float:
        return _objective_tpccap(
            normalized, part, cfg.n_qpus, comm, sp, 1.0, 5.0, 0.05, "ecmp"
        )[0]

    return score(annealed) > score(start) + 1e-9


def annealing() -> None:
    """The annealed partition scored with its seed's weighting (w_cong=0.05).

    Two instance sets. 240 random draws (random.Random(0)) over 2, 3, 4 or 6
    QPUs, 2-4 compute and 1-2 comm qubits each, every interconnect, half-full
    to full, depth 3, 8 or 20, decayed weights. And the default 10-QPU
    configuration with 80 logical qubits at depth 20, for switch, ring and
    degree_d, 1 and 2 ports, seeds, decayed and uniform weights.
    """
    rng = random.Random(0)
    lost = 0
    for trial in range(240):
        n_qpus = rng.choice([2, 3, 4, 6])
        compute = rng.choice([2, 3, 4])
        comm = rng.choice([1, 2])
        topology = rng.choice(["switch", "ring", "degree_d", "fat_tree", "clos"])
        cfg = MultiQPUConfig(
            n_qpus=n_qpus,
            compute_qubits_per_qpu=compute,
            comm_qubits_per_qpu=comm,
            inter_topology=topology,  # type: ignore[arg-type]
        )
        capacity = cfg.capacity_per_qpu()
        n = rng.randint(max(2, n_qpus * capacity // 2), n_qpus * capacity)
        depth = rng.choice([3, 8, 20])
        qc = _translate_to_basis(
            random_benchmark_circuit(n, depth, trial), cfg.basis_gates, trial
        )
        lost += _loses_to_its_seed(
            n, extract_temporal_twoq_weights(qc, 0.98), cfg, trial
        )
    print(f"random draws: annealing lost ground in {lost} of 240")

    lost = total = 0
    for topology in ("switch", "ring", "degree_d"):
        for comm in (1, 2):
            cfg = MultiQPUConfig(inter_topology=topology, comm_qubits_per_qpu=comm)  # type: ignore[arg-type]
            for seed in SEEDS:
                qc = _translate_to_basis(
                    random_benchmark_circuit(80, 20, seed), cfg.basis_gates, seed
                )
                for weights in (
                    extract_temporal_twoq_weights(qc, 0.98),
                    extract_twoq_weights(qc),
                ):
                    lost += _loses_to_its_seed(80, weights, cfg, seed)
                    total += 1
    print(
        f"default configuration, 80 qubits: annealing lost ground in {lost} of {total}"
    )


# ---------------------------------------------------------------------------
# README "Sweep CSV": per-instance cost skew between strategies
# ---------------------------------------------------------------------------


def sweep_skew() -> None:
    """Per-instance cost of one strategy relative to another.

    6 QPUs x (4 compute + 1 comm), 24 logical qubits, depth 8, 20 trials,
    for clique and ring local topologies on switch and ring interconnects --
    the setup of examples/topology_sweep.py with more trials. Also counts the
    settings where the mean and the median rank a pair of strategies the
    opposite way round, which is what the sweep's cost_median column is for.
    """
    strategies = ("baseline", "balanced", "tpccap")
    settings: list[dict[str, list[float]]] = []
    for intra in ("clique", "ring"):
        for inter in ("switch", "ring"):
            cfg = MultiQPUConfig(
                n_qpus=6,
                compute_qubits_per_qpu=4,
                comm_qubits_per_qpu=1,
                intra_topology=intra,  # type: ignore[arg-type]
                inter_topology=inter,  # type: ignore[arg-type]
            )
            rows = benchmark_random_circuits(
                cfg, 24, 8, 20, seed=7, strategies=strategies
            )
            settings.append(
                {
                    name: [
                        float(row["cost_total"])
                        for row in rows
                        if row["strategy"] == name
                    ]
                    for name in strategies
                }
            )
    for first, second in (
        ("tpccap", "balanced"),
        ("tpccap", "baseline"),
        ("balanced", "baseline"),
    ):
        ratios = [
            a / b - 1.0
            for costs in settings
            for a, b in zip(costs[first], costs[second])
        ]
        disagree = sum(
            (_mean(costs[first]) < _mean(costs[second]))
            != (statistics.median(costs[first]) < statistics.median(costs[second]))
            for costs in settings
        )
        print(
            f"{first} vs {second}: per-instance cost ratio {min(ratios):+.0%} to "
            f"{max(ratios):+.0%} over {len(ratios)} instances; mean and median "
            f"disagree on the ranking in {disagree} of {len(settings)} settings"
        )


# ---------------------------------------------------------------------------
# README "Rescaling the rest of the objective": how large the old terms are
# ---------------------------------------------------------------------------


def penalty_scale() -> None:
    """The port penalty and the cut term, each against an e-bit count.

    Two instance sets: the ``rescaling`` set (two comm ports per QPU, depth 20)
    and the ``calibration`` set (one port, depth 10). For each instance, the
    weighted port penalty ``w_port * sum(overflow**2)`` with ``w_port=5`` is
    divided by the e-bit count at 20 uniformly random feasible partitions
    (``random.Random(seed)``) -- the landscape the search moves through -- and
    at the solution the pre-rescaling ebit objective returns. The aggregation
    factor is cut gates over e-bits at the same random partitions.
    """
    instance_sets = {
        "two ports (rescaling set)": [
            (n, n_qpus, topology, 2, RESCALING_DEPTH)
            for n, n_qpus in RESCALING_SIZES
            for topology in RESCALING_TOPOLOGIES
        ],
        "one port (calibration set)": [
            (n, n_qpus, "switch", 1, CALIBRATION_DEPTH)
            for n, n_qpus in CALIBRATION_FAMILIES
        ],
    }
    for label, instances in instance_sets.items():
        landscape: list[float] = []
        solutions: list[float] = []
        aggregation: list[float] = []
        for n, n_qpus, topology, comm, depth in instances:
            cfg = _config(n, n_qpus, comm, topology)
            capacity = cfg.capacity_per_qpu()
            sp = MultiQPUArchitecture(cfg).qpu_shortest_paths()
            for seed in SEEDS:
                qc = _translate_to_basis(
                    random_benchmark_circuit(n, depth, seed), cfg.basis_gates, seed
                )
                weights = extract_temporal_twoq_weights(qc, 0.98)
                uniform = extract_twoq_weights(qc)
                packets = build_distributable_packets(qc)

                def penalty(part: list[int]) -> float:
                    boundary = compute_boundary_counts(weights, part, n_qpus)
                    return 5.0 * sum(max(0, b - comm) ** 2 for b in boundary)

                rng = random.Random(seed)
                for _ in range(20):
                    slots = [q for q in range(n_qpus) for _ in range(capacity)]
                    rng.shuffle(slots)
                    part = slots[:n]
                    ebits = ebit_cost(packets, part, n_qpus)
                    if ebits:
                        landscape.append(penalty(part) / ebits)
                        aggregation.append(cut_weight(uniform, part) / ebits)
                solved = _pre_rescaling_ebit(
                    n, weights, n_qpus, capacity, comm, sp, packets, seed=seed
                )
                part = solved[0].part  # type: ignore[index]
                ebits = ebit_cost(packets, part, n_qpus)
                if ebits:
                    solutions.append(penalty(part) / ebits)

        def summary(values: list[float]) -> str:
            return (
                f"median {statistics.median(values):.1f}x, "
                f"range {min(values):.1f}x to {max(values):.1f}x"
            )

        print(label)
        print(f"  port penalty / e-bits, random partitions: {summary(landscape)}")
        print(f"  port penalty / e-bits, pre-rescaling solution: {summary(solutions)}")
        print(f"  cut gates / e-bits, random partitions: {summary(aggregation)}")


SECTIONS: dict[str, Callable[[], None]] = {
    "measured_effect": measured_effect,
    "rescaling": rescaling,
    "calibration": calibration,
    "temporal": temporal,
    "neighbourhood": neighbourhood,
    "schedules": schedules,
    "annealing": annealing,
    "sweep_skew": sweep_skew,
    "penalty_scale": penalty_scale,
}


def main(names: Sequence[str]) -> None:
    unknown = [name for name in names if name not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    for name in names or SECTIONS:
        print(f"== {name}")
        SECTIONS[name]()
        print(flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
