# Examples

The snippets below are intentionally small, but they are complete enough to paste
into a Python file or notebook. For reproducible comparisons, keep seeds and config
fields fixed except for the variable you are studying.

## Build an architecture and inspect blocks

```python
from quport import MultiQPUArchitecture, MultiQPUConfig

cfg = MultiQPUConfig(n_qpus=3, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
arch = MultiQPUArchitecture(cfg)

for qpu_id, block in enumerate(arch.all_blocks()):
    print(qpu_id, block.compute, block.comm)
```

Expected interpretation: each QPU has two compute qubits and one communication
qubit, and the physical indices are contiguous by QPU.

## Inspect a coupling map

```python
from quport import MultiQPUArchitecture, MultiQPUConfig

cfg = MultiQPUConfig(
    n_qpus=2,
    compute_qubits_per_qpu=2,
    comm_qubits_per_qpu=1,
    intra_topology="line",
    inter_topology="ring",
)
arch = MultiQPUArchitecture(cfg)
print(arch.build_coupling_map().get_edges())
```

This is helpful when validating whether a custom architecture setting creates the
physical links you expect.

## Global mapping

```python
from quport import LatencyModel, MultiQPUConfig, map_and_transpile
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1)
qc = random_benchmark_circuit(4, depth=4, seed=7)
res = map_and_transpile(qc, cfg, LatencyModel(), seed=7, strategy="tpccap")

print(res.partition)
print(res.metrics.remote_2q)
print(res.cost.total)
```

Use this workflow when you want Qiskit's final globally routed circuit and standard
circuit metrics.

## Distributed compilation bundle

```python
from quport import LatencyModel, MultiQPUConfig, compile_distributed, write_distributed_program
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1)
qc = random_benchmark_circuit(4, depth=4, seed=7)
res = compile_distributed(qc, cfg, LatencyModel(), seed=7, strategy="tpccap_sa")

write_distributed_program(res.program, "distributed_bundle")
print(res.schedule)
```

The written bundle contains local QASM programs and a remote-operation manifest.
If you need locally routed QPU programs exactly as produced by `compile_distributed`,
write `res.local_routed` yourself or use the CLI `compile-dist` command.

## EPR-pair accounting and communication aggregation

```python
from quport import (
    LatencyModel,
    MultiQPUArchitecture,
    MultiQPUConfig,
    aggregate_remote_operations,
    compile_distributed,
    ebit_cost,
    estimate_entanglement_schedule,
)
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(n_qpus=4, compute_qubits_per_qpu=4, comm_qubits_per_qpu=2)
qc = random_benchmark_circuit(16, depth=20, seed=0)
res = compile_distributed(qc, cfg, LatencyModel(), seed=0, strategy="ebit")

print(res.ebits.ebits, "e-bits with unlimited ports")
print(res.aggregation.epr_pairs, "EPR pairs under the real port budget")
print(res.aggregation.baseline_epr_pairs, "EPR pairs without aggregation")
print(f"{res.aggregation.reduction:.1%} saved")
print(res.entanglement_schedule.makespan)

for block in res.aggregation.blocks[:3]:
    print(block.protocol, block.root_phys, "->", block.remote_qpu, block.gate_indices)
```

Score an alternative partition without recompiling, then re-plan and re-schedule
against a different port budget — a plan and its schedule must share that budget:

```python
arch = MultiQPUArchitecture(cfg)
print(ebit_cost(res.packets, res.partition, cfg.n_qpus))

plan = aggregate_remote_operations(res.physical_circuit, arch, ports_per_qpu=6)
summary = estimate_entanglement_schedule(
    res.physical_circuit,
    arch,
    LatencyModel(epr_success_prob=0.5),
    plan=plan,
    ports_per_qpu=6,
)
print(summary.makespan, summary.peak_ports_in_use)
```

## What moving qubits between QPUs would save

```python
from quport import MultiQPUConfig, compile_distributed
from quport.pipeline import random_benchmark_circuit
from quport.temporal import optimize_temporal_partition, split_windows

cfg = MultiQPUConfig(
    n_qpus=4, compute_qubits_per_qpu=4, comm_qubits_per_qpu=2, optimization_level=0
)
res = compile_distributed(
    random_benchmark_circuit(16, depth=20, seed=0), cfg, seed=0, strategy="ebit"
)

windows = split_windows(res.packets, 4)
plan = optimize_temporal_partition(
    res.packets,
    res.partition,
    cfg.n_qpus,
    cfg.capacity_per_qpu(),
    windows,
    seed=0,
)

# Three costs, because two effects contribute: the seed placement, the best the
# search reaches without moving anything, and the best with migration allowed.
print(plan.static_cost, plan.stationary_cost, plan.cost.total)
print(f"{plan.migration_reduction:.1%} of that came from moving qubits")

for qubit, boundary, source, target in plan.partition.migrations():
    print(f"qubit {qubit}: QPU {source} -> {target} after window {boundary}")
```

`cost.total <= stationary_cost <= static_cost` holds by construction, so a plan
that moved qubits never loses to one that did not. Price a teleport out of reach
with `migration_cost=` and the plan collapses to pure re-placement:

```python
held = optimize_temporal_partition(
    res.packets, res.partition, cfg.n_qpus, cfg.capacity_per_qpu(), windows,
    migration_cost=10_000, seed=0,
)
assert held.cost.moves == 0
assert held.cost.total == held.stationary_cost
```

## Scoring a partition against the exact optimum

```python
from quport import MultiQPUConfig, compile_distributed
from quport.exact import optimal_partition, partition_gap
from quport.interaction import extract_twoq_weights
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(
    n_qpus=3, compute_qubits_per_qpu=3, comm_qubits_per_qpu=2, optimization_level=0
)
qc = random_benchmark_circuit(9, depth=10, seed=0)
res = compile_distributed(qc, cfg, seed=0, strategy="ebit")
capacity = cfg.capacity_per_qpu()

# `partition` and `packets` are indexed by the basis-translated circuit, so any
# re-derived partitioning input has to come from `res.basis_circuit`.
weights = extract_twoq_weights(res.basis_circuit)

best = optimal_partition(
    res.basis_circuit.num_qubits,
    cfg.n_qpus,
    capacity,
    objective="ebits",
    packets=res.packets,
)
print(best.objective, "e-bits is optimal" if best.proved_optimal else "(unproved)")
print(best.nodes, "nodes explored")

for objective, kwargs in (("cut", {"weights": weights}), ("ebits", {"packets": res.packets})):
    gap = partition_gap(res.partition, cfg.n_qpus, capacity, objective=objective, **kwargs)
    print(f"{objective}: {gap.heuristic} vs {gap.optimal} optimal ({gap.relative:.1%})")
```

`partition_gap` raises if the partition is infeasible, or if it scores below a
proved optimum — that would mean one of the two implementations is wrong, which is
worth failing over rather than reporting as a negative gap. Keep instances small:
the search enumerates set partitions.

## Detailed schedule trace

```python
from quport import MultiQPUArchitecture, MultiQPUConfig, LatencyModel, estimate_topology_schedule_plan
from quport.pipeline import random_benchmark_circuit, map_and_transpile

cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1)
qc = random_benchmark_circuit(4, depth=4, seed=2)
mapped = map_and_transpile(qc, cfg, LatencyModel(), seed=2, strategy="balanced")
plan = estimate_topology_schedule_plan(mapped.mapped_circuit, MultiQPUArchitecture(cfg), LatencyModel())

print(plan.summary)
print(plan.to_dict()["summary"])
for layer in plan.layers:
    print(layer.layer_index, layer.start_time, layer.end_time, layer.remote_ops)
    for round_trace in layer.remote_rounds:
        print("  round", round_trace.round_index, round_trace.start_time, round_trace.end_time)
```

Use the trace when a summary value such as `remote_rounds` or `makespan` changes and
you need to understand which layer/round caused it. The absolute timing fields let
you plot a timeline directly without reconstructing cumulative offsets. Use
`plan.to_dict()` for JSON export; it converts tuple-heavy resource fields into
JSON-native values and validates malformed timing/count/pair data before returning.

## Benchmark CSV

```python
from quport import MultiQPUConfig, benchmark_random_circuits

cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=4, comm_qubits_per_qpu=1)
rows = benchmark_random_circuits(
    cfg,
    n_logical=6,
    depth=10,
    trials=3,
    seed=10,
    out_csv="results.csv",
    strategies=("baseline", "balanced", "tpccap"),
)
print(rows[0])
```

The returned rows match the CSV content, so you can inspect results immediately and
also persist them for later analysis.

## Topology sweep

```python
from quport import sweep_topologies

sweep_topologies(
    n_logical=6,
    depth=10,
    trials=2,
    seed=5,
    out_csv="sweep.csv",
    intra_topologies=("clique", "ring"),
    inter_topologies=("switch", "ring"),
    comm_ports=(1, 2),
    compute_per_qpu=4,
    n_qpus=2,
    strategies=("baseline", "balanced", "tpccap"),
)
```

Start with small sweeps like this before scaling up. Topology sweeps can become
expensive because each setting runs multiple random circuits and strategies.

## Loading and dumping config files

```python
from quport.config import dump_config, load_config
from quport import MultiQPUConfig

cfg = MultiQPUConfig(n_qpus=4, compute_qubits_per_qpu=6, comm_qubits_per_qpu=2)
dump_config(cfg, "config.json")
loaded = load_config("config.json")
assert loaded == cfg
```

Use config files for benchmark runs that need to be repeated or shared with other
researchers.
