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

## Detailed schedule trace

```python
from quport import MultiQPUArchitecture, MultiQPUConfig, LatencyModel, estimate_topology_schedule_plan
from quport.pipeline import random_benchmark_circuit, map_and_transpile

cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1)
qc = random_benchmark_circuit(4, depth=4, seed=2)
mapped = map_and_transpile(qc, cfg, LatencyModel(), seed=2, strategy="balanced")
plan = estimate_topology_schedule_plan(mapped.mapped_circuit, MultiQPUArchitecture(cfg), LatencyModel())

print(plan.summary)
for layer in plan.layers:
    print(layer.layer_index, layer.start_time, layer.end_time, layer.remote_ops)
    for round_trace in layer.remote_rounds:
        print("  round", round_trace.round_index, round_trace.start_time, round_trace.end_time)
```

Use the trace when a summary value such as `remote_rounds` or `makespan` changes and
you need to understand which layer/round caused it. The absolute timing fields let
you plot a timeline directly without reconstructing cumulative offsets.

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
