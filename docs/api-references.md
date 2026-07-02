# API reference

This page documents the public API exported from `quport.__all__` plus the most
commonly used lower-level helpers. It is intentionally descriptive rather than
auto-generated so it can explain validation, modeling assumptions, and how return
objects should be interpreted.

All public validators reject booleans where an integer or numeric value is expected.
This matters because Python treats `bool` as a subclass of `int`, but QuPort treats
boolean values as almost always accidental in numerical configuration.

## Package marker

QuPort ships `py.typed`, so type checkers can consume the package's inline type
annotations when it is installed from a built distribution. Public annotations are
therefore part of the developer-facing API and should not be weakened just to make
negative tests easier to write.

## Configuration and architecture

### `MultiQPUConfig`

```python
MultiQPUConfig(
    n_qpus=10,
    compute_qubits_per_qpu=8,
    comm_qubits_per_qpu=1,
    intra_topology="clique",
    inter_topology="switch",
    inter_degree=2,
    link_capacity=1,
    switch_parallel_links=1_000_000,
    switch_reconfig_delay=0.0,
    async_classical=True,
    async_overlap=0.5,
    grid_rows=None,
    grid_cols=None,
    basis_gates=("rz", "sx", "x", "cx"),
    optimization_level=3,
    layout_method="sabre",
    routing_method="sabre",
)
```

Primary architecture/transpiler configuration. Use `total_physical_qubits()` and
`capacity_per_qpu()` to derive aggregate sizes. The dataclass validates fields that
can be checked locally in `__post_init__`, while topology-specific validation occurs
when graph/coupling-map builders are invoked.

### `LatencyModel`

```python
LatencyModel(oneq=1.0, twoq=10.0, swap=30.0, epr_gen=200.0,
             classical_rtt=20.0, remote_gate_overhead=50.0)
```

Cost model used by mapping costs and schedule estimators. `estimate_latency(...)`
returns a finite non-negative scalar latency proxy. Coefficients are arbitrary units;
use consistent units within a study.

### `MultiQPUArchitecture`

```python
arch = MultiQPUArchitecture(cfg)
```

Builds physical and QPU-level topology from `MultiQPUConfig`.

Important methods:

- `qpu_of_phys(p) -> int`: map a physical qubit index to a QPU id; rejects out-of-range indices.
- `block_of_qpu(qpu_id) -> QPUBlock`: return compute and communication qubits for one QPU.
- `all_blocks() -> list[QPUBlock]`: return all QPU blocks in QPU-id order.
- `build_coupling_map() -> CouplingMap`: build full directed global coupling map.
- `build_intra_coupling_map(qpu_id) -> CouplingMap`: build local directed coupling map.
- `qpu_shortest_paths() -> QpuShortestPaths`: compute QPU-level shortest paths.

`QPUBlock` has `compute` and `comm` lists. Those lists contain physical qubit indices,
not logical qubit indices.

### `topology_metrics`

```python
from quport import MultiQPUConfig, topology_metrics
from quport.network import build_qpu_graph

metrics = topology_metrics(build_qpu_graph(MultiQPUConfig(inter_topology="ring")))
```

Computes validated inter-QPU graph diagnostics as a `TopologyMetrics` dataclass.
Fields include `n_qpus`, `edges`, `min_degree`, `max_degree`, `average_degree`,
`connected`, `components`, `diameter`, `average_shortest_path`, and
`unreachable_pairs`. Distances are calculated over reachable unordered QPU pairs;
`unreachable_pairs` makes disconnected or zero-port topologies explicit instead of
folding them into the average.

## Mapping pipeline

### `map_and_transpile`

```python
map_and_transpile(qc, cfg, latency=None, seed=None, strategy="balanced") -> MapResult
```

End-to-end global mapping flow:

1. validate circuit capacity and optional non-negative integer seed;
2. translate to configured basis gates;
3. partition logical qubits;
4. compute layout hints;
5. run Qiskit transpilation on the global coupling map;
6. compute metrics and cost.

Supported strategies: `balanced`, `cluster`, `tpccap`, `tpccap_sa`.

`MapResult` fields:

| Field | Meaning |
|---|---|
| `mapped_circuit` | routed Qiskit `QuantumCircuit` on the global coupling map |
| `cfg` | architecture config used for the run |
| `partition` | logical-qubit-to-QPU assignment |
| `partition_cut` | weighted cut value used by/derived from the partition |
| `strategy` | partitioning strategy name |
| `partition_diagnostics` | topology-aware diagnostics when available |
| `mapping_time_s` | partition/layout-hint time |
| `transpile_time_s` | Qiskit transpilation time |
| `metrics` | `CircuitMetrics` for the mapped circuit |
| `cost` | `CostBreakdown` estimated from metrics and latency model |

Important edge cases:

- If `qc.num_qubits > cfg.total_physical_qubits()`, `ValueError` is raised.
- If `seed` is provided, it must be a non-negative integer and not a boolean.
- Unknown strategies are rejected before layout/transpilation.

### `benchmark_random_circuits`

```python
benchmark_random_circuits(cfg, n_logical, depth, trials, seed=0,
                          latency=None, out_csv=None,
                          strategies=("baseline", "balanced", "tpccap"))
```

Runs random-circuit benchmarks and optionally writes a numeric-friendly CSV.
Strategies may include `baseline`, `balanced`, `cluster`, `tpccap`, and `tpccap_sa`.
The returned rows include trial, seed, method id, strategy, SWAPs, remote-2Q count,
depth, size, costs, and timing columns.

Validation highlights:

- `n_logical`, `depth`, `trials`, and `seed` must be non-negative integers.
- `strategies` must be a non-string sequence of strings.
- Duplicate and unknown strategies are rejected.
- Zero trials are allowed and still write a header when `out_csv` is provided.

### `sweep_topologies`

```python
sweep_topologies(n_logical, depth, trials, seed, out_csv,
                 intra_topologies=("clique", "line", "ring"),
                 inter_topologies=("switch", "ring", "degree_d", "clos"),
                 comm_ports=(1, 2), compute_per_qpu=8,
                 n_qpus=10, inter_degree=2,
                 strategies=("baseline", "balanced", "tpccap"))
```

Sweeps topology and communication-port settings and writes a summary CSV with mean
SWAPs, remote 2Q operations, depth, cost, and transpilation time. Configurations
whose physical capacity cannot fit `n_logical` are skipped.

## Distributed compilation

### `compile_distributed`

```python
compile_distributed(qc, cfg, latency=None, seed=None,
                    strategy="tpccap_sa", temporal_decay=0.98) -> DistributedCompileResult
```

Distributed compilation flow that preserves cross-QPU operations as explicit remote
events. The optional seed is validated as a non-negative integer before Qiskit calls.
Supported strategies: `balanced`, `cluster`, `tpccap`, `tpccap_sa`.

The `temporal_decay` argument is used for topology-aware strategies. It must be in
`(0, 1]` when applicable. Values closer to `1` behave more like uniform interaction
weights; smaller values emphasize earlier two-qubit interactions more strongly.

`DistributedCompileResult` fields:

| Field | Meaning |
|---|---|
| `physical_circuit` | basis-translated, physically laid-out circuit without global routing |
| `cfg` | architecture config |
| `strategy` | partitioning strategy |
| `partition` | logical-qubit-to-QPU assignment |
| `partition_cut` | weighted cut value |
| `partition_diagnostics` | topology-aware partition diagnostics when available |
| `anneal_diagnostics` | simulated annealing diagnostics for `tpccap_sa` |
| `program` | `DistributedProgram` containing local circuits and remote ops |
| `local_routed` | per-QPU locally routed circuits |
| `global_metrics` | metrics on the physical, not globally routed, circuit |
| `local_metrics` | per-QPU operation counts after local routing |
| `schedule` | topology-aware schedule summary |
| `schedule_plan` | detailed layer/round schedule trace |
| `mapping_time_s` | partition/layout time |
| `local_transpile_time_s` | local per-QPU transpilation time |

### `split_into_qpus`

```python
split_into_qpus(mapped, arch) -> DistributedProgram
```

Splits a mapped physical circuit into per-QPU local circuits and ordered remote
operation metadata. Local one-qubit and intra-QPU two-qubit operations are appended
to the owning QPU circuit. Cross-QPU operations become `RemoteOp` entries and
barriers are inserted on involved local circuits to mark synchronization points.

Zero-qubit operations are broadcast to all local circuits unless they are barriers,
which are represented as barriers. Multi-qubit operations that span multiple QPUs
are conservatively represented as remote composite operations.

### `DistributedProgram`

```python
DistributedProgram(local_circuits: dict[int, QuantumCircuit],
                   remote_ops: list[RemoteOp])
```

Methods:

- `remote_ops_payload() -> list[dict[str, Any]]`: JSON-safe remote-operation payload.

The local circuits currently use the full physical register for clarity. Downstream
consumers may shrink circuits if their execution environment prefers per-QPU local
registers only.

### `RemoteOp`

```python
RemoteOp(name, q0_phys, q1_phys, qpu0, qpu1, params, clbits, index)
```

Represents one remote operation placeholder. `to_dict()` validates fields and
returns a deterministic JSON-safe representation. Non-finite floats, complex values,
bytes, sets, mappings, and nested sequences in parameters are encoded safely for JSON.

Field meanings:

- `name`: source operation name, such as `cx`;
- `q0_phys`, `q1_phys`: physical qubit indices involved in the remote operation;
- `qpu0`, `qpu1`: owning QPUs for those physical qubits;
- `params`: operation parameters converted through JSON-safe encoding;
- `clbits`: classical bit indices associated with the source instruction;
- `index`: source instruction index in the physical circuit.

### Writers

```python
write_remote_ops_json(remote_ops, path) -> None
write_distributed_program(program, path, *, include_empty_circuits=True) -> dict[str, Path]
```

`write_remote_ops_json` writes a standards-compliant JSON manifest with
`allow_nan=False`. `write_distributed_program` writes `qpu_<id>.qasm` files and a
`remote_ops.json` manifest to a directory, validating all artifact inputs before writing.

`write_distributed_program` returns a mapping of artifact labels to `Path` objects.
Keys are stable (`qpu_<id>` and `remote_ops`) so callers can report or post-process
outputs without re-deriving file names.

## Scheduling API

### `estimate_parallel_makespan`

```python
estimate_parallel_makespan(mapped, arch, model) -> ScheduleSummary
```

Coarse synchronized QPU timeline estimator. It walks the circuit, accumulates local
operation costs per QPU, and synchronizes timelines when remote operations occur.

### `estimate_parallel_makespan_layered`

```python
estimate_parallel_makespan_layered(mapped, arch, model) -> ScheduleSummary
```

DAG-layer estimator with communication-port-limited remote rounds. It approximates
parallel local execution within each DAG layer and groups remote operations into
rounds according to available QPU communication ports.

### `estimate_parallel_makespan_topology`

```python
estimate_parallel_makespan_topology(mapped, arch, model) -> TopologyScheduleSummary
```

Topology-aware estimator with QPU ports, link capacity, switch pair budgets,
reconfiguration delay, asynchronous classical overlap, and unschedulable penalties.
Use this estimator for network-resource studies.

### `estimate_topology_schedule_plan`

```python
estimate_topology_schedule_plan(mapped, arch, model) -> TopologySchedulePlan
```

Returns the topology-aware summary plus a detailed trace of layers and remote rounds.
Each layer and each remote round carries absolute `start_time` and `end_time` offsets,
so callers can render timelines or feed simulators without recomputing cumulative
durations. This is the most useful API when diagnosing why a makespan increased.

### Schedule dataclasses

- `ScheduleSummary(makespan, steps, remote_ops)`
- `TopologyScheduleSummary(makespan, layers, remote_ops, remote_rounds, peak_link_util, peak_qpu_ports_used)`
- `RemoteRoundTrace(layer_index, round_index, qpu_pairs, duration, qpu_ports_used, link_utilization, unschedulable_ops=0, start_time=0.0, end_time=0.0)`
- `LayerScheduleTrace(layer_index, local_duration, remote_ops, remote_rounds, duration, start_time=0.0, end_time=0.0)`
- `TopologySchedulePlan(summary, layers)`

`LayerScheduleTrace.start_time` is the absolute offset at which the DAG layer begins,
and `LayerScheduleTrace.end_time` is `start_time + duration`. Remote rounds are
serialized from the containing layer's `start_time`; each `RemoteRoundTrace.start_time`
is the previous round's `end_time` (or the layer start for the first round), and
`RemoteRoundTrace.end_time` is `start_time + duration`. If local work is longer than
the cumulative remote rounds, the layer end can be later than the final remote-round
end because local work occupies the full layer interval.

## Interaction and metrics helpers

### `WeightValue`

`WeightValue` is a type alias for values accepted by validated interaction-weight
APIs: `SupportsFloat | SupportsIndex | str`. Runtime validation still rejects
booleans, non-finite values, negative values, invalid edge keys, and out-of-range
logical indices.

### `degree`

```python
degree(weights, n) -> list[float]
```

Computes weighted degree for `n` logical nodes. Self-loops and zero-weight edges are
ignored after validation.

### `cut_weight` / `compute_cut`

```python
cut_weight(weights, part) -> float
compute_cut(weights, part) -> float
```

Computes the total weight of interactions crossing partition boundaries. `part`
must be a list of non-negative integer QPU assignments and must be long enough for
all logical indices referenced by `weights`.

### `CircuitMetrics`

```python
CircuitMetrics(swaps, depth, size, n_1q, n_2q, remote_2q)
```

Returned by `compute_metrics(qc, arch)` and embedded in mapping/distributed results.
`remote_2q` is derived by comparing the QPU ownership of each two-qubit operation's
physical operands.

## Lower-level modules worth knowing

Although not all lower-level helpers are exported from `quport.__all__`, advanced
users may import from submodules directly:

- `quport.pipeline.random_benchmark_circuit` for deterministic random circuits;
- `quport.config.load_config` and `quport.config.dump_config` for JSON/YAML configs;
- `quport.metrics.compute_metrics` and `quport.metrics.count_ops` for direct analysis;
- `quport.interaction.extract_twoq_weights` and `extract_temporal_twoq_weights` for partition diagnostics;
- `quport.network.build_qpu_graph` for QPU-level topology inspection.
