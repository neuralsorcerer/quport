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
             classical_rtt=20.0, remote_gate_overhead=50.0,
             epr_success_prob=1.0)
```

Cost model used by mapping costs and schedule estimators. `estimate_latency(...)`
returns a finite non-negative scalar latency proxy. Coefficients are arbitrary units;
use consistent units within a study.

`epr_success_prob` is the heralded entanglement success probability per attempt, in
`(0, 1]`. `expected_epr_time(hops)` returns `hops * epr_gen / epr_success_prob`, the
expected time to distribute one pair across `hops` links. Only
`estimate_entanglement_schedule` reads it; the default of `1.0` models a
deterministic link, so `estimate_cost` and the older estimators are unaffected.

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
map_and_transpile(qc, cfg, latency=None, seed=None, strategy="balanced",
                  temporal_decay=None) -> MapResult
```

End-to-end global mapping flow:

1. validate circuit capacity and optional non-negative integer seed;
2. translate to configured basis gates;
3. partition logical qubits;
4. compute layout hints;
5. run Qiskit transpilation on the global coupling map;
6. compute metrics and cost.

Supported strategies: `balanced`, `cluster`, `ebit`, `tpccap`, `tpccap_sa`.

`temporal_decay` selects the interaction weighting for the topology-aware
strategies and applies to `tpccap` and `tpccap_sa` alike, so a run isolates the
annealing rather than also changing the objective's inputs. Leaving it `None`
keeps the historical split, in which `tpccap` uses uniform counts and
`tpccap_sa` uses a decay of `0.98`; that default is unchanged so existing
benchmark numbers do not move. See
[Partitioning strategies](concepts.md#partitioning-strategies) for why this
matters when interpreting benchmark output.

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
Strategies may include `baseline`, `balanced`, `cluster`, `ebit`, `tpccap`, and `tpccap_sa`.
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
Supported strategies: `balanced`, `cluster`, `ebit`, `tpccap`, `tpccap_sa`.

The `temporal_decay` argument is used for topology-aware strategies. It must be in
`(0, 1]` when applicable. Values closer to `1` behave more like uniform interaction
weights; smaller values emphasize earlier two-qubit interactions more strongly.

`DistributedCompileResult` fields:

| Field | Meaning |
|---|---|
| `physical_circuit` | basis-translated, physically laid-out circuit without global routing |
| `basis_circuit` | the circuit after basis translation, before layout: what `partition` and `packets` are indexed by |
| `cfg` | architecture config |
| `strategy` | partitioning strategy |
| `partition` | logical-qubit-to-QPU assignment |
| `partition_cut` | weighted cut value |
| `partition_diagnostics` | topology-aware partition diagnostics when available |
| `anneal_diagnostics` | simulated annealing diagnostics for `tpccap_sa` and `ebit` |
| `program` | `DistributedProgram` containing local circuits and remote ops |
| `local_routed` | per-QPU locally routed circuits |
| `routed_remote_ops` | remote-op manifest re-expressed in `local_routed`'s labelling |
| `global_metrics` | metrics on the physical, not globally routed, circuit |
| `local_metrics` | per-QPU operation counts after local routing |
| `schedule` | topology-aware schedule summary |
| `schedule_plan` | detailed layer/round schedule trace |
| `packets` | distributable packets of the basis-translated circuit |
| `ebits` | `EbitReport` for the chosen partition, with unlimited ports |
| `aggregation` | `AggregationPlan` of EPR blocks under the real comm-port budget |
| `entanglement_schedule` | `EntanglementScheduleSummary` for that plan |
| `mapping_time_s` | partition/layout time |
| `local_transpile_time_s` | local per-QPU transpilation time |

### `split_into_qpus`

```python
split_into_qpus(mapped, arch) -> DistributedProgram
```

```python
remap_remote_ops_to_routed(remote_ops, local_routed) -> list[RemoteOp]
```

Each `RemoteOp` also carries `qpu0_marker` and `qpu1_marker`: the position of
its synchronization barrier among all barriers of that QPU's program, counting
from zero. Emitted QASM carries no labels, and routing can reorder barriers that
sit on disjoint qubits, so these are what let a consumer pair a manifest entry
with the right barrier.

Re-expresses a manifest in the labelling of routed local programs. Routing
permutes qubits inside a QPU unless the intra-QPU topology is a clique, so a
manifest recorded before routing points at the wrong qubits afterwards. The
routed circuits carry the answer: `split_into_qpus` labels each synchronization
barrier with the operation it marks, and the transpiler remaps those barriers
along with everything else. `compile_distributed` applies this automatically and
returns the result as `routed_remote_ops`.

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

### `estimate_entanglement_schedule`

```python
estimate_entanglement_schedule(mapped, arch, model, *, plan=None,
                               ports_per_qpu=None) -> EntanglementScheduleSummary
```

Event-driven as-soon-as-possible schedule over aggregated EPR blocks, with one
timeline per physical qubit, a pool of comm ports held for a whole block, per-link
channels, and hop-scaled probabilistic entanglement distribution. Unlike the
layer-based estimators it imposes no global barrier between layers and charges one
transaction per block rather than per gate, so its makespan is not comparable with
theirs.

`plan` defaults to `aggregate_remote_operations(mapped, arch, ports_per_qpu=...)`.
A supplied plan must have been built with the same port budget; a larger one raises
`ValueError` rather than silently reporting the work as unschedulable.

See [Entanglement model](entanglement.md).

### `audit_topology_schedule_plan`

```python
audit_topology_schedule_plan(plan, arch, model, mapped=None) -> tuple[str, ...]
```

Re-derives a `TopologySchedulePlan`'s numbers from the outside and returns one
description per inconsistency; an empty tuple means every figure was reproduced.
The estimator builds the summary and the trace in one pass, so nothing in it
cross-checks the two or checks either against the budgets the plan claims to
respect. This checks that:

- layer and round intervals chain, and each `end_time` is `start_time + duration`;
- a layer lasts `max(local_duration, sum of its round durations)`;
- each round's `qpu_ports_used` and `link_utilization` are exactly what routing its
  `qpu_pairs` consumes, within `comm_qubits_per_qpu` and `link_capacity`;
- each round lasts as long as its slowest placed operation;
- the six summary fields agree with the trace they summarise.

Passing `mapped` adds the check that needs the circuit: that the plan accounts for
exactly the operations spanning more than one QPU, counted as `split_into_qpus`
counts them. The cost model itself — per-hop EPR time, classical-RTT overlap, the
round-packing policy — is taken as given; those are modelling choices, and the
audit checks the plan is a faithful, feasible account of them.

`quport compile-dist` runs this before writing `schedule_trace.json` and refuses to
write a manifest that does not add up.

### `audit_entanglement_schedule`

```python
audit_entanglement_schedule(summary, mapped, arch, model, *,
                            plan=None, ports_per_qpu=None) -> tuple[str, ...]
```

The counterpart for `estimate_entanglement_schedule`, which returns aggregates and
no trace — so instead of replaying events, this checks everything the aggregates
are related by *theorem*:

- a QPU with `P` ports cannot hold more than `P` cat copies at once, nor accrue
  more than `P * makespan` of port-busy time; a link with `C` channels likewise;
- `entanglement_time` equals the sum of `link_busy_time`, by definition;
- every gate on a qubit occupies that qubit's timeline, so the busiest qubit's own
  work is a lower bound on the makespan;
- `remote_gates` is the circuit's own count of operations spanning more than one
  QPU, and no more gates can be unschedulable than there are remote ones;
- with `plan`, the summary describes that plan's blocks and EPR pairs.

`ports_per_qpu` must be whatever the schedule was produced with. Monotonicity —
that widening a resource never lengthens a schedule — is a property of the
estimator rather than of one result, so it is obtained by scheduling one fixed
`plan` twice and comparing; `tests/test_entanglement_schedule.py` does exactly
that for ports, link channels, `epr_gen`, and `epr_success_prob`.

`quport compile-dist` and `quport ebits --out` both run this before writing
`entanglement_plan.json`.

### Entanglement API

```python
diagonal_positions(operation) -> frozenset[int]
build_distributable_packets(qc, *, symmetric_root="greedy") -> PacketDecomposition
ebit_cost(decomposition, part, n_qpus) -> int
ebit_objective(decomposition, part, n_qpus, dist=None) -> tuple[int, float]
ebit_report(decomposition, part, n_qpus) -> EbitReport
ebit_traffic_matrix(decomposition, part, n_qpus) -> list[list[float]]
aggregate_remote_operations(mapped, arch, *, ports_per_qpu=None,
                            max_block_gates=None) -> AggregationPlan
```

- `diagonal_positions` reports which operand positions commute with `Z`, the
  condition under which a cat copy of that operand survives an operation.
- `build_distributable_packets` extracts placement-independent packets; `ebit_cost`
  scores any partition against them with the λ−1 metric, and `ebit_objective` adds
  the hop-weighted variant used by the `ebit` strategy.
- `ebit_report` returns `EbitReport(ebits, baseline_ebits, reduction, packets,
  active_packets, packed_gates, cut_gates, unpackable_ebits, peak_cat_copies,
  pair_ebits)`.
- `aggregate_remote_operations` turns a mapped circuit into `RemoteBlock` objects
  under a real comm-port budget, evicting the least recently used cat copy when a
  port is needed. With unlimited ports its `epr_pairs` equals `ebit_cost`.

`tpccap_partition` and `tpccap_sa_partition` accept `packets`, `w_ebit`, and
`congestion_source`. `w_ebit=0.0` and `congestion_source="gates"` (the defaults)
leave their historical objectives unchanged and only fill in the `ebits` and
`weighted_ebit_distance` diagnostics. `congestion_source="ebits"` routes EPR demand
rather than gate demand through the congestion term, and requires `packets`; gate
demand upper-bounds EPR demand, because aggregation only ever removes transactions.

`compile_distributed` returns `packets`, `ebits`, `aggregation`, and
`entanglement_schedule` alongside its existing fields.

### Exact partitioning

```python
optimal_partition(n, n_qpus, capacity, *, objective="cut", weights=None,
                  packets=None, max_nodes=DEFAULT_MAX_NODES) -> ExactPartition
partition_gap(part, n_qpus, capacity, *, objective="cut", weights=None,
              packets=None, max_nodes=DEFAULT_MAX_NODES) -> PartitionGap
```

Branch and bound over set partitions of at most `n_qpus` blocks, for both
objectives QuPort optimises: `"cut"` (needs `weights`) and `"ebits"` (needs
`packets`). Symmetry is broken by restricted-growth canonical form, which is valid
because both objectives are invariant under relabelling QPUs and the capacity is
uniform.

- `ExactPartition(part, objective, proved_optimal, nodes)`. `proved_optimal` is
  false when `max_nodes` ran out first; `objective` is then an upper bound on the
  optimum, and `part` is still feasible and still the best found.
- `PartitionGap(heuristic, optimal, proved_optimal)` with `absolute` and `relative`
  properties. `relative` is `0.0` when the heuristic is optimal and `math.inf` when
  the optimum is zero and the heuristic is not. `partition_gap` raises if `part` is
  infeasible, or if it scores below a proved optimum — which would mean one of the
  two implementations is wrong.

The tree is over set partitions, so this terminates on roughly a dozen qubits. It
exists to calibrate the heuristics, not to compile.

### Time-varying placement

```python
split_windows(decomposition, count) -> tuple[TemporalWindow, ...]
static_temporal_partition(part, windows) -> TemporalPartition
temporal_ebit_cost(decomposition, partition, n_qpus, *,
                   migration_cost=1) -> TemporalCost
optimize_temporal_partition(decomposition, part, n_qpus, capacity, windows, *,
                            migration_cost=1, max_passes=8,
                            seed=None) -> TemporalResult
```

Prices letting qubits move between QPUs mid-circuit, in the same e-bit model.

- `TemporalWindow(start, stop)` is a half-open instruction range; windows must be
  contiguous and the first must start at 0. `split_windows` cuts the stream into
  windows of similar communication content, returning fewer than asked for when
  the circuit has too few relevant gates.
- `TemporalPartition(windows, assignments)` holds one assignment per window, with
  `migrations()` and `to_dict()`.
- `TemporalCost(packet_ebits, unpackable_ebits, migration_ebits, moves)` and its
  `total`. Cat copies are counted over *root epochs*, so a copy survives a window
  boundary when its root does not move.
- `TemporalResult(partition, cost, static_cost, stationary_cost, passes)` with
  `saved` / `reduction` against the seed, and `migration_saved` /
  `migration_reduction` against the best placement the same search reaches with
  migration forbidden. `cost.total <= stationary_cost <= static_cost` holds by
  construction.

With one window, or the same assignment in every window, `temporal_ebit_cost`
returns exactly `ebit_cost` for that assignment.

### Reassembly and verification of a distributed program

```python
reassemble_distributed_program(mapped, local_routed, remote_ops, arch, *,
                               restore_layout=True) -> QuantumCircuit
verify_distributed_program(mapped, local_routed, remote_ops, arch, *,
                           seed=0, atol=1e-9) -> bool
```

`reassemble_distributed_program` is the inverse of `split_into_qpus`: it merges
per-QPU programs and a remote-op manifest back into one circuit and, with
`restore_layout=True`, undoes each QPU's routing permutation so the result is
directly comparable with the mapped circuit. Classical bits and registers are
carried across, so measurements and conditioned operations survive the round
trip. `verify_distributed_program` does that comparison by state-vector
simulation: terminating measurements are dropped, and a circuit with mid-circuit
measurement or reset is refused.

A distributed program is a **partial** order. Within a QPU the constraint is per
qubit, so instructions on disjoint qubits may run in either order, and two QPUs
can list the same remote operations in opposite orders. Merging follows qubit
dataflow: an instruction runs once it leads on every qubit it touches, and a
remote operation once its marker leads on both sides. Reading each program
strictly linearly can deadlock; `reassemble_distributed_program` raises when the
programs genuinely contradict each other.

Pass `routed_remote_ops` with `local_routed`, or `program.remote_ops` with
`program.local_circuits`.

### Protocol emission and verification

```python
build_telegate_circuit(mapped, arch, plan=None, *, coherent=True,
                       ports_per_qpu=None) -> TelegateProgram
verify_telegate_equivalence(mapped, arch, plan=None, *, seed=0, atol=1e-9,
                            ports_per_qpu=None) -> bool
```

`build_telegate_circuit` expands an aggregation plan into the circuit that
performs it. `coherent=True` (default) emits the deferred-measurement form,
which is unitary and therefore verifiable; `coherent=False` emits real
mid-circuit measurement with `if` feedforward and `reset`, suitable for OpenQASM
3 export. Ancillas are recycled between blocks, so the emitted width follows the
number of concurrently live cat copies.

`TelegateProgram(circuit, n_data, ancillas, blocks, epr_pairs,
unschedulable_gates, measured)` — the first `n_data` qubits of `circuit` are the
mapped circuit's, in order; the rest are protocol ancillas that start and end in
the ground state. `n_ancillas` is a convenience property.

`verify_telegate_equivalence` simulates the unitary form on a pseudo-random
product input, traces out the ancillas, and compares the data qubits' reduced
state with the mapped circuit's. It raises for plans with unschedulable gates,
and for circuits wider than `quport.protocol.MAX_VERIFIABLE_QUBITS` (24).

### Schedule dataclasses

- `ScheduleSummary(makespan, steps, remote_ops)`
- `TopologyScheduleSummary(makespan, layers, remote_ops, remote_rounds, peak_link_util, peak_qpu_ports_used)`
- `RemoteRoundTrace(layer_index, round_index, qpu_pairs, duration, qpu_ports_used, link_utilization, unschedulable_ops=0, start_time=0.0, end_time=0.0)`
- `LayerScheduleTrace(layer_index, local_duration, remote_ops, remote_rounds, duration, start_time=0.0, end_time=0.0)`
- `TopologySchedulePlan(summary, layers)`
- `EntanglementScheduleSummary(makespan, blocks, epr_pairs, remote_gates, unschedulable_gates, entanglement_time, peak_ports_in_use, port_busy_time, qpu_busy_time, link_busy_time)`

`TopologyScheduleSummary`, `RemoteRoundTrace`, `LayerScheduleTrace`, and
`TopologySchedulePlan` provide `to_dict()` methods that return stable JSON-ready
payloads. The conversion normalizes tuple-heavy fields such as QPU pairs and link
utilization into ordinary JSON arrays/objects, so command-line schedule exports and
downstream simulators can consume traces without depending on Python dataclass
serialization details. The serializers also revalidate manifest fields and reject
non-finite timings, negative counts, booleans in integer fields, malformed QPU/link
pairs, and self-loop pairs before emitting payloads.

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

`swaps` counts instructions literally named `swap`, which makes it basis-dependent.
The default `basis_gates` of `("rz", "sx", "x", "cx")` has no `swap`, so Qiskit
rewrites routing SWAPs into CX gates and `swaps` is `0` for every run, with the
routing overhead appearing in `n_2q` instead. Include `"swap"` in `basis_gates` to
keep the instructions and make the metric meaningful.

A `swap` instruction counts toward both `swaps` and `n_2q`, so in `estimate_cost`
the `swap` latency is an overhead on top of the two-qubit cost: a surviving SWAP
costs `twoq + swap`, which is `40` under the default `LatencyModel` rather than `30`.

## Lower-level modules worth knowing

Although not all lower-level helpers are exported from `quport.__all__`, advanced
users may import from submodules directly:

- `quport.pipeline.random_benchmark_circuit` for deterministic random circuits;
- `quport.config.load_config` and `quport.config.dump_config` for JSON/YAML configs;
- `quport.metrics.compute_metrics` and `quport.metrics.count_ops` for direct analysis;
- `quport.interaction.extract_twoq_weights` and `extract_temporal_twoq_weights` for partition diagnostics;
- `quport.network.build_qpu_graph` for QPU-level topology inspection;
- `quport.network.prepare_routing_tables` and `route_prepared_link_loads` to route many traffic matrices over one interconnect without re-validating the shortest-path tables each time, and `accumulate_traffic` / `accumulate_boundary_counts` for the same reason on edge weights. These reproduce their validating counterparts exactly; the caller takes on the input contract;
- `quport.entanglement.acts_diagonally` and `breaks_cat_copy` for custom cat-copy analyses;
- `quport.hypergraph.DistributablePacket` for inspecting individual packets.
