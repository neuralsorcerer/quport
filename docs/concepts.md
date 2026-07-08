# Concepts

This page explains the modeling assumptions behind QuPort. The goal is to make
metrics and schedules interpretable: QuPort is a research toolkit, so a result is
only meaningful when the architecture, partitioning, routing, and cost assumptions
are clear.

## Architecture model

A QuPort machine is configured as `N` QPUs. Each QPU has:

- compute qubits for ordinary local computation;
- communication qubits for inter-QPU links;
- local connectivity controlled by `intra_topology`;
- network connectivity controlled by `inter_topology`.

For QPU `q`, physical qubits are contiguous. If each QPU has block size
`compute_qubits_per_qpu + comm_qubits_per_qpu`, QPU `q` owns the corresponding
block of physical qubit indices. This convention keeps physical-to-QPU mapping
simple and makes per-QPU circuit splitting deterministic.

## Compute qubits vs communication qubits

Compute qubits are ordinary local qubits used for circuit operations. Communication
qubits represent ports that can participate in inter-QPU communication. In the
current global coupling-map abstraction, communication qubits are still physical
qubits in the Qiskit circuit. In the distributed compiler abstraction, they guide
placement and scheduling pressure while remote operations remain explicit metadata.

A configuration with zero communication qubits can still be useful as a stress test:
remote operations become difficult or unschedulable in topology-aware estimates,
which exposes whether a partitioning/layout choice depends on unavailable network
resources.

## Local topologies

`intra_topology` controls edges inside each QPU:

| Topology | Meaning | Notes |
|---|---|---|
| `clique` | all local qubits connect to all other local qubits | idealized all-to-all local device |
| `line` | local qubits form a path | strict nearest-neighbor baseline |
| `ring` | local qubits form a cycle when at least three local qubits exist | reduces endpoint effects vs `line` |
| `grid2d` | local qubits are arranged row-major in a 2D grid | uses inferred or configured rows/columns |

All physical links are represented bidirectionally in Qiskit's directed
`CouplingMap`, because Qiskit treats two-qubit coupling direction as part of the
routing model.

## Inter-QPU topologies

`inter_topology` controls QPU-network connectivity:

| Topology | Meaning | Typical use |
|---|---|---|
| `switch` | switch-like all-to-all QPU communication model | idealized centralized network |
| `mesh` | all-to-all QPU adjacency | all-pairs QPU graph abstraction |
| `ring` | each QPU connects to the next QPU modulo `n_qpus` | sparse baseline with uniform degree |
| `degree_d` | bounded-degree circulant-style QPU graph controlled by `inter_degree` | degree/cost scaling experiments |
| `clos` | Clos-style approximation; one-port configurations fall back to a ring abstraction | multi-port network experiments |
| `fat_tree` | tree-like/pod-style approximation with representative communication links | hierarchical network experiments |

Scheduling and congestion calculations operate on the QPU-level graph, not only on
physical-qubit coupling edges. This is why topology choice can affect both
partition quality and makespan estimates.

## Interaction graph

QuPort scans two-qubit circuit instructions and builds an undirected weighted
logical interaction graph. An edge `(i, j)` means logical qubits `i` and `j`
interact. The weight is normally the count of two-qubit interactions. Some
partitioners can use temporal decay so earlier two-qubit gates receive higher
weight.

This interaction graph is a heuristic summary. It does not encode the full circuit
DAG, gate commutation, or all timing dependencies. It is used to choose partitions
that are likely to reduce cross-QPU traffic before routing/scheduling.

## Partitioning strategies

| Strategy | Use case | Main objective |
|---|---|---|
| `baseline` | identity-style layout baseline for benchmark comparisons | expose unoptimized placement costs |
| `balanced` | capacity-constrained greedy partitioning baseline | spread logical qubits while considering interaction weight |
| `cluster` | heavy-edge clustering baseline | keep strongly interacting qubits together |
| `tpccap` | topology, port, and congestion-aware partitioning | reduce cut and route pressure together |
| `tpccap_sa` | `tpccap` with simulated annealing refinement | refine topology-aware objective after construction |

`baseline` is available in benchmark workflows. `map_and_transpile` and
`compile_distributed` accept `balanced`, `cluster`, `tpccap`, and `tpccap_sa`.

## Capacity model

A QPU can host at most `compute_qubits_per_qpu + comm_qubits_per_qpu` logical qubits
in the global layout model. This capacity choice intentionally treats communication
qubits as physical placement resources. For distributed compilation, communication
ports also influence how remote operations are scheduled.

If the circuit has more logical qubits than total physical qubits, public pipelines
raise `ValueError` before invoking Qiskit.

## Global mapping and routing

`map_and_transpile` builds the full architecture coupling map, computes a partition
and partition-aware initial layout, and asks Qiskit to route the entire circuit on
the global graph. This is useful for comparing routed circuit metrics such as
SWAPs, depth, and cross-QPU two-qubit operations.

Because this flow gives Qiskit a global coupling map, it is intentionally a global
routing experiment. It is not the same as producing executable distributed-control
programs with explicit network operations.

## Distributed compilation

`compile_distributed` keeps cross-QPU two-qubit operations explicit:

1. translate the input circuit into the configured basis gates;
2. partition logical qubits across QPUs;
3. assign logical qubits to physical qubits;
4. split the physical circuit into local per-QPU circuits and remote-operation metadata;
5. route only the local circuits inside each QPU;
6. estimate topology-aware remote-operation scheduling.

This mode is the recommended representation when remote gates are implemented by a
network protocol instead of physical cross-QPU SWAP routing. It does not implement
a hardware-specific teleportation, telegate, or entanglement-swapping protocol; it
produces the artifacts needed for such a controller or simulator to consume.

## Scheduling

QuPort exposes three schedule estimators:

- `estimate_parallel_makespan`: coarse QPU timeline synchronization model;
- `estimate_parallel_makespan_layered`: DAG-layer model with QPU port limits;
- `estimate_parallel_makespan_topology`: topology-aware model with QPU ports, link capacity,
  switch pair budgets, reconfiguration delay, and unreachable-pair penalties.

Use `estimate_topology_schedule_plan` when you need the detailed layer and round trace.
Its trace includes absolute `start_time` / `end_time` offsets for both layers and
remote rounds, making it suitable for timeline visualizations and simulator input.
The topology-aware estimator is the most informative one for network bottleneck
analysis because it reports remote rounds, peak link utilization, and peak QPU port
usage.

For artifact export, call `plan.to_dict()` rather than serializing the dataclass
directly. The serializer converts QPU pairs and link-utilization pairs into
JSON-native arrays/objects and validates finite non-negative timings, non-negative
counts, and non-self QPU/link pairs before returning the manifest.

## Interpreting metrics

| Metric | Interpretation | Caveat |
|---|---|---|
| SWAP count | routing overhead inserted by transpilation | depends heavily on basis/layout/routing settings |
| Depth | Qiskit circuit depth | not a complete wall-clock execution time |
| Remote 2Q count | two-qubit operations crossing QPU boundaries | protocol cost depends on latency model |
| Cut weight | weighted partition boundary cost | computed before final routing |
| Makespan | schedule-estimator time proxy | model-dependent, not hardware-calibrated by default |
| Peak link utilization | maximum simultaneous per-link pressure in a round | topology-scheduler metric |

For fair comparisons, keep random seeds, config, transpiler settings, strategies,
and latency model fixed except for the variable being studied.
