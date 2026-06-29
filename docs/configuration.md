# Configuration

Configuration is split into two public dataclasses:

- `MultiQPUConfig` describes topology, capacity, scheduler resource limits, and Qiskit transpiler knobs.
- `LatencyModel` describes scalar operation/network costs used by cost and schedule estimators.

Both are intentionally lightweight and serializable so experiments can store config
files beside benchmark outputs.

## `MultiQPUConfig`

`MultiQPUConfig` describes the physical architecture and Qiskit transpiler options.

| Field | Default | Meaning | Validation / interaction |
|---|---:|---|---|
| `n_qpus` | `10` | number of QPUs | architecture builders require positive values |
| `compute_qubits_per_qpu` | `8` | local compute qubits per QPU | non-negative; contributes to capacity and physical qubit count |
| `comm_qubits_per_qpu` | `1` | communication qubits per QPU | non-negative; controls inter-QPU physical links and schedule port limits |
| `intra_topology` | `"clique"` | one of `clique`, `line`, `ring`, `grid2d` | invalid topology names are rejected by architecture construction |
| `inter_topology` | `"switch"` | one of `switch`, `mesh`, `ring`, `degree_d`, `clos`, `fat_tree` | invalid topology names are rejected by graph/architecture construction |
| `inter_degree` | `2` | degree target for `degree_d` topology | clamped/validated by QPU graph helpers depending on context |
| `link_capacity` | `1` | max simultaneous remote ops per inter-QPU link per topology-scheduler round | topology scheduler uses this as a per-round resource limit |
| `switch_parallel_links` | `1_000_000` | max distinct QPU pairs per round for switch/mesh scheduling | use smaller values to model limited switch fanout |
| `switch_reconfig_delay` | `0.0` | additional delay per communication round | added by topology-aware scheduling |
| `async_classical` | `True` | enables asynchronous classical-latency hiding in topology scheduling | must be boolean |
| `async_overlap` | `0.5` | fraction of classical RTT that can be hidden | finite numeric value in `[0, 1]` |
| `grid_rows` | `None` | optional rows for `grid2d` | inferred if omitted; explicit rows/cols must cover local qubits |
| `grid_cols` | `None` | optional columns for `grid2d` | inferred if omitted; explicit rows/cols must cover local qubits |
| `basis_gates` | `("rz", "sx", "x", "cx")` | Qiskit basis gates used by translation/transpilation | non-empty sequence of non-empty strings |
| `optimization_level` | `3` | Qiskit optimization level | integer `0`, `1`, `2`, or `3` |
| `layout_method` | `"sabre"` | Qiskit layout method for global transpilation | non-empty string |
| `routing_method` | `"sabre"` | Qiskit routing method | non-empty string |

### Derived quantities

- `total_physical_qubits()` returns `n_qpus * (compute_qubits_per_qpu + comm_qubits_per_qpu)`.
- `capacity_per_qpu()` returns `compute_qubits_per_qpu + comm_qubits_per_qpu`.
- The physical block size for one QPU is `capacity_per_qpu()`.

### Topology-specific notes

- `grid2d` uses row-major local-qubit placement. If both dimensions are omitted,
  dimensions are inferred. If one dimension is provided, the other is inferred.
- `degree_d` uses `inter_degree` to create a bounded-degree QPU graph.
- `clos` and `fat_tree` are abstractions, not detailed switch-level hardware descriptions.
- `switch_parallel_links`, `link_capacity`, `async_classical`, and `async_overlap`
  matter most in topology-aware scheduling.

## `LatencyModel`

`LatencyModel` stores cost coefficients for metrics and schedule estimates.

| Field | Default | Meaning | Used by |
|---|---:|---|---|
| `oneq` | `1.0` | one-qubit operation cost | cost and schedule estimators |
| `twoq` | `10.0` | local two-qubit operation cost | cost and schedule estimators |
| `swap` | `30.0` | SWAP cost | cost and coarse schedule estimators |
| `epr_gen` | `200.0` | entanglement generation/network setup cost | remote operation cost |
| `classical_rtt` | `20.0` | classical round-trip component | remote operation cost and async overlap model |
| `remote_gate_overhead` | `50.0` | remote gate protocol overhead | remote operation cost |

`estimate_latency(n_1q, n_2q, swaps, remote_2q, depth=None)` computes a scalar
latency proxy and validates all counts and coefficients before computing. The
optional `depth` argument adds a soft depth penalty and is not a replacement for a
schedule estimator.

## Config files

Use the CLI to create a config file:

```bash
quport gen-config --out quport_config.yaml
```

JSON and YAML are supported by `load_config`/`dump_config`; YAML requires the
`yaml` extra. Config files must contain a mapping/object and unknown fields are
rejected so misspelled settings do not silently take effect.

Example JSON:

```json
{
  "n_qpus": 4,
  "compute_qubits_per_qpu": 8,
  "comm_qubits_per_qpu": 2,
  "intra_topology": "ring",
  "inter_topology": "degree_d",
  "inter_degree": 2,
  "basis_gates": ["rz", "sx", "x", "cx"],
  "optimization_level": 3,
  "layout_method": "sabre",
  "routing_method": "sabre"
}
```

## Experiment design guidance

- Change one family of fields at a time when running sweeps.
- Keep `basis_gates`, `layout_method`, `routing_method`, and `optimization_level`
  stable when comparing partitioning strategies.
- Use more than one random trial for benchmark claims; a single random circuit can
  overstate or understate topology effects.
- Store the config file, seed range, strategies, and QuPort version/commit beside
  benchmark outputs for reproducibility.
