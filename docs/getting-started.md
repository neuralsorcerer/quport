# Getting started

This guide gets you from a clean checkout to a first mapped circuit and a first
distributed compile. It also explains what to look at in the returned objects so
that the first run is useful rather than just successful.

## Install

From a checked-out repository:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e '.[yaml]'
python -m pip install -e '.[viz]'
python -m pip install -e '.[graph]'
```

- `yaml` enables YAML config loading and dumping.
- `viz` enables CSV plotting workflows used by the CLI sweep command.
- `graph` is reserved for graph-oriented workflows that may use NetworkX.

If you are developing QuPort itself, also install pre-commit as described in
[Development](development.md).

## Minimal Python mapping run

```python
from quport import LatencyModel, MultiQPUConfig, map_and_transpile
from quport.pipeline import random_benchmark_circuit

cfg = MultiQPUConfig(
    n_qpus=2,
    compute_qubits_per_qpu=4,
    comm_qubits_per_qpu=1,
    intra_topology="ring",
    inter_topology="ring",
)
qc = random_benchmark_circuit(n_logical=6, depth=5, seed=1)
result = map_and_transpile(qc, cfg, latency=LatencyModel(), seed=1, strategy="balanced")

print(result.partition)
print(result.metrics)
print(result.cost)
```

What to inspect:

- `result.partition` tells you which QPU each logical qubit was assigned to.
- `result.mapped_circuit` is the globally routed physical Qiskit circuit.
- `result.metrics.swaps` shows local/global routing overhead from Qiskit.
- `result.metrics.remote_2q` counts two-qubit operations whose endpoints are on different QPUs.
- `result.cost` combines local operations, remote operations, SWAPs, and depth into a scalar proxy.

## Minimal CLI run

Generate a config:

```bash
quport gen-config --out quport_config.yaml
```

Map and transpile a random circuit:

```bash
quport map --n-logical 6 --depth 5 --seed 1 --strategy balanced --config quport_config.yaml
```

Compile into a distributed program bundle:

```bash
quport compile-dist --n-logical 6 --depth 5 --seed 1 --strategy tpccap_sa --out-dir compile_out
```

Inspect `compile_out/`:

- `qpu_<id>_routed.qasm` files are locally routed QPU programs.
- `remote_ops.json` is the ordered list of remote operations.
- `schedule.json` is a strict JSON topology-aware summary produced by `TopologyScheduleSummary.to_dict()`.
- `schedule_trace.json` is the strict JSON layer/round trace produced by `TopologySchedulePlan.to_dict()`, including absolute `start_time` and `end_time` offsets for timeline inspection.

Both schedule exports reject non-finite timings and malformed resource fields before
writing JSON, which keeps generated artifacts compatible with `python -m json.tool`
and other standards-compliant JSON consumers.

## Choosing a compilation mode

Use **global mapping and routing** (`map_and_transpile`) when you want one Qiskit
physical circuit on the global coupling map and want to measure how global routing
behaves. This mode is useful for comparing conventional transpiler output, SWAP
counts, depth, and global circuit size.

Use **distributed compilation** (`compile_distributed`) when inter-QPU operations
should stay explicit as remote events rather than becoming ordinary routed gates or
cross-device SWAPs. This mode better matches distributed-control experiments where
remote gates are implemented by networking protocols.

## Reproducibility

Pass a non-negative integer `seed` to mapping, benchmarking, and distributed
compilation APIs. QuPort validates seeds before forwarding them to Qiskit in the
public pipelines. Use the same seed, config, strategy, basis gates, layout method,
routing method, and optimization level when comparing results.

## Common first-run errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Logical qubits exceed physical qubits | `n_logical > cfg.total_physical_qubits()` | Increase QPUs/qubits or reduce the circuit size |
| YAML config fails to load | PyYAML extra is missing | Install `python -m pip install -e '.[yaml]'` |
| Plotting a sweep fails | visualization extras are missing | Install `python -m pip install -e '.[viz]'` |
| Strategy is rejected | strategy is not supported by that workflow | Use `balanced`, `cluster`, `tpccap`, or `tpccap_sa`; `baseline` is benchmark-only |
| Seed is rejected | seed is boolean, negative, or non-integral | Use a non-negative integer such as `0` or `123` |
