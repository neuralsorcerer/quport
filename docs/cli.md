# CLI reference

The package installs the `quport` command. The CLI is intended for quick
experiments, reproducible benchmark generation, and artifact export. For complex
custom workflows, use the Python API documented in [API reference](api-references.md).

## General conventions

- Command names use hyphens, for example `gen-config` and `compile-dist`.
- Strategy values use the Python strategy names, for example `tpccap_sa`.
- Config paths may be JSON or YAML; YAML requires the `yaml` extra.
- Output directories are created when possible by artifact-writing commands.
- Commands that operate on one circuit (`map`, `schedule`, `split`, and
  `compile-dist`) can either generate a random benchmark with `--n-logical` or
  load a user OpenQASM 2/3 circuit with `--input-qasm path/to/circuit.qasm`.
  OpenQASM 2 loads with Qiskit's built-in parser; OpenQASM 3 requires Qiskit's
  optional `qiskit_qasm3_import` package.

## `quport gen-config`

```bash
quport gen-config --out quport_config.yaml
```

Writes an example JSON/YAML config and prints the resolved `MultiQPUConfig`.
Use this as the safest starting point for editing architecture fields because it
contains all current config keys.

## `quport map`

```bash
quport map --n-logical 6 --depth 5 --seed 1 --strategy balanced --config quport_config.yaml --out mapped.qasm
```

Generates a random circuit, maps/transpiles it on the global architecture, prints
SWAP/remote/depth/cost/timing metrics, and optionally writes OpenQASM 3.

Options:

- `--n-logical`: logical qubit count;
- `--depth`: random circuit depth;
- `--seed`: random circuit and transpiler seed;
- `--strategy`: `balanced`, `cluster`, `tpccap`, or `tpccap_sa`;
- `--config`: optional JSON/YAML config path;
- `--input-qasm`: optional OpenQASM 2/3 file to map instead of generating a random circuit;
- `--out`: optional mapped OpenQASM 3 output.

Use this command when you want to see the globally routed circuit that Qiskit
produces for one architecture and one partitioning strategy.

## `quport bench`

```bash
quport bench --n-logical 8 --depth 20 --trials 10 --strategies baseline,balanced,tpccap --out results.csv
```

Runs random-circuit benchmarks and writes one CSV row per trial/strategy. The CSV
is suitable for pandas, spreadsheets, and plotting tools.

Important columns include:

- `trial` and `seed` for reproducibility;
- `method` and `strategy` for grouping;
- `swaps`, `remote_2q`, `depth`, and `size` for mapped circuit metrics;
- `cost_total`, `cost_local`, and `cost_remote` for latency-model costs;
- `mapping_time_s` and `transpile_time_s` for runtime comparisons.

## `quport sweep`

```bash
quport sweep --n-logical 8 --depth 20 --trials 5 --out sweep.csv --plot sweep.png
```

Sweeps built-in topology and port settings. `--plot` requires `quport[viz]`. The
CSV contains aggregate means rather than one row per random circuit. Use `bench`
when you need raw per-trial rows.

## `quport schedule`

```bash
quport schedule --n-logical 6 --depth 5 --seed 1 --strategy tpccap
```

Maps a random or `--input-qasm` circuit and prints a layered makespan estimate.
This is a fast way to compare whether a mapped circuit's remote operations are
likely to serialize under communication-port limits.

## `quport split`

```bash
quport split --n-logical 6 --depth 5 --seed 1 --strategy tpccap --out-dir distributed_out
```

Maps a random or `--input-qasm` circuit globally, splits the mapped circuit into
per-QPU local QASM files, and writes `remote_ops.json`.

This command is useful for inspecting how a globally routed circuit is decomposed,
but it is not the preferred distributed-compilation workflow. Prefer `compile-dist`
when you want to avoid cross-QPU global routing and keep remote operations explicit
from the compilation flow.

## `quport compile-dist`

```bash
quport compile-dist --n-logical 6 --depth 5 --seed 1 --strategy tpccap_sa --temporal-decay 0.98 --out-dir compile_out
```

Runs distributed compilation without cross-QPU SWAP routing. Use `--input-qasm`
to compile an application circuit from OpenQASM 2/3 instead of generating a random
benchmark. Output artifacts:

- `qpu_<id>_routed.qasm`: locally routed per-QPU programs;
- `remote_ops.json`: ordered remote operation manifest;
- `schedule.json`: topology-aware schedule summary;
- `schedule_trace.json`: detailed per-layer/per-round communication plan.

Recommended checks after running:

```bash
python -m json.tool compile_out/remote_ops.json >/dev/null
python -m json.tool compile_out/schedule.json >/dev/null
python -m json.tool compile_out/schedule_trace.json >/dev/null
```

## Choosing between CLI commands

| If you need... | Use... |
|---|---|
| one globally routed Qiskit circuit | `quport map` |
| repeated global-routing comparisons | `quport bench` |
| topology/port aggregate summaries | `quport sweep` |
| quick makespan estimate | `quport schedule` |
| per-QPU split of a globally mapped circuit | `quport split` |
| explicit distributed compile artifacts | `quport compile-dist` |
