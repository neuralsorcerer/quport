# QuPort Documentation

QuPort is a Python and Qiskit toolkit for modeling, mapping, routing, splitting,
scheduling, and benchmarking quantum circuits on modular multi-QPU machines. The
project models each QPU as a block of compute and communication qubits, exposes
multiple interconnect abstractions, and provides both global-routing and explicit
distributed-compilation workflows.

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

getting-started
concepts
configuration
cli
api-references
examples
development
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project links

GitHub repository <https://github.com/neuralsorcerer/quport>
Package releases <https://github.com/neuralsorcerer/quport/releases>
```

## Documentation map

| Page | What it answers | Primary audience |
|---|---|---|
| [Getting started](getting-started.md) | How do I install QuPort and run the first mapping/compile? | New users |
| [Concepts](concepts.md) | What are the machine, partitioning, routing, and scheduling models? | Researchers and users interpreting results |
| [Configuration](configuration.md) | What does every config/latency field mean, and how do fields interact? | Experiment authors |
| [CLI reference](cli.md) | What commands are available and what files do they produce? | CLI users and automation authors |
| [API reference](api-references.md) | What Python objects/functions are public and what do they return? | Library users |
| [Examples](examples.md) | What are complete snippets for common workflows? | Notebook/script authors |
| [Development](development.md) | How should contributors test, format, type-check, and update docs? | Contributors |

## Recommended reading order

1. Start with [Getting started](getting-started.md) if you want to run QuPort quickly.
2. Read [Concepts](concepts.md) before interpreting benchmark or scheduling numbers.
3. Use [Configuration](configuration.md) while designing architecture sweeps.
4. Use [API reference](api-references.md) while writing experiments against the Python API.
5. Use [CLI reference](cli.md) when scripting command-line workflows.
6. Use [Development](development.md) before contributing changes.

## Workflow decision guide

| Goal | Recommended entry point | Why |
|---|---|---|
| Compare mapped-circuit depth/SWAPs across partitioning strategies | `map_and_transpile` or `quport map` | Produces one globally routed Qiskit circuit and standard circuit metrics |
| Produce per-QPU local programs and explicit remote operations | `compile_distributed` or `quport compile-dist` | Keeps cross-QPU gates as remote events instead of hiding them inside global routing |
| Run repeated random-circuit comparisons | `benchmark_random_circuits` or `quport bench` | Writes row-oriented benchmark metrics suitable for CSV analysis |
| Sweep topology/port settings | `sweep_topologies` or `quport sweep` | Aggregates repeated benchmark rows by architecture setting |
| Inspect communication bottlenecks | `estimate_topology_schedule_plan` | Returns layer/round traces with port and link utilization |

## Documentation accuracy policy

The API reference mirrors the exported objects in `quport.__all__` and the CLI
reference mirrors the Typer commands in `quport.cli`. If a public symbol, command,
configuration field, output artifact, validation rule, or conceptual model changes,
update the corresponding page in this directory in the same pull request.

When docs and implementation disagree, treat the implementation and tests as the
source of truth, then fix the documentation immediately. For behavior that is subtle
or research-model dependent, document both what QuPort currently does and what it
does not claim to model.
