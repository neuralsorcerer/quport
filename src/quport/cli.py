# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.exceptions import MissingOptionalLibraryError
from rich.console import Console
from rich.table import Table

from quport.compiler import compile_distributed
from quport.config import (
    LatencyModel,
    MultiQPUConfig,
    dump_config,
    load_config,
    optional_module_available,
)
from quport.distributed import write_remote_ops_json
from quport.pipeline import (
    benchmark_method_labels,
    benchmark_random_circuits,
    map_and_transpile,
    random_benchmark_circuit,
    sweep_topologies,
)

app = typer.Typer(
    add_completion=False, help="QuPort: multi-QPU circuit mapping + benchmarks"
)
console = Console()

_QASM_VERSION_RE = re.compile(r"\AOPENQASM\s+([23])(?:\.0)?\s*;", re.ASCII)


def _qasm_version(source: str) -> int | None:
    """Return the declared OpenQASM major version, ignoring leading comments."""
    remaining = source.lstrip("\ufeff \t\r\n")
    while True:
        if remaining.startswith("//"):
            _comment, separator, rest = remaining.partition("\n")
            if not separator:
                return None
            remaining = rest.lstrip(" \t\r\n")
            continue
        if remaining.startswith("/*"):
            end = remaining.find("*/", 2)
            if end < 0:
                return None
            remaining = remaining[end + 2 :].lstrip(" \t\r\n")
            continue
        break

    match = _QASM_VERSION_RE.match(remaining)
    if match is None:
        return None
    return int(match.group(1))


def _load_qasm_circuit(input_qasm: str) -> QuantumCircuit:
    """Load an OpenQASM 2/3 circuit with clear errors for optional dependencies."""
    input_path = Path(input_qasm)
    try:
        source = input_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise typer.BadParameter(
            f"Unable to read --input-qasm file {input_qasm!r}: {exc}"
        ) from exc

    version = _qasm_version(source)
    if version == 2:
        try:
            return qasm2.load(str(input_path))
        except Exception as exc:
            raise typer.BadParameter(
                f"Unable to parse OpenQASM 2 input {input_qasm!r}: {exc}"
            ) from exc

    if version == 3:
        try:
            return qasm3.load(str(input_path))
        except MissingOptionalLibraryError as exc:
            raise typer.BadParameter(
                "OpenQASM 3 input requires Qiskit's optional importer. "
                "Install it with: pip install qiskit_qasm3_import"
            ) from exc
        except Exception as exc:
            raise typer.BadParameter(
                f"Unable to parse OpenQASM 3 input {input_qasm!r}: {exc}"
            ) from exc

    try:
        return qasm3.load(str(input_path))
    except MissingOptionalLibraryError:
        try:
            return qasm2.load(str(input_path))
        except Exception as exc:
            raise typer.BadParameter(
                "Unable to detect an OpenQASM version header and the input could "
                f"not be parsed as OpenQASM 2: {exc}"
            ) from exc
    except Exception as qasm3_exc:
        try:
            return qasm2.load(str(input_path))
        except Exception as qasm2_exc:
            raise typer.BadParameter(
                "Unable to detect an OpenQASM version header. Parsing failed as "
                f"OpenQASM 3 ({qasm3_exc}) and OpenQASM 2 ({qasm2_exc})."
            ) from qasm2_exc


def _load_or_random_circuit(
    *,
    input_qasm: str | None,
    n_logical: int | None,
    depth: int,
    seed: int,
) -> QuantumCircuit:
    """Load an OpenQASM 2/3 circuit or generate the configured random benchmark."""
    if input_qasm:
        return _load_qasm_circuit(input_qasm)
    if n_logical is None:
        raise typer.BadParameter(
            "--n-logical is required when --input-qasm is not provided"
        )
    return random_benchmark_circuit(n_logical, depth, seed)


def _load_plot_modules() -> tuple[Any, Any]:
    missing = [
        module_name
        for module_name in ("matplotlib.pyplot", "pandas")
        if not optional_module_available(module_name)
    ]
    if missing:
        raise typer.BadParameter("Plot requires extras: pip install -e '.[viz]'")
    return importlib.import_module("matplotlib.pyplot"), importlib.import_module(
        "pandas"
    )


def _pretty_config(cfg: MultiQPUConfig) -> None:
    t = Table(title="MultiQPUConfig")
    t.add_column("field")
    t.add_column("value")
    for k, v in cfg.__dict__.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def gen_config(
    out: str = typer.Option("quport_config.yaml", help="Output path (.json/.yaml)"),
) -> None:
    """Generate an example config file."""
    cfg = MultiQPUConfig()
    dump_config(cfg, out)
    console.print(f"Wrote config to {out}")
    _pretty_config(cfg)


@app.command()
def map(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed for random circuit + transpiler"),
    strategy: str = typer.Option(
        "tpccap", help="Partition strategy: balanced, cluster, tpccap, tpccap_sa"
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
    out: str | None = typer.Option(None, help="Write mapped circuit as OpenQASM 3.0"),
) -> None:
    """Map+transpile a single random circuit and print key metrics."""
    cfg = load_config(config) if config else MultiQPUConfig()
    latency = LatencyModel()
    qc = _load_or_random_circuit(
        input_qasm=input_qasm, n_logical=n_logical, depth=depth, seed=seed
    )

    res = map_and_transpile(qc, cfg, latency=latency, seed=seed, strategy=strategy)
    m = res.metrics

    console.print(
        f"[bold]SWAPs:[/bold] {m.swaps}  [bold]Remote2Q:[/bold] {m.remote_2q}  [bold]Depth:[/bold] {m.depth}"
    )
    console.print(
        f"[bold]Cost:[/bold] {res.cost.total:.2f} (local={res.cost.local:.2f}, remote={res.cost.remote:.2f})"
    )
    console.print(
        f"[bold]Times:[/bold] mapping={res.mapping_time_s:.4f}s  transpile={res.transpile_time_s:.4f}s"
    )

    if out:
        Path(out).write_text(qasm3.dumps(res.mapped_circuit), encoding="utf-8")
        console.print(f"Wrote mapped circuit to {out}")


@app.command()
def bench(
    n_logical: int = typer.Option(..., help="Number of logical qubits"),
    depth: int = typer.Option(20, help="Random circuit depth"),
    trials: int = typer.Option(10, help="Number of random circuits"),
    seed: int = typer.Option(0, help="Base seed"),
    strategies: str = typer.Option(
        "baseline,balanced,tpccap",
        help="Comma-separated strategies: baseline,balanced,cluster,tpccap,tpccap_sa",
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    out: str = typer.Option("results.csv", help="Output CSV path"),
) -> None:
    """Benchmark baseline vs QuPort on multiple random circuits."""
    cfg = load_config(config) if config else MultiQPUConfig()
    latency = LatencyModel()

    strats = [s.strip() for s in strategies.split(",") if s.strip()]
    rows = benchmark_random_circuits(
        cfg,
        n_logical,
        depth,
        trials,
        seed=seed,
        latency=latency,
        out_csv=out,
        strategies=strats,
    )
    console.print(f"Wrote {len(rows)} rows to {out}")
    _pretty_config(cfg)


@app.command()
def sweep(
    n_logical: int = typer.Option(..., help="Number of logical qubits"),
    depth: int = typer.Option(20, help="Random circuit depth"),
    trials: int = typer.Option(5, help="Trials per setting"),
    seed: int = typer.Option(0, help="Base seed"),
    out: str = typer.Option("sweep.csv", help="Output CSV summary"),
    strategies: str = typer.Option(
        "baseline,balanced,tpccap",
        help="Comma-separated strategies: baseline,balanced,cluster,tpccap,tpccap_sa",
    ),
    plot: str | None = typer.Option(
        None, help="Optional PNG plot (requires quport[viz])"
    ),
) -> None:
    """Sweep multiple topologies and comm-port counts; save summary CSV."""
    sweep_topologies(
        n_logical=n_logical,
        depth=depth,
        trials=trials,
        seed=seed,
        out_csv=out,
        strategies=[s.strip() for s in strategies.split(",") if s.strip()],
    )
    console.print(f"Wrote sweep summary to {out}")

    if plot:
        plt, pd = _load_plot_modules()

        df = pd.read_csv(out)
        fig = plt.figure()
        method_labels = benchmark_method_labels()
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            plt.scatter(
                sub["ports"],
                sub["cost_mean"],
                label=method_labels.get(float(method), str(method)),
            )
        plt.xlabel("comm ports per QPU")
        plt.ylabel("mean estimated cost")
        plt.legend()
        fig.savefig(plot, dpi=180, bbox_inches="tight")
        console.print(f"Wrote plot to {plot}")


@app.command()
def schedule(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed"),
    strategy: str = typer.Option(
        "tpccap", help="Partition strategy: balanced, cluster, tpccap, tpccap_sa"
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
) -> None:
    """Estimate parallel multi-QPU makespan for a mapped random circuit."""
    from .architecture import MultiQPUArchitecture
    from .schedule import estimate_parallel_makespan_layered

    cfg = load_config(config) if config else MultiQPUConfig()
    latency = LatencyModel()
    qc = _load_or_random_circuit(
        input_qasm=input_qasm, n_logical=n_logical, depth=depth, seed=seed
    )
    res = map_and_transpile(qc, cfg, latency=latency, seed=seed, strategy=strategy)
    arch = MultiQPUArchitecture(cfg)
    summ = estimate_parallel_makespan_layered(res.mapped_circuit, arch, latency)
    console.print(
        f"[bold]Makespan:[/bold] {summ.makespan:.2f}  [bold]RemoteOps:[/bold] {summ.remote_ops}  [bold]SyncSteps:[/bold] {summ.steps}"
    )


@app.command()
def split(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed"),
    strategy: str = typer.Option(
        "tpccap", help="Partition strategy: balanced, cluster, tpccap, tpccap_sa"
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
    out_dir: str = typer.Option(
        "distributed_out", help="Output directory for per-QPU QASM files"
    ),
) -> None:
    """Split a mapped circuit into per-QPU local circuits + remote-op list (JSON)."""
    from .architecture import MultiQPUArchitecture
    from .distributed import split_into_qpus

    cfg = load_config(config) if config else MultiQPUConfig()
    latency = LatencyModel()
    qc = _load_or_random_circuit(
        input_qasm=input_qasm, n_logical=n_logical, depth=depth, seed=seed
    )
    res = map_and_transpile(qc, cfg, latency=latency, seed=seed, strategy=strategy)
    arch = MultiQPUArchitecture(cfg)
    prog = split_into_qpus(res.mapped_circuit, arch)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # write per-QPU QASM3
    for qpu, c in prog.local_circuits.items():
        (outp / f"qpu_{qpu}.qasm").write_text(qasm3.dumps(c), encoding="utf-8")

    # write remote ops
    write_remote_ops_json(prog.remote_ops, outp / "remote_ops.json")

    console.print(
        f"Wrote {len(prog.local_circuits)} local circuits and {len(prog.remote_ops)} remote ops to {out_dir}"
    )


@app.command()
def compile_dist(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed for random circuit + transpiler"),
    strategy: str = typer.Option(
        "tpccap_sa", help="Partition strategy: balanced, cluster, tpccap, tpccap_sa"
    ),
    temporal_decay: float = typer.Option(
        0.98, help="Time-decay factor for 2Q weights (<=1). Use 1 for uniform."
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
    out_dir: str = typer.Option(
        "compile_out", help="Output directory (per-QPU QASM3 + remote/schedule JSON)"
    ),
) -> None:
    """Distributed compile (no cross-QPU SWAPs).

    Outputs:
      - qpu_<id>_routed.qasm : routed per-QPU local programs
      - remote_ops.json     : ordered remote-op trace
      - schedule.json       : topology-aware schedule summary
      - schedule_trace.json : detailed per-layer/per-round communication plan
    """
    cfg = load_config(config) if config else MultiQPUConfig()
    latency = LatencyModel()
    qc = _load_or_random_circuit(
        input_qasm=input_qasm, n_logical=n_logical, depth=depth, seed=seed
    )

    res = compile_distributed(
        qc,
        cfg,
        latency=latency,
        seed=seed,
        strategy=strategy,
        temporal_decay=temporal_decay,
    )

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    for qpu, c in res.local_routed.items():
        (outp / f"qpu_{qpu}_routed.qasm").write_text(qasm3.dumps(c), encoding="utf-8")

    write_remote_ops_json(res.program.remote_ops, outp / "remote_ops.json")
    (outp / "schedule.json").write_text(
        json.dumps(asdict(res.schedule), indent=2), encoding="utf-8"
    )
    (outp / "schedule_trace.json").write_text(
        json.dumps(asdict(res.schedule_plan), indent=2), encoding="utf-8"
    )

    swaps_total = sum(m.get("swap", 0) for m in res.local_metrics.values())
    console.print(
        f"[bold]Remote2Q:[/bold] {res.global_metrics.remote_2q}  [bold]Local SWAPs:[/bold] {swaps_total}"
    )
    console.print(
        f"[bold]Makespan (topology-aware):[/bold] {res.schedule.makespan:.2f}  [bold]Remote rounds:[/bold] {res.schedule.remote_rounds}"
    )
    console.print(
        f"[bold]Times:[/bold] mapping={res.mapping_time_s:.4f}s  local_transpile={res.local_transpile_time_s:.4f}s"
    )
    console.print(f"Wrote artifacts to {out_dir}")
