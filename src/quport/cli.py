# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.exceptions import MissingOptionalLibraryError
from rich.console import Console
from rich.table import Table

from quport.architecture import MultiQPUArchitecture
from quport.compiler import compile_distributed
from quport.config import (
    LatencyModel,
    MultiQPUConfig,
    dump_config,
    load_config,
    optional_module_available,
)
from quport.distributed import write_remote_ops_json
from quport.exact import DEFAULT_MAX_NODES, Objective, partition_gap
from quport.interaction import extract_twoq_weights
from quport.network import build_qpu_graph, topology_metrics
from quport.pipeline import (
    benchmark_method_labels,
    benchmark_random_circuits,
    map_and_transpile,
    random_benchmark_circuit,
    sweep_topologies,
)
from quport.schedule import (
    audit_entanglement_schedule,
    audit_topology_schedule_plan,
)

app = typer.Typer(
    add_completion=False, help="QuPort: multi-QPU circuit mapping + benchmarks"
)
console = Console()

_QASM_VERSION_RE = re.compile(r"\AOPENQASM\s+([23])(?:\.0)?\s*;", re.ASCII)


def _print_path(message: str) -> None:
    """Print a message containing a filesystem path, keeping it on one line.

    Rich wraps at the console width -- 80 columns when stdout is not a
    terminal -- which splits long paths across lines and breaks copy/paste
    and any downstream parsing. Deep temporary directories on macOS and
    Windows cross that width routinely. Markup and highlighting are off so a
    path containing square brackets is printed verbatim rather than being
    read as markup.
    """
    console.print(message, soft_wrap=True, markup=False, highlight=False)


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
        # Typer/Rich may syntax-highlight option-looking text inside validation
        # errors (for example under FORCE_COLOR), which can split the literal
        # "--n-logical" with ANSI escape codes.  Emit a plain copy first so CLI
        # users and tests can reliably match the actionable requirement.
        typer.echo(
            "--n-logical is required when --input-qasm is not provided", color=False
        )
        raise typer.BadParameter(
            "--n-logical is required when --input-qasm is not provided"
        )
    return random_benchmark_circuit(n_logical, depth, seed)


def _load_config_or_default(config: str | None) -> MultiQPUConfig:
    """Load a config file, or fall back to defaults when none is given.

    ``load_config`` raises ``RuntimeError`` when a YAML path is requested without
    the optional PyYAML dependency.  Surface that as a CLI error so users see the
    install hint instead of a traceback, matching how the plotting extra is
    handled.
    """
    if config is None:
        return MultiQPUConfig()
    try:
        return load_config(config)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _dump_config_or_fail(cfg: MultiQPUConfig, out: str) -> None:
    """Write a config file, reporting a missing optional dependency cleanly."""
    try:
        dump_config(cfg, out)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    out: str = typer.Option(
        "quport_config.json",
        help="Output path. Use a .yaml/.yml suffix for YAML (needs quport[yaml]).",
    ),
) -> None:
    """Generate an example config file."""
    cfg = MultiQPUConfig()
    _dump_config_or_fail(cfg, out)
    _print_path(f"Wrote config to {out}")
    _pretty_config(cfg)


@app.command("topology-info")
def topology_info(
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
) -> None:
    """Print structural metrics for the configured inter-QPU topology."""
    cfg = _load_config_or_default(config)
    metrics = topology_metrics(build_qpu_graph(cfg))

    t = Table(title="Inter-QPU Topology Metrics")
    t.add_column("metric")
    t.add_column("value")
    for key, value in asdict(metrics).items():
        if isinstance(value, float):
            t.add_row(key, f"{value:.6g}")
        else:
            t.add_row(key, str(value))
    console.print(t)


@app.command()
def map(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed for random circuit + transpiler"),
    strategy: str = typer.Option(
        "tpccap",
        help="Partition strategy: balanced, cluster, ebit, tpccap, tpccap_sa",
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
    cfg = _load_config_or_default(config)
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
        _print_path(f"Wrote mapped circuit to {out}")


@app.command()
def bench(
    n_logical: int = typer.Option(..., help="Number of logical qubits"),
    depth: int = typer.Option(20, help="Random circuit depth"),
    trials: int = typer.Option(10, help="Number of random circuits"),
    seed: int = typer.Option(0, help="Base seed"),
    strategies: str = typer.Option(
        "baseline,balanced,tpccap",
        help="Comma-separated strategies: baseline,balanced,cluster,ebit,tpccap,tpccap_sa",
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    out: str = typer.Option("results.csv", help="Output CSV path"),
) -> None:
    """Benchmark baseline vs QuPort on multiple random circuits."""
    cfg = _load_config_or_default(config)
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
    _print_path(f"Wrote {len(rows)} rows to {out}")
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
        help="Comma-separated strategies: baseline,balanced,cluster,ebit,tpccap,tpccap_sa",
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
    _print_path(f"Wrote sweep summary to {out}")

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
        _print_path(f"Wrote plot to {plot}")


@app.command()
def schedule(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed"),
    strategy: str = typer.Option(
        "tpccap",
        help="Partition strategy: balanced, cluster, ebit, tpccap, tpccap_sa",
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
) -> None:
    """Estimate parallel multi-QPU makespan for a mapped random circuit."""
    from .schedule import estimate_parallel_makespan_layered

    cfg = _load_config_or_default(config)
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
        "tpccap",
        help="Partition strategy: balanced, cluster, ebit, tpccap, tpccap_sa",
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
    from .distributed import split_into_qpus

    cfg = _load_config_or_default(config)
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

    _print_path(
        f"Wrote {len(prog.local_circuits)} local circuits and {len(prog.remote_ops)} remote ops to {out_dir}"
    )


@app.command()
def ebits(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed for random circuit + transpiler"),
    strategy: str = typer.Option(
        "ebit",
        help="Partition strategy: balanced, cluster, ebit, tpccap, tpccap_sa",
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
    out: str | None = typer.Option(
        None, help="Optional JSON path for the full entanglement plan"
    ),
    emit_qasm: str | None = typer.Option(
        None,
        "--emit-qasm",
        help=(
            "Optional OpenQASM 3 path for the executable telegate circuit "
            "(explicit EPR pairs, mid-circuit measurement, and feedforward)"
        ),
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help=(
            "Check by state-vector simulation that the emitted protocol "
            "reproduces the mapped circuit (small circuits only)"
        ),
    ),
) -> None:
    """Report EPR-pair (e-bit) demand after communication aggregation.

    Compares the entanglement a per-gate telegate compiler would consume against
    the aggregated plan, then schedules that plan against the configured
    comm-port and link budgets.
    """
    cfg = _load_config_or_default(config)
    latency = LatencyModel()
    qc = _load_or_random_circuit(
        input_qasm=input_qasm, n_logical=n_logical, depth=depth, seed=seed
    )

    res = compile_distributed(qc, cfg, latency=latency, seed=seed, strategy=strategy)
    plan = res.aggregation
    report = res.ebits
    sched = res.entanglement_schedule

    table = Table(title="Entanglement Demand")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("cross-QPU gates", str(plan.remote_gates))
    table.add_row("EPR pairs (aggregated)", str(plan.epr_pairs))
    table.add_row("EPR pairs (per gate)", str(plan.baseline_epr_pairs))
    table.add_row("saved", f"{plan.reduction * 100:.1f}%")
    table.add_row("blocks", str(len(plan.blocks)))
    table.add_row("port evictions", str(plan.evictions))
    table.add_row("peak cat copies per QPU", str(list(plan.peak_cat_copies)))
    table.add_row("e-bits (port-unconstrained)", str(report.ebits))
    table.add_row("distributable packets", str(report.active_packets))
    table.add_row("makespan (entanglement-aware)", f"{sched.makespan:.2f}")
    table.add_row("makespan (topology-aware)", f"{res.schedule.makespan:.2f}")
    table.add_row("unschedulable gates", str(sched.unschedulable_gates))
    console.print(table)

    if out:
        # The same rule as `compile-dist`: a manifest written for someone else to
        # act on is checked before it ships, not after they trust it.
        problems = audit_entanglement_schedule(
            sched, res.physical_circuit, MultiQPUArchitecture(cfg), latency, plan=plan
        )
        if problems:
            console.print("[bold red]Entanglement schedule is inconsistent:[/bold red]")
            for problem in problems[:10]:
                console.print(f"  {problem}")
            raise typer.Exit(code=1)

        Path(out).write_text(
            json.dumps(
                {
                    "aggregation": plan.to_dict(),
                    "ebits": report.to_dict(),
                    "schedule": sched.to_dict(),
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        _print_path(f"Wrote entanglement plan to {out}")

    if emit_qasm or verify:
        from .protocol import build_telegate_circuit, verify_telegate_equivalence

        arch = MultiQPUArchitecture(cfg)

    if emit_qasm:
        program = build_telegate_circuit(
            res.physical_circuit, arch, plan, coherent=False
        )
        Path(emit_qasm).write_text(qasm3.dumps(program.circuit), encoding="utf-8")
        _print_path(
            f"Wrote telegate circuit ({program.n_ancillas} protocol ancillas) "
            f"to {emit_qasm}"
        )

    if verify:
        try:
            equivalent = verify_telegate_equivalence(res.physical_circuit, arch, plan)
        except ValueError as exc:
            raise typer.BadParameter(f"Cannot verify this plan: {exc}") from exc
        if equivalent:
            console.print("[bold green]Verified:[/bold green] protocol matches circuit")
        else:
            console.print("[bold red]Verification failed[/bold red]")
            raise typer.Exit(code=1)


@app.command()
def optimal(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(12, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed for random circuit + transpiler"),
    strategy: str = typer.Option(
        "ebit",
        help="Partition strategy to score: balanced, cluster, ebit, tpccap, tpccap_sa",
    ),
    config: str | None = typer.Option(None, help="Path to config JSON/YAML"),
    input_qasm: str | None = typer.Option(
        None,
        "--input-qasm",
        help="Load an OpenQASM 2/3 circuit instead of generating one",
    ),
    max_nodes: int = typer.Option(
        DEFAULT_MAX_NODES,
        "--max-nodes",
        help="Branch-and-bound node budget; exhausting it reports a bound, not a proof",
    ),
    out: str | None = typer.Option(None, help="Optional JSON path for the gap report"),
) -> None:
    """Score a strategy's partition against the exact optimum.

    Solves the same instance exactly by branch and bound and reports how much
    the heuristic leaves on the table, under both the classical cut objective
    and the e-bit objective. The tree is over set partitions, so this is for
    calibration on small instances -- roughly a dozen qubits -- not compiling.
    """
    cfg = _load_config_or_default(config)
    qc = _load_or_random_circuit(
        input_qasm=input_qasm, n_logical=n_logical, depth=depth, seed=seed
    )

    res = compile_distributed(qc, cfg, seed=seed, strategy=strategy)
    weights = extract_twoq_weights(res.basis_circuit)
    capacity = cfg.capacity_per_qpu()

    table = Table(title=f"Optimality gap ({strategy}, {res.basis_circuit.num_qubits}q)")
    table.add_column("objective")
    table.add_column("heuristic", justify="right")
    table.add_column("optimal", justify="right")
    table.add_column("gap", justify="right")
    table.add_column("proved", justify="right")

    payload: dict[str, Any] = {"strategy": strategy, "n_qpus": cfg.n_qpus}
    objectives: tuple[tuple[Objective, dict[str, Any]], ...] = (
        ("cut", {"weights": weights}),
        ("ebits", {"packets": res.packets}),
    )
    for objective, kwargs in objectives:
        try:
            gap = partition_gap(
                res.partition,
                cfg.n_qpus,
                capacity,
                objective=objective,
                max_nodes=max_nodes,
                **kwargs,
            )
        except ValueError as exc:
            # A heuristic below a proved optimum is a bug in one of the two, not
            # a bad command line, so say which comparison failed and stop.
            console.print(f"[bold red]{objective}:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
        # Without a proof, `optimal` is only an upper bound on the true optimum,
        # so the computed gap is a lower bound on the real one -- and can even
        # come out negative. Render it as the bound it is rather than as a
        # number that reads like a measurement.
        if gap.proved_optimal:
            rendered = f"{gap.relative * 100:.1f}%"
        else:
            rendered = f">= {max(gap.relative, 0.0) * 100:.1f}%"
        table.add_row(
            objective,
            f"{gap.heuristic:g}",
            f"{gap.optimal:g}",
            rendered,
            "yes" if gap.proved_optimal else "no",
        )
        payload[objective] = {
            "heuristic": gap.heuristic,
            "optimal": gap.optimal,
            "absolute": gap.absolute,
            # `relative` is infinite when the optimum is zero and the heuristic
            # is not. JSON has no infinity, so that case is written as null
            # rather than as a finite number that would misreport it.
            "relative": gap.relative if math.isfinite(gap.relative) else None,
            "proved_optimal": gap.proved_optimal,
        }

    console.print(table)
    if not all(payload[key]["proved_optimal"] for key in ("cut", "ebits")):
        console.print(
            "[yellow]Node budget exhausted:[/yellow] the reported optimum is an "
            "upper bound, so the true gap is at least as large as shown."
        )

    if out:
        Path(out).write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        _print_path(f"Wrote gap report to {out}")


@app.command()
def compile_dist(
    n_logical: int | None = typer.Option(
        None, help="Number of logical qubits for generated random circuits"
    ),
    depth: int = typer.Option(20, help="Random circuit depth"),
    seed: int = typer.Option(0, help="Seed for random circuit + transpiler"),
    strategy: str = typer.Option(
        "tpccap_sa",
        help="Partition strategy: balanced, cluster, ebit, tpccap, tpccap_sa",
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
      - remote_ops.json     : ordered remote-op trace, in the routed programs'
        physical-qubit labelling
      - schedule.json       : topology-aware schedule summary
      - schedule_trace.json : detailed per-layer/per-round communication plan with absolute timing
      - entanglement_plan.json : aggregated EPR blocks, e-bit report, and the
        entanglement-aware schedule summary
    """
    cfg = _load_config_or_default(config)
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

    # The bundle ships routed programs, so it ships the manifest that matches
    # them: local routing permutes qubits inside a QPU unless the intra-QPU
    # topology is a clique, and the pre-routing indices would point elsewhere.
    write_remote_ops_json(res.routed_remote_ops, outp / "remote_ops.json")
    (outp / "schedule.json").write_text(
        json.dumps(res.schedule.to_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    # The trace is written for downstream consumers to schedule against, and a
    # consumer cannot tell a sound manifest from a self-consistent-looking wrong
    # one. Re-derive its numbers before shipping it, and refuse to ship a
    # manifest that does not add up.
    inconsistencies = audit_topology_schedule_plan(
        res.schedule_plan, MultiQPUArchitecture(cfg), latency, res.physical_circuit
    )
    if inconsistencies:
        console.print("[bold red]Schedule plan is inconsistent:[/bold red]")
        for problem in inconsistencies[:10]:
            console.print(f"  {problem}")
        if len(inconsistencies) > 10:
            console.print(f"  ... and {len(inconsistencies) - 10} more")
        raise typer.Exit(code=1)

    (outp / "schedule_trace.json").write_text(
        json.dumps(res.schedule_plan.to_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    entanglement_problems = audit_entanglement_schedule(
        res.entanglement_schedule,
        res.physical_circuit,
        MultiQPUArchitecture(cfg),
        latency,
        plan=res.aggregation,
    )
    if entanglement_problems:
        console.print("[bold red]Entanglement schedule is inconsistent:[/bold red]")
        for problem in entanglement_problems[:10]:
            console.print(f"  {problem}")
        raise typer.Exit(code=1)

    (outp / "entanglement_plan.json").write_text(
        json.dumps(
            {
                "aggregation": res.aggregation.to_dict(),
                "ebits": res.ebits.to_dict(),
                "schedule": res.entanglement_schedule.to_dict(),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    swaps_total = sum(m.get("swap", 0) for m in res.local_metrics.values())
    console.print(
        f"[bold]Remote2Q:[/bold] {res.global_metrics.remote_2q}  [bold]Local SWAPs:[/bold] {swaps_total}"
    )
    console.print(
        f"[bold]EPR pairs:[/bold] {res.aggregation.epr_pairs} "
        f"(un-aggregated {res.aggregation.baseline_epr_pairs}, "
        f"saved {res.aggregation.reduction * 100:.1f}%)  "
        f"[bold]Blocks:[/bold] {len(res.aggregation.blocks)}"
    )
    console.print(
        f"[bold]Makespan (topology-aware):[/bold] {res.schedule.makespan:.2f}  [bold]Remote rounds:[/bold] {res.schedule.remote_rounds}"
    )
    console.print(
        f"[bold]Makespan (entanglement-aware):[/bold] {res.entanglement_schedule.makespan:.2f}"
    )
    console.print(
        f"[bold]Times:[/bold] mapping={res.mapping_time_s:.4f}s  local_transpile={res.local_transpile_time_s:.4f}s"
    )
    _print_path(f"Wrote artifacts to {out_dir}")
