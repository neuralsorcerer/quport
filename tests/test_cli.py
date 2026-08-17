# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import pytest
import typer

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit, qasm2
from qiskit.exceptions import MissingOptionalLibraryError
from typer.testing import CliRunner

from quport.cli import _load_or_random_circuit, _qasm_version, app


def _write_small_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "n_qpus": 2,
                "compute_qubits_per_qpu": 2,
                "comm_qubits_per_qpu": 1,
                "intra_topology": "clique",
                "inter_topology": "switch",
                "optimization_level": 0,
            }
        ),
        encoding="utf-8",
    )


def _write_qasm(path: Path, *, leading_text: str = "") -> None:
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    path.write_text(leading_text + qasm2.dumps(circuit), encoding="utf-8")


@pytest.mark.parametrize(
    ("source", "version"),
    [
        ("OPENQASM 2.0;\n", 2),
        ("\ufeff  // generated file\nOPENQASM 2.0;\n", 2),
        ("/* block comment */\nOPENQASM 3.0;\n", 3),
        ("/* unterminated block comment", None),
        ("qreg q[1];\n", None),
        ("// trailing line comment with no newline", None),
        ("// one\n// two\nOPENQASM 3;\n", 3),
        ("OPENQASM 4.0;\n", None),
    ],
)
def test_qasm_version_detects_headers_after_leading_comments(
    source: str, version: int | None
) -> None:
    assert _qasm_version(source) == version


def test_input_qasm_loader_accepts_qasm2_with_leading_comments(tmp_path: Path) -> None:
    input_path = tmp_path / "commented.qasm"
    _write_qasm(input_path, leading_text="// created by external tool\n")

    circuit = _load_or_random_circuit(
        input_qasm=str(input_path),
        n_logical=None,
        depth=0,
        seed=0,
    )

    assert circuit.num_qubits == 3


def test_input_qasm_loader_reports_missing_qasm3_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.qasm"
    input_path.write_text("OPENQASM 3.0;\nqubit[1] q;\n", encoding="utf-8")

    def _missing_importer(path: str) -> QuantumCircuit:
        raise MissingOptionalLibraryError(
            libname="qiskit_qasm3_import",
            name="loading from OpenQASM 3",
            pip_install="pip install qiskit_qasm3_import",
        )

    monkeypatch.setattr("quport.cli.qasm3.load", _missing_importer)

    with pytest.raises(typer.BadParameter, match="qiskit_qasm3_import"):
        _load_or_random_circuit(
            input_qasm=str(input_path),
            n_logical=None,
            depth=0,
            seed=0,
        )


def test_input_qasm_loader_reports_unreadable_file() -> None:
    with pytest.raises(typer.BadParameter, match="Unable to read --input-qasm"):
        _load_or_random_circuit(
            input_qasm="/definitely/missing/input.qasm",
            n_logical=None,
            depth=0,
            seed=0,
        )


def test_map_accepts_input_qasm_without_n_logical(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.json"
    input_path = tmp_path / "input.qasm"
    output_path = tmp_path / "mapped.qasm"
    _write_small_config(config_path)
    _write_qasm(input_path)

    result = runner.invoke(
        app,
        [
            "map",
            "--input-qasm",
            str(input_path),
            "--config",
            str(config_path),
            "--strategy",
            "balanced",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "SWAPs" in result.output


def test_compile_dist_accepts_input_qasm_without_n_logical(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.json"
    input_path = tmp_path / "input.qasm"
    out_dir = tmp_path / "compile"
    _write_small_config(config_path)
    _write_qasm(input_path)

    result = runner.invoke(
        app,
        [
            "compile-dist",
            "--input-qasm",
            str(input_path),
            "--config",
            str(config_path),
            "--strategy",
            "balanced",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "remote_ops.json").exists()
    schedule_path = out_dir / "schedule.json"
    assert schedule_path.exists()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert set(schedule) == {
        "makespan",
        "layers",
        "remote_ops",
        "remote_rounds",
        "peak_link_util",
        "peak_qpu_ports_used",
    }
    trace_path = out_dir / "schedule_trace.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["summary"] == schedule
    assert all(
        "start_time" in layer and "end_time" in layer for layer in trace["layers"]
    )
    assert all(
        "start_time" in round_ and "end_time" in round_
        for layer in trace["layers"]
        for round_ in layer["remote_rounds"]
    )


def test_circuit_commands_require_n_logical_without_input_qasm() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["map", "--depth", "0"])

    assert result.exit_code != 0
    assert "--n-logical is required" in result.output


def test_topology_info_command_outputs_metrics() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["topology-info"])

    assert result.exit_code == 0
    assert "Inter-QPU Topology Metrics" in result.stdout
    assert "average_shortest_path" in result.stdout


def _run(args: list[str]) -> "object":
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, f"{args} failed: {result.output}\n{result.exception}"
    return result


def test_gen_config_writes_a_loadable_config(tmp_path: Path) -> None:
    from quport.config import MultiQPUConfig, load_config

    out = tmp_path / "cfg.json"
    _run(["gen-config", "--out", str(out)])

    assert load_config(str(out)) == MultiQPUConfig()


def test_written_paths_are_reported_on_a_single_line(tmp_path: Path) -> None:
    """Rich wraps at 80 columns when stdout is not a terminal, which used to
    split long output paths across lines and break copy/paste.

    Deep temporary directories cross that width on macOS and Windows CI but not
    on Linux, so build a path that is long on every platform instead of
    trusting `tmp_path` to be long enough.
    """
    nested = tmp_path / ("long_output_directory_" * 3)
    nested.mkdir()
    out = nested / "cfg.json"
    assert len(str(out)) > 80  # guards the premise, not the behaviour

    result = _run(["gen-config", "--out", str(out)])

    assert f"Wrote config to {out}" in result.output  # type: ignore[attr-defined]


def test_bench_writes_one_csv_row_per_trial_and_strategy(tmp_path: Path) -> None:
    import csv

    config = tmp_path / "cfg.json"
    _write_small_config(config)
    out = tmp_path / "results.csv"

    _run(
        [
            "bench",
            "--n-logical",
            "4",
            "--depth",
            "2",
            "--trials",
            "2",
            "--strategies",
            "balanced,cluster",
            "--config",
            str(config),
            "--out",
            str(out),
        ]
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 4
    assert {row["strategy"] for row in rows} == {"balanced", "cluster"}
    assert {row["seed"] for row in rows} == {"0.0", "1.0"}
    for row in rows:
        assert float(row["cost_total"]) >= 0.0


def test_schedule_reports_makespan(tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    _write_small_config(config)

    result = _run(
        ["schedule", "--n-logical", "4", "--depth", "2", "--config", str(config)]
    )

    assert "Makespan:" in result.output  # type: ignore[attr-defined]
    assert "RemoteOps:" in result.output  # type: ignore[attr-defined]


def test_split_writes_one_qasm_per_qpu_plus_remote_ops(tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    _write_small_config(config)
    out_dir = tmp_path / "split_out"

    _run(
        [
            "split",
            "--n-logical",
            "4",
            "--depth",
            "2",
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert {path.name for path in out_dir.iterdir()} == {
        "qpu_0.qasm",
        "qpu_1.qasm",
        "remote_ops.json",
    }
    assert isinstance(json.loads((out_dir / "remote_ops.json").read_text()), list)


def test_compile_dist_writes_routed_programs_and_schedule_manifests(
    tmp_path: Path,
) -> None:
    config = tmp_path / "cfg.json"
    _write_small_config(config)
    out_dir = tmp_path / "compile_out"

    _run(
        [
            "compile-dist",
            "--n-logical",
            "4",
            "--depth",
            "2",
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert {path.name for path in out_dir.iterdir()} == {
        "qpu_0_routed.qasm",
        "qpu_1_routed.qasm",
        "remote_ops.json",
        "schedule.json",
        "schedule_trace.json",
        "entanglement_plan.json",
    }

    summary = json.loads((out_dir / "schedule.json").read_text(encoding="utf-8"))
    trace = json.loads((out_dir / "schedule_trace.json").read_text(encoding="utf-8"))
    assert trace["summary"] == summary
    assert len(trace["layers"]) == summary["layers"]
    assert summary["remote_ops"] == sum(
        layer["remote_ops"] for layer in trace["layers"]
    )

    remote_ops = json.loads((out_dir / "remote_ops.json").read_text(encoding="utf-8"))
    for op in remote_ops:
        assert op["qpu0_marker"] is not None
        assert op["qpu1_marker"] is not None

    entanglement = json.loads(
        (out_dir / "entanglement_plan.json").read_text(encoding="utf-8")
    )
    aggregation = entanglement["aggregation"]
    assert aggregation["epr_pairs"] == sum(
        block["epr_pairs"] for block in aggregation["blocks"]
    )
    assert aggregation["epr_pairs"] <= aggregation["baseline_epr_pairs"]
    assert entanglement["schedule"]["epr_pairs"] == aggregation["epr_pairs"]


def test_sweep_writes_summary_csv(tmp_path: Path) -> None:
    import csv

    out = tmp_path / "sweep.csv"

    _run(
        [
            "sweep",
            "--n-logical",
            "3",
            "--depth",
            "1",
            "--trials",
            "1",
            "--strategies",
            "balanced",
            "--out",
            str(out),
        ]
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows
    assert {row["intra"] for row in rows} <= {"clique", "line", "ring"}
    assert {row["inter"] for row in rows} <= {"switch", "ring", "degree_d", "clos"}
    assert all(float(row["ports"]) in (1.0, 2.0) for row in rows)


@pytest.mark.parametrize("source", ["OPENQASM 1.0;", "OPENQASM 4.0;", "OPENQASM 23;"])
def test_qasm_version_returns_none_for_unsupported_versions(source: str) -> None:
    """Only OpenQASM 2 and 3 are recognised; anything else falls back to sniffing."""
    assert _qasm_version(source) is None


def test_bench_accepts_strategies_with_surrounding_whitespace(tmp_path: Path) -> None:
    """`--strategies` is a human-typed list, so spaces after commas must work."""
    import csv

    config = tmp_path / "cfg.json"
    _write_small_config(config)
    out = tmp_path / "results.csv"

    _run(
        [
            "bench",
            "--n-logical",
            "4",
            "--depth",
            "2",
            "--trials",
            "1",
            "--strategies",
            " balanced , cluster ",
            "--config",
            str(config),
            "--out",
            str(out),
        ]
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert {row["strategy"] for row in rows} == {"balanced", "cluster"}


def test_gen_config_default_output_needs_no_optional_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`quport gen-config` with no arguments must work on a base install.

    YAML support is an optional extra, so the default output cannot be a YAML
    path or the first documented command fails for anyone who installed the
    package without extras.
    """
    from quport.config import MultiQPUConfig, load_config

    monkeypatch.chdir(tmp_path)
    _run(["gen-config"])

    written = tmp_path / "quport_config.json"
    assert written.is_file()
    assert load_config(str(written)) == MultiQPUConfig()


def test_gen_config_reports_missing_yaml_extra_as_a_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing optional dependency is a usage error, not a traceback."""
    import quport.config

    monkeypatch.setattr(
        quport.config, "optional_module_available", lambda module_name: False
    )
    result = CliRunner().invoke(
        app, ["gen-config", "--out", str(tmp_path / "cfg.yaml")]
    )

    assert result.exit_code != 0
    assert "quport[yaml]" in result.output
    assert not isinstance(result.exception, RuntimeError)


def test_commands_report_missing_yaml_extra_when_loading_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quport.config

    config = tmp_path / "cfg.yaml"
    config.write_text("n_qpus: 2\n", encoding="utf-8")
    monkeypatch.setattr(
        quport.config, "optional_module_available", lambda module_name: False
    )

    result = CliRunner().invoke(app, ["topology-info", "--config", str(config)])

    assert result.exit_code != 0
    assert "quport[yaml]" in result.output
    assert not isinstance(result.exception, RuntimeError)


def test_sweep_writes_a_plot_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `--plot` branch of `sweep` was never executed, even with viz installed.

    A blank figure is still a valid PNG, so checking the file alone would pass
    even if no series were drawn. Record the scatter calls instead.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from quport.pipeline import benchmark_method_labels

    series: list[str] = []
    real_scatter = plt.scatter

    def recording_scatter(*args: object, **kwargs: object) -> object:
        series.append(str(kwargs.get("label")))
        return real_scatter(*args, **kwargs)

    monkeypatch.setattr(plt, "scatter", recording_scatter)

    out_csv = tmp_path / "sweep.csv"
    out_png = tmp_path / "sweep.png"

    result = CliRunner().invoke(
        app,
        [
            "sweep",
            "--n-logical",
            "2",
            "--depth",
            "1",
            "--trials",
            "1",
            "--strategies",
            "baseline,tpccap",
            "--out",
            str(out_csv),
            "--plot",
            str(out_png),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_csv.is_file()
    assert out_png.is_file()
    assert out_png.read_bytes().startswith(b"\x89PNG")
    assert str(out_png) in result.output

    # one labelled series per strategy, named -- not left as a bare method id
    assert sorted(series) == ["baseline", "tpccap"]
    assert set(series) <= set(benchmark_method_labels().values())


def test_sweep_reports_missing_viz_extra_as_a_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing plotting extra is a usage error, not a traceback."""
    import quport.cli

    monkeypatch.setattr(
        quport.cli, "optional_module_available", lambda module_name: False
    )

    result = CliRunner().invoke(
        app,
        [
            "sweep",
            "--n-logical",
            "2",
            "--depth",
            "1",
            "--trials",
            "1",
            "--out",
            str(tmp_path / "sweep.csv"),
            "--plot",
            str(tmp_path / "sweep.png"),
        ],
    )

    assert result.exit_code != 0
    assert "[viz]" in result.output
    assert not isinstance(result.exception, ImportError)
    assert not (tmp_path / "sweep.png").exists()


def test_module_entry_point_runs_the_cli() -> None:
    """`python -m quport` is a shipped entry point that nothing exercised.

    A subprocess does not inherit the `src`-first `sys.path` that conftest sets
    up, so point it at the working tree explicitly. Otherwise this silently
    tests whatever copy of quport happens to be installed.
    """
    import os
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    )

    result = subprocess.run(
        [sys.executable, "-m", "quport", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "QuPort" in result.stdout
    assert "sweep" in result.stdout


def _write_malformed_qasm(path: Path) -> None:
    path.write_text("this is not a QASM program at all\n", encoding="utf-8")


def test_input_qasm_loader_reports_a_bad_qasm2_body(tmp_path: Path) -> None:
    """A declared OpenQASM 2 header with an unparsable body names the version."""
    input_path = tmp_path / "broken2.qasm"
    input_path.write_text("OPENQASM 2.0;\nnot_a_gate q[0];\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="Unable to parse OpenQASM 2"):
        _load_or_random_circuit(
            input_qasm=str(input_path),
            n_logical=None,
            depth=1,
            seed=0,
        )


def test_input_qasm_loader_reports_a_bad_qasm3_body(tmp_path: Path) -> None:
    """A declared OpenQASM 3 header with an unparsable body names the version.

    The parse failure must not be mistaken for the missing-importer case, which
    is reported with an install hint instead.
    """
    input_path = tmp_path / "broken3.qasm"
    input_path.write_text("OPENQASM 3.0;\nnot_a_gate q[0];\n", encoding="utf-8")

    def _fails(*_args: object, **_kwargs: object) -> QuantumCircuit:
        raise ValueError("bad qasm3 body")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("quport.cli.qasm3.load", _fails)
        with pytest.raises(typer.BadParameter, match="Unable to parse OpenQASM 3"):
            _load_or_random_circuit(
                input_qasm=str(input_path),
                n_logical=None,
                depth=1,
                seed=0,
            )


def test_headerless_input_falls_back_from_qasm3_to_qasm2(tmp_path: Path) -> None:
    """Without a version header, a QASM 3 failure must still try QASM 2."""
    input_path = tmp_path / "headerless.qasm"
    _write_qasm(input_path)
    body = input_path.read_text(encoding="utf-8")
    input_path.write_text(
        "\n".join(line for line in body.splitlines() if not line.startswith("OPENQASM"))
        + "\n",
        encoding="utf-8",
    )
    assert _qasm_version(input_path.read_text(encoding="utf-8")) is None

    def _fails(*_args: object, **_kwargs: object) -> QuantumCircuit:
        raise ValueError("no qasm3 here")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("quport.cli.qasm3.load", _fails)
        circuit = _load_or_random_circuit(
            input_qasm=str(input_path),
            n_logical=None,
            depth=1,
            seed=0,
        )

    assert circuit.num_qubits > 0


def test_headerless_input_falls_back_when_the_qasm3_importer_is_missing(
    tmp_path: Path,
) -> None:
    """The missing-importer branch of the headerless ladder also retries QASM 2."""
    input_path = tmp_path / "headerless.qasm"
    _write_qasm(input_path)
    body = input_path.read_text(encoding="utf-8")
    input_path.write_text(
        "\n".join(line for line in body.splitlines() if not line.startswith("OPENQASM"))
        + "\n",
        encoding="utf-8",
    )

    def _missing_importer(*_args: object, **_kwargs: object) -> QuantumCircuit:
        raise MissingOptionalLibraryError(
            libname="qiskit_qasm3_import",
            name="OpenQASM 3 importer",
            pip_install="pip install qiskit_qasm3_import",
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("quport.cli.qasm3.load", _missing_importer)
        circuit = _load_or_random_circuit(
            input_qasm=str(input_path),
            n_logical=None,
            depth=1,
            seed=0,
        )

    assert circuit.num_qubits > 0


@pytest.mark.parametrize("qasm3_error", ["missing_importer", "parse_error"])
def test_headerless_input_reports_both_parser_failures(
    tmp_path: Path, qasm3_error: str
) -> None:
    """When neither parser can read a headerless file, say so for both."""
    input_path = tmp_path / "garbage.qasm"
    _write_malformed_qasm(input_path)

    def _missing_importer(*_args: object, **_kwargs: object) -> QuantumCircuit:
        raise MissingOptionalLibraryError(
            libname="qiskit_qasm3_import",
            name="OpenQASM 3 importer",
            pip_install="pip install qiskit_qasm3_import",
        )

    def _parse_error(*_args: object, **_kwargs: object) -> QuantumCircuit:
        raise ValueError("no qasm3 here")

    expected = (
        "could not be parsed as OpenQASM 2"
        if qasm3_error == "missing_importer"
        else "Parsing failed as OpenQASM 3"
    )
    failure = _missing_importer if qasm3_error == "missing_importer" else _parse_error

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("quport.cli.qasm3.load", failure)
        with pytest.raises(typer.BadParameter, match=expected):
            _load_or_random_circuit(
                input_qasm=str(input_path),
                n_logical=None,
                depth=1,
                seed=0,
            )


def test_compile_dist_bundle_is_consumable_from_the_qasm_alone(tmp_path: Path) -> None:
    """The shipped manifest and QASM files must agree qubit for qubit.

    A consumer only has the emitted text: barriers carry no labels there, so the
    manifest's marker index is the only thing that says which barrier belongs to
    which remote operation. A ``line`` intra-topology makes both hazards real --
    routing permutes qubits inside a QPU, and it reorders barriers that sit on
    disjoint qubits -- so pairing by position or trusting pre-routing indices
    would both fail here.
    """
    import re

    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "n_qpus": 2,
                "compute_qubits_per_qpu": 4,
                "comm_qubits_per_qpu": 1,
                "intra_topology": "line",
                "optimization_level": 1,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bundle"

    _run(
        [
            "compile-dist",
            "--n-logical",
            "9",
            "--depth",
            "14",
            "--seed",
            "4",
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
        ]
    )

    remote_ops = json.loads((out_dir / "remote_ops.json").read_text(encoding="utf-8"))
    assert remote_ops, "the fixture must produce remote operations"

    barriers = {
        qpu: [
            int(match)
            for match in re.findall(
                r"barrier \$(\d+);",
                (out_dir / f"qpu_{qpu}_routed.qasm").read_text(encoding="utf-8"),
            )
        ]
        for qpu in (0, 1)
    }

    for op in remote_ops:
        assert barriers[op["qpu0"]][op["qpu0_marker"]] == op["q0_phys"]
        assert barriers[op["qpu1"]][op["qpu1_marker"]] == op["q1_phys"]

    # The markers genuinely disagree with positional pairing, so the assertions
    # above are not passing by accident.
    assert any(op["qpu0_marker"] != index for index, op in enumerate(remote_ops))
