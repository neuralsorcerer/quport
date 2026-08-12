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
    }

    summary = json.loads((out_dir / "schedule.json").read_text(encoding="utf-8"))
    trace = json.loads((out_dir / "schedule_trace.json").read_text(encoding="utf-8"))
    assert trace["summary"] == summary
    assert len(trace["layers"]) == summary["layers"]
    assert summary["remote_ops"] == sum(
        layer["remote_ops"] for layer in trace["layers"]
    )


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
