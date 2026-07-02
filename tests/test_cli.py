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
    assert (out_dir / "schedule.json").exists()
    trace_path = out_dir / "schedule_trace.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
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
