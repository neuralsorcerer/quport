# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any

import pytest

from quport.config import MultiQPUConfig


@pytest.mark.parametrize(
    ("field", "value", "msg"),
    [
        ("async_overlap", -0.1, "async_overlap must be non-negative"),
        ("async_overlap", 1.1, r"async_overlap must be in \[0, 1\]"),
        ("optimization_level", 4, "optimization_level must be between 0 and 3"),
        ("layout_method", "   ", "layout_method must be a non-empty string"),
        ("routing_method", "", "routing_method must be a non-empty string"),
    ],
)
def test_rejects_invalid_fields(field: str, value: object, msg: str) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=msg):
        MultiQPUConfig(**kwargs)


def test_rejects_non_boolean_async_classical() -> None:
    with pytest.raises(ValueError, match="async_classical must be a boolean"):
        MultiQPUConfig(async_classical=1)  # type: ignore[arg-type]


def test_accepts_and_normalizes_config_file_basis_gate_lists() -> None:
    cfg = MultiQPUConfig(basis_gates=[" rz ", "sx", "x", "cx"])

    assert cfg.basis_gates == ("rz", "sx", "x", "cx")


def test_rejects_string_basis_gates() -> None:
    with pytest.raises(ValueError, match="basis_gates must be a sequence"):
        MultiQPUConfig(basis_gates="cx")  # type: ignore[arg-type]


def test_rejects_empty_basis_gates() -> None:
    with pytest.raises(ValueError, match="basis_gates must contain at least one"):
        MultiQPUConfig(basis_gates=())


def test_rejects_empty_gate_name() -> None:
    with pytest.raises(
        ValueError, match="basis_gates entries must be a non-empty string"
    ):
        MultiQPUConfig(basis_gates=("cx", "   "))


def test_dumped_json_config_round_trips_basis_gates(tmp_path: Path) -> None:
    from quport.config import dump_config, load_config

    path = tmp_path / "quport_config.json"
    cfg = MultiQPUConfig(basis_gates=("rz", "sx", "x", "cx"))

    dump_config(cfg, str(path))
    loaded = load_config(str(path))

    assert loaded.basis_gates == cfg.basis_gates


def test_load_config_rejects_non_string_keys(tmp_path: Path) -> None:
    """YAML permits non-string mapping keys; JSON does not.

    Without this check the key reaches ``MultiQPUConfig(**data)`` and surfaces as
    a ``TypeError`` about keyword arguments instead of naming the offending key.
    """
    pytest.importorskip("yaml")
    from quport.config import load_config

    config = tmp_path / "cfg.yaml"
    config.write_text("1: 2\nn_qpus: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-string key"):
        load_config(str(config))


def test_load_config_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    from quport.config import load_config

    config = tmp_path / "cfg.json"
    config.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping/object"):
        load_config(str(config))


def test_yaml_config_round_trips_through_dump_and_load(tmp_path: Path) -> None:
    """The YAML branches of `dump_config`/`load_config` had no round-trip test.

    Only the JSON path was exercised, so a broken YAML writer would have shipped
    behind the optional extra.
    """
    pytest.importorskip("yaml")
    from quport.config import MultiQPUConfig, dump_config, load_config

    original = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=4,
        comm_qubits_per_qpu=2,
        intra_topology="grid2d",
        inter_topology="clos",
        optimization_level=2,
    )
    path = tmp_path / "cfg.yaml"

    dump_config(original, str(path))
    assert path.exists()

    # YAML is a superset of JSON, so round-tripping alone cannot tell a YAML
    # writer from a JSON one: assert the file is written in YAML block style.
    written = path.read_text(encoding="utf-8")
    assert "n_qpus: 3" in written
    assert not written.lstrip().startswith("{")

    assert load_config(str(path)) == original
