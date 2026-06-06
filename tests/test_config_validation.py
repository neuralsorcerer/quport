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
