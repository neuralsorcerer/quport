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


@pytest.mark.parametrize(
    ("name", "text"),
    [("cfg.json", '{"n_qpus": 3}'), ("cfg.yaml", "n_qpus: 3\n")],
)
def test_load_config_reads_a_file_saved_with_a_byte_order_mark(
    tmp_path: Path, name: str, text: str
) -> None:
    """Editors on Windows prepend a BOM, and the JSON parser refuses one.

    PyYAML strips it already, so before this the same config was readable saved
    as YAML and unreadable saved as JSON. The CLI's ``--input-qasm`` has always
    tolerated a BOM; ``--config`` feeds every subcommand and now matches it.
    """
    from quport.config import load_config

    if name.endswith(".yaml"):
        pytest.importorskip("yaml")
    config = tmp_path / name
    config.write_text(text, encoding="utf-8-sig")
    assert config.read_bytes().startswith(b"\xef\xbb\xbf")

    assert load_config(str(config)).n_qpus == 3


@pytest.mark.parametrize("suffix", [".YAML", ".YML", ".Yaml"])
def test_yaml_is_parsed_as_yaml_whatever_the_suffix_case(
    tmp_path: Path, suffix: str
) -> None:
    """``config.YAML`` is an ordinary name on macOS and Windows.

    An uppercase YAML suffix used to fall through to the JSON parser, which
    reported a decode error against a file that was perfectly valid. A
    round-trip check cannot see this, because writing picked the same wrong
    format as reading; the file has to be written as YAML by hand.
    """
    pytest.importorskip("yaml")
    from quport.config import load_config

    path = tmp_path / f"cfg{suffix}"
    path.write_text("n_qpus: 5\n", encoding="utf-8")

    assert load_config(str(path)).n_qpus == 5


@pytest.mark.parametrize("suffix", [".YAML", ".YML", ".yaml", ".JSON", ".json"])
def test_config_round_trips_whatever_the_suffix_case(
    tmp_path: Path, suffix: str
) -> None:
    """Saving and loading must agree on the format for a given name."""
    if suffix.lower() in {".yaml", ".yml"}:
        pytest.importorskip("yaml")
    from quport.config import dump_config, load_config

    path = tmp_path / f"cfg{suffix}"
    cfg = MultiQPUConfig(n_qpus=5)

    dump_config(cfg, str(path))

    assert load_config(str(path)) == cfg


def test_dump_config_writes_numpy_and_int_subclass_values(tmp_path: Path) -> None:
    """A config the validators accept must save and load back in both formats."""
    import numpy as np

    from quport.config import dump_config, load_config

    class FancyInt(int):
        pass

    cfg = MultiQPUConfig(
        n_qpus=FancyInt(3),
        compute_qubits_per_qpu=2,
        optimization_level=np.arange(4)[2],
        async_overlap=np.float64(0.25),
    )
    expected = MultiQPUConfig(
        n_qpus=3, compute_qubits_per_qpu=2, optimization_level=2, async_overlap=0.25
    )
    suffixes = [".json"]
    try:
        import yaml  # noqa: F401

        suffixes.append(".yaml")
    except ImportError:  # pragma: no cover - yaml is a test extra
        pass
    for suffix in suffixes:
        path = tmp_path / f"cfg{suffix}"
        dump_config(cfg, str(path))
        assert load_config(str(path)) == expected


def test_a_failed_dump_leaves_the_existing_config_untouched(tmp_path: Path) -> None:
    from quport.config import dump_config, load_config

    path = tmp_path / "cfg.json"
    dump_config(MultiQPUConfig(), str(path))
    before = path.read_text(encoding="utf-8")

    unserializable = MultiQPUConfig(grid_rows=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dump_config(unserializable, str(path))

    assert path.read_text(encoding="utf-8") == before
    assert load_config(str(path)) == MultiQPUConfig()


@pytest.mark.parametrize(
    "call",
    [
        lambda np: MultiQPUConfig(async_overlap=np.True_),
        lambda np: __import__(
            "quport.config", fromlist=["validate_epr_success_prob"]
        ).validate_epr_success_prob(np.True_),
        lambda np: __import__(
            "quport.network", fromlist=["compute_traffic_matrix"]
        ).compute_traffic_matrix({(0, 1): np.True_}, [0, 1], 2),
        lambda np: __import__(
            "quport.network", fromlist=["congestion_metrics"]
        ).congestion_metrics({(0, 1): np.True_}),
        lambda np: __import__(
            "quport.partition", fromlist=["balanced_greedy_partition"]
        ).balanced_greedy_partition(2, {(0, 1): np.True_}, 2, 1),
    ],
    ids=["async_overlap", "epr_success_prob", "traffic", "link-loads", "weights"],
)
def test_numpy_booleans_are_rejected_like_python_booleans(call: Any) -> None:
    """``type(x) is bool`` misses ``numpy.bool_``, which ``float()`` accepts."""
    import numpy as np

    with pytest.raises(ValueError, match="boolean"):
        call(np)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_dump_config_writes_decimal_values_as_numbers(tmp_path, suffix: str) -> None:
    """Validators read a Decimal through float(); the encoders refused it."""
    from decimal import Decimal

    from quport.config import dump_config, load_config

    if suffix == ".yaml":
        pytest.importorskip("yaml")
    cfg = MultiQPUConfig(
        async_overlap=Decimal("0.5"),  # type: ignore[arg-type]
        switch_reconfig_delay=Decimal("2.5"),  # type: ignore[arg-type]
    )
    path = tmp_path / f"cfg{suffix}"
    dump_config(cfg, str(path))

    loaded = load_config(str(path))
    assert loaded.async_overlap == 0.5
    assert loaded.switch_reconfig_delay == 2.5


def test_transpiler_method_names_are_stored_stripped() -> None:
    """Basis gate names were stripped but plugin names were not, so Qiskit later
    refused 'sabre ' -- after partitioning."""
    cfg = MultiQPUConfig(layout_method=" dense", routing_method="sabre ")

    assert (cfg.layout_method, cfg.routing_method) == ("dense", "sabre")
