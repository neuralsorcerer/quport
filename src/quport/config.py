# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from quport._validation import (
    validate_finite_result,
    validate_nonnegative_finite_float,
    validate_nonnegative_integral,
)


def optional_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _load_yaml_module() -> Any:
    if not optional_module_available("yaml"):
        raise RuntimeError(
            "PyYAML not installed. Install with: pip install quport[yaml]"
        )
    return importlib.import_module("yaml")


IntraTopology = Literal["clique", "line", "ring", "grid2d"]
InterTopology = Literal["switch", "mesh", "ring", "degree_d", "clos", "fat_tree"]


def _validate_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    out = value.strip()
    if not out:
        raise ValueError(f"{label} must be a non-empty string")
    return out


@dataclass(frozen=True)
class MultiQPUConfig:
    """Configuration for the multi-QPU architecture and mapping pipeline."""

    n_qpus: int = 10
    compute_qubits_per_qpu: int = 8
    comm_qubits_per_qpu: int = 1

    intra_topology: IntraTopology = "clique"
    inter_topology: InterTopology = "switch"

    # Used only when inter_topology == "degree_d"
    inter_degree: int = 2
    # Network resource limits (used by topology-aware schedulers)
    link_capacity: int = 1  # max simultaneous remote ops per inter-QPU *link* per round
    switch_parallel_links: int = (
        1_000_000  # for inter_topology="switch"/"mesh": max distinct pairs per round
    )
    switch_reconfig_delay: float = (
        0.0  # additional delay per communication round (models optical switch reconfig)
    )

    # Classical-latency hiding (asynchronous telegate/teledata models)
    async_classical: bool = True
    async_overlap: float = 0.5  # fraction of classical_rtt that can be hidden (0..1)

    # Grid dimensions when intra_topology == "grid2d"
    grid_rows: int | None = None
    grid_cols: int | None = None

    # Transpiler settings
    basis_gates: tuple[str, ...] = ("rz", "sx", "x", "cx")
    optimization_level: int = 3
    layout_method: str = "sabre"
    routing_method: str = "sabre"

    def __post_init__(self) -> None:
        if not isinstance(self.async_classical, bool):
            raise ValueError("async_classical must be a boolean")
        async_overlap = validate_nonnegative_finite_float(
            self.async_overlap, label="async_overlap"
        )
        if async_overlap > 1.0:
            raise ValueError("async_overlap must be in [0, 1]")

        if not isinstance(self.basis_gates, tuple):
            raise ValueError("basis_gates must be a tuple of gate names")
        for gate in self.basis_gates:
            _validate_nonempty_string(gate, label="basis_gates entries")

        optimization_level = validate_nonnegative_integral(
            self.optimization_level, label="optimization_level"
        )
        if optimization_level > 3:
            raise ValueError("optimization_level must be between 0 and 3")

        _validate_nonempty_string(self.layout_method, label="layout_method")
        _validate_nonempty_string(self.routing_method, label="routing_method")

    def total_physical_qubits(self) -> int:
        return self.n_qpus * (self.compute_qubits_per_qpu + self.comm_qubits_per_qpu)

    def capacity_per_qpu(self) -> int:
        return self.compute_qubits_per_qpu + self.comm_qubits_per_qpu


@dataclass(frozen=True)
class LatencyModel:
    """A simple, extensible latency/cost model for multi-QPU execution.

    This is *not* a hardware-accurate model; it's a research knob to compare mappings.

    Parameters represent average times (arbitrary units). You can interpret them as:
    - nanoseconds for superconducting devices
    - microseconds for ion traps
    - normalized cost units

    Notes:
    - `remote_2q` is a proxy for distributed operations that require entanglement generation,
      teleportation, or remote-gate protocols.
    - `swap` is the transpiler-inserted SWAP count (local routing overhead).
    """

    oneq: float = 1.0
    twoq: float = 10.0
    swap: float = 30.0

    # Entanglement / networking overheads
    epr_gen: float = 200.0
    classical_rtt: float = 20.0
    remote_gate_overhead: float = 50.0

    def estimate_latency(
        self, n_1q: int, n_2q: int, swaps: int, remote_2q: int, depth: int | None = None
    ) -> float:
        n_1q_value = validate_nonnegative_integral(n_1q, label="n_1q")
        n_2q_value = validate_nonnegative_integral(n_2q, label="n_2q")
        swaps_value = validate_nonnegative_integral(swaps, label="swaps")
        remote_2q_value = validate_nonnegative_integral(remote_2q, label="remote_2q")

        oneq = validate_nonnegative_finite_float(self.oneq, label="oneq")
        twoq = validate_nonnegative_finite_float(self.twoq, label="twoq")
        swap = validate_nonnegative_finite_float(self.swap, label="swap")
        epr_gen = validate_nonnegative_finite_float(self.epr_gen, label="epr_gen")
        classical_rtt = validate_nonnegative_finite_float(
            self.classical_rtt, label="classical_rtt"
        )
        remote_gate_overhead = validate_nonnegative_finite_float(
            self.remote_gate_overhead, label="remote_gate_overhead"
        )

        base = oneq * n_1q_value + twoq * n_2q_value + swap * swaps_value
        remote = remote_2q_value * (epr_gen + classical_rtt + remote_gate_overhead)
        # Optionally include depth as a soft penalty
        if depth is not None:
            depth_value = validate_nonnegative_integral(depth, label="depth")
            base += 0.1 * depth_value * twoq
        return validate_finite_result(base + remote, label="estimated latency")


def _validate_config_data(data: Any, path: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path!r} must contain a mapping/object")

    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Config file {path!r} contains a non-string key: {key!r}")
        out[key] = value

    valid_fields = {field.name for field in fields(MultiQPUConfig)}
    unknown = sorted(set(out) - valid_fields)
    if unknown:
        unknown_list = ", ".join(unknown)
        raise ValueError(
            f"Config file {path!r} contains unknown field(s): {unknown_list}"
        )

    return out


def load_config(path: str) -> MultiQPUConfig:
    """Load MultiQPUConfig from JSON or YAML."""
    if path.endswith((".yaml", ".yml")):
        yaml = _load_yaml_module()
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    return MultiQPUConfig(**_validate_config_data(data, path))


def dump_config(cfg: MultiQPUConfig, path: str) -> None:
    """Save MultiQPUConfig to JSON or YAML."""
    data: dict[str, Any] = asdict(cfg)
    if path.endswith((".yaml", ".yml")):
        yaml = _load_yaml_module()
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
