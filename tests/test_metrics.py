# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig
from quport.metrics import CircuitMetrics, compute_metrics


def _two_qpu_arch() -> MultiQPUArchitecture:
    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    return MultiQPUArchitecture(cfg)


def test_compute_metrics_counts_local_and_remote_operations() -> None:
    arch = _two_qpu_arch()

    qc = QuantumCircuit(arch.n_phys)
    qc.h(0)
    qc.cx(0, 1)  # local 2q on QPU 0
    qc.cx(0, 3)  # remote 2q across QPU 0 -> QPU 1
    qc.swap(1, 2)  # local swap on QPU 0

    metrics = compute_metrics(qc, arch)

    assert metrics.n_1q == 1
    assert metrics.n_2q == 3
    assert metrics.swaps == 1
    assert metrics.remote_2q == 1


def test_compute_metrics_ignores_barrier_directives() -> None:
    arch = _two_qpu_arch()

    qc = QuantumCircuit(arch.n_phys)
    qc.barrier(0)
    qc.barrier(0, 3)  # spans both QPUs but is a directive, not a remote op
    qc.barrier()

    metrics = compute_metrics(qc, arch)

    assert metrics == CircuitMetrics(
        swaps=0, depth=0, size=0, n_1q=0, n_2q=0, remote_2q=0
    )


def test_compute_metrics_counts_gates_but_not_barriers_in_mixed_circuit() -> None:
    arch = _two_qpu_arch()

    qc = QuantumCircuit(arch.n_phys)
    qc.cx(0, 1)
    qc.barrier(0, 3)

    metrics = compute_metrics(qc, arch)

    assert metrics.n_2q == 1
    assert metrics.remote_2q == 0
