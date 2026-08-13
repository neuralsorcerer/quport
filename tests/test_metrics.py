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


def test_count_ops_tallies_each_instruction_once() -> None:
    from qiskit import QuantumCircuit

    from quport.metrics import count_ops

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.barrier()
    circuit.h(0)

    assert dict(count_ops(circuit)) == {"h": 2, "cx": 2, "barrier": 1}


def test_compute_cut_sums_only_cross_partition_weight() -> None:
    """`compute_cut` is exported but was never called by the suite.

    It must count an edge exactly when its endpoints sit on different QPUs.
    """
    from quport.metrics import compute_cut

    weights = {(0, 1): 2.0, (1, 2): 5.0, (0, 2): 7.0}

    assert compute_cut(weights, [0, 1, 1]) == pytest.approx(9.0)
    assert compute_cut(weights, [0, 0, 0]) == pytest.approx(0.0)
    assert compute_cut(weights, [0, 1, 2]) == pytest.approx(14.0)
    assert compute_cut({}, [0, 1]) == pytest.approx(0.0)


def test_swaps_is_basis_dependent_and_reads_zero_under_the_default_basis() -> None:
    """Pin the documented `swaps` semantics, which the published numbers rest on.

    README and docs/api-references.md both state that `swaps` counts instructions
    literally named `swap`, so the default basis -- which has no `swap` -- rewrites
    every routing SWAP into CX gates and leaves `swaps` reading 0. Nothing tested
    that, so adding `swap` to the default basis, or changing how `n_2q` treats it,
    would silently make the documentation false.

    The claim is only meaningful if routing really does insert SWAPs, so this
    transpiles the same circuit twice and compares.
    """
    from qiskit import transpile

    from quport.pipeline import map_and_transpile, random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=5,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="switch",
        optimization_level=1,
    )
    assert "swap" not in cfg.basis_gates

    arch = MultiQPUArchitecture(cfg)
    circuit = random_benchmark_circuit(n_logical=8, depth=8, seed=0)

    result = map_and_transpile(circuit, cfg, seed=0, strategy="balanced")

    # the same routing, but with `swap` kept in the basis so the SWAPs survive
    keeping_swaps = transpile(
        transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
            seed_transpiler=0,
        ),
        coupling_map=arch.build_coupling_map(),
        basis_gates=["rz", "sx", "x", "cx", "swap"],
        optimization_level=cfg.optimization_level,
        layout_method=cfg.layout_method,
        routing_method=cfg.routing_method,
        seed_transpiler=0,
    )
    routing_swaps = sum(
        1 for inst in keeping_swaps.data if inst.operation.name == "swap"
    )

    assert routing_swaps > 0, "expected this circuit to need routing"
    assert result.metrics.swaps == 0
    # the routing cost is not lost: it lands in n_2q as three CX per SWAP
    assert result.metrics.n_2q > 0


def test_a_swap_instruction_counts_toward_both_swaps_and_two_qubit_gates() -> None:
    """`n_2q` counts every two-qubit instruction, SWAPs included.

    That makes `LatencyModel.swap` an increment over the two-qubit cost rather
    than the total cost of a SWAP, which is what `estimate_cost` assumes. The
    two definitions have to agree, so pin the counting side here.
    """
    arch = _two_qpu_arch()

    qc = QuantumCircuit(arch.n_phys)
    qc.swap(0, 1)

    metrics = compute_metrics(qc, arch)

    assert metrics.swaps == 1
    assert metrics.n_2q == 1
    assert metrics.n_1q == 0
