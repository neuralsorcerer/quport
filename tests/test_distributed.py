# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig
from quport.distributed import split_into_qpus


def test_split_into_qpus_multiqubit_remote_op_is_deterministic() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    mapped.ccx(0, 1, 2)

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1
    remote = program.remote_ops[0]
    assert remote.qpu0 == 0
    assert remote.qpu1 == 1


def test_split_into_qpus_multiqubit_remote_op_prefers_q1_qubit_qpu() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    # q0 -> QPU2, q1 -> QPU1, q2 -> QPU0
    mapped.ccx(4, 2, 0)

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1
    remote = program.remote_ops[0]
    assert remote.q0_phys == 4
    assert remote.q1_phys == 2
    assert remote.qpu0 == 2
    assert remote.qpu1 == 1


def test_split_into_qpus_multiqubit_remote_op_fallback_when_q1_local() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    # q0 and q1 are on QPU0; q2 is on QPU2.
    mapped.ccx(0, 1, 4)

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1
    remote = program.remote_ops[0]
    assert remote.q0_phys == 0
    assert remote.q1_phys == 4
    assert remote.qpu0 == 0
    assert remote.qpu1 == 2


def test_split_into_qpus_multiqubit_remote_op_adds_barriers_to_participating_qpus() -> (
    None
):
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    # Spans all three QPUs under this configuration.
    mapped.ccx(0, 2, 4)

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1

    expected_barrier_qubit = {0: 0, 1: 2, 2: 4}
    for qpu, phys in expected_barrier_qubit.items():
        local = program.local_circuits[qpu]
        assert local.count_ops().get("barrier", 0) == 1
        barrier_inst = next(
            inst for inst in local.data if inst.operation.name == "barrier"
        )
        assert [local.find_bit(q).index for q in barrier_inst.qubits] == [phys]


def test_split_into_qpus_multiqubit_barrier_is_not_remote_op() -> None:
    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    mapped.barrier(0, 2, 4)

    program = split_into_qpus(mapped, arch)

    assert program.remote_ops == []
    expected_barrier_qubit = {0: 0, 1: 2, 2: 4}
    for qpu, phys in expected_barrier_qubit.items():
        local = program.local_circuits[qpu]
        assert local.count_ops().get("barrier", 0) == 1
        barrier_inst = next(
            inst for inst in local.data if inst.operation.name == "barrier"
        )
        assert [local.find_bit(q).index for q in barrier_inst.qubits] == [phys]


def test_split_into_qpus_global_barrier_is_supported() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    mapped.barrier()

    program = split_into_qpus(mapped, arch)

    assert program.remote_ops == []
    for qpu in range(cfg.n_qpus):
        assert program.local_circuits[qpu].count_ops().get("barrier", 0) == 1


def test_split_into_qpus_multiqubit_remote_op_preserves_qpu_local_operand_order() -> (
    None
):
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    # q0 and q2 on QPU0, q1 on QPU1
    mapped.ccx(0, 3, 1)

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1
    q0_local = program.local_circuits[0]
    barrier_inst = next(
        inst for inst in q0_local.data if inst.operation.name == "barrier"
    )
    assert [q0_local.find_bit(q).index for q in barrier_inst.qubits] == [0, 1]


def test_split_into_qpus_zero_qubit_instruction_is_broadcast() -> None:
    from qiskit.circuit import Instruction

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys)
    mapped.append(Instruction("global_sync", 0, 0, []), [], [])

    program = split_into_qpus(mapped, arch)

    assert program.remote_ops == []
    for qpu in range(cfg.n_qpus):
        local = program.local_circuits[qpu]
        assert local.count_ops().get("global_sync", 0) == 1


def test_split_into_qpus_preserves_single_qpu_clbit_operations() -> None:
    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys, 2)
    mapped.measure(0, 1)

    program = split_into_qpus(mapped, arch)

    assert program.remote_ops == []
    qpu0 = program.local_circuits[0]
    assert qpu0.num_clbits == 2
    measure_inst = next(inst for inst in qpu0.data if inst.operation.name == "measure")
    assert [qpu0.find_bit(q).index for q in measure_inst.qubits] == [0]
    assert [qpu0.find_bit(c).index for c in measure_inst.clbits] == [1]
    assert program.local_circuits[1].count_ops().get("measure", 0) == 0


def test_split_into_qpus_zero_qubit_clbit_instruction_is_broadcast() -> None:
    from qiskit.circuit import Instruction

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys, 2)
    mapped.append(Instruction("classical_sync", 0, 1, []), [], [mapped.clbits[1]])

    program = split_into_qpus(mapped, arch)

    assert program.remote_ops == []
    for qpu in range(cfg.n_qpus):
        local = program.local_circuits[qpu]
        inst = next(i for i in local.data if i.operation.name == "classical_sync")
        assert [local.find_bit(c).index for c in inst.clbits] == [1]


def test_split_into_qpus_remote_op_preserves_clbit_indices() -> None:
    from qiskit.circuit import Instruction

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys, 3)
    mapped.append(
        Instruction("remote_with_clbit", 2, 1, []),
        [mapped.qubits[0], mapped.qubits[2]],
        [mapped.clbits[2]],
    )

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1
    remote = program.remote_ops[0]
    assert remote.qpu0 == 0
    assert remote.qpu1 == 1
    assert remote.clbits == (2,)


def test_split_into_qpus_multiqubit_remote_op_preserves_clbit_indices() -> None:
    from qiskit.circuit import Instruction

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)

    mapped = QuantumCircuit(arch.n_phys, 4)
    mapped.append(
        Instruction("remote_multi_with_clbit", 3, 2, []),
        [mapped.qubits[0], mapped.qubits[2], mapped.qubits[4]],
        [mapped.clbits[1], mapped.clbits[3]],
    )

    program = split_into_qpus(mapped, arch)

    assert len(program.remote_ops) == 1
    remote = program.remote_ops[0]
    assert remote.qpu0 == 0
    assert remote.qpu1 == 1
    assert remote.clbits == (1, 3)


@pytest.mark.parametrize(
    ("bad_decay", "message"),
    [
        (0.0, "decay must be within"),
        (-0.5, "decay must be within"),
        (1.01, "decay must be within"),
        (float("nan"), "decay must be finite"),
        (float("inf"), "decay must be finite"),
        (float("-inf"), "decay must be finite"),
        (True, "decay must be numeric, not boolean"),
        (None, "decay must be numeric"),
        (object(), "decay must be numeric"),
    ],
)
def test_temporal_twoq_weights_reject_invalid_decay(
    bad_decay: object, message: str
) -> None:
    from quport.interaction import extract_temporal_twoq_weights

    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match=message):
        extract_temporal_twoq_weights(qc, decay=bad_decay)


def test_temporal_twoq_weights_decay_one_matches_uniform_count() -> None:
    from quport.interaction import extract_temporal_twoq_weights

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cz(0, 1)

    assert extract_temporal_twoq_weights(qc, decay=1.0) == {(0, 1): 2.0}


def test_temporal_twoq_weights_accumulates_undirected_decayed_weights() -> None:
    from quport.interaction import extract_temporal_twoq_weights

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cz(0, 1)

    assert extract_temporal_twoq_weights(qc, decay=0.5) == {(0, 1): 1.75}


def test_validate_temporal_decay_uses_custom_label() -> None:
    from quport.interaction import validate_temporal_decay

    assert validate_temporal_decay("0.25", label="temporal_decay") == 0.25
    with pytest.raises(ValueError, match="temporal_decay must be within"):
        validate_temporal_decay(2.0, label="temporal_decay")


def test_compile_distributed_rejects_invalid_temporal_decay_for_tpccap() -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="clique",
        inter_topology="switch",
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match="temporal_decay must be within"):
        compile_distributed(qc, cfg, strategy="tpccap", temporal_decay=1.1)


def test_compile_distributed_validates_strategy_before_temporal_decay() -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
    )
    qc = QuantumCircuit(2)

    with pytest.raises(ValueError, match="Unknown strategy"):
        compile_distributed(qc, cfg, strategy="invalid", temporal_decay=1.1)


def test_compile_distributed_ignores_temporal_decay_for_non_tpccap_strategies() -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="clique",
        inter_topology="switch",
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    # Non-TPCCAP strategies always use uniform interaction weights and should not
    # validate temporal_decay at all.
    result = compile_distributed(qc, cfg, strategy="balanced", temporal_decay=1.1)
    assert result.strategy == "balanced"


def test_compile_distributed_rejects_circuit_larger_than_physical_capacity() -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=1,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="switch",
    )
    qc = QuantumCircuit(2)

    with pytest.raises(ValueError, match="exceed physical qubits"):
        compile_distributed(qc, cfg, strategy="balanced")
