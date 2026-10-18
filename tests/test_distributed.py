# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from quport.architecture import MultiQPUArchitecture
from quport.config import MultiQPUConfig
from quport.distributed import DistributedProgram, split_into_qpus


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


def test_split_into_qpus_puts_a_zero_qubit_instruction_in_one_program() -> None:
    """No QPU owns it, but it must run once: a copy per program ran it once
    per QPU, and reassembly then dropped every copy."""
    from qiskit.circuit import Instruction

    from quport.distributed import reassemble_distributed_program

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
    assert [
        program.local_circuits[qpu].count_ops().get("global_sync", 0)
        for qpu in range(cfg.n_qpus)
    ] == [1, 0]
    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch
    )
    assert merged.count_ops().get("global_sync", 0) == 1


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


def test_split_into_qpus_keeps_a_zero_qubit_clbit_instruction_once() -> None:
    from qiskit.circuit import Instruction

    from quport.distributed import reassemble_distributed_program

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
    local = program.local_circuits[0]
    inst = next(i for i in local.data if i.operation.name == "classical_sync")
    assert [local.find_bit(c).index for c in inst.clbits] == [1]
    assert not program.local_circuits[1].data
    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch
    )
    assert [
        [merged.find_bit(c).index for c in i.clbits]
        for i in merged.data
        if i.operation.name == "classical_sync"
    ] == [[1]]


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


def test_compile_distributed_exposes_schedule_plan_matching_summary() -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    result = compile_distributed(qc, cfg, strategy="balanced", seed=0)

    assert result.schedule_plan.summary == result.schedule
    assert result.schedule_plan.summary.remote_ops == len(result.program.remote_ops)
    assert len(result.schedule_plan.layers) == result.schedule.layers


def test_remote_op_to_dict_is_json_safe_for_symbolic_and_complex_params() -> None:
    import json

    from qiskit.circuit import Parameter

    from quport.distributed import RemoteOp

    theta = Parameter("θ")
    remote = RemoteOp(
        name="remote_parametric",
        q0_phys=0,
        q1_phys=2,
        qpu0=0,
        qpu1=1,
        params=(theta, theta + 1, 1.25, 2 + 3j, ("nested", theta)),
        clbits=(1, 3),
        index=7,
    )

    payload = remote.to_dict()

    json.dumps(payload)
    assert payload["name"] == "remote_parametric"
    assert payload["params"][0] == {"type": "Parameter", "repr": "θ"}
    assert payload["params"][2] == 1.25
    assert payload["params"][3] == {"type": "complex", "real": 2.0, "imag": 3.0}
    assert payload["params"][4][0] == "nested"
    assert payload["clbits"] == [1, 3]


def test_distributed_program_remote_ops_payload_and_writer_are_json_safe(
    tmp_path: Path,
) -> None:
    import json

    from qiskit.circuit import Parameter

    from quport.distributed import write_remote_ops_json

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    theta = Parameter("theta")
    mapped = QuantumCircuit(arch.n_phys)
    mapped.crx(theta, 0, 2)

    program = split_into_qpus(mapped, arch)
    payload = program.remote_ops_payload()
    out = tmp_path / "remote_ops.json"
    write_remote_ops_json(program.remote_ops, out)

    assert json.loads(json.dumps(payload)) == json.loads(
        out.read_text(encoding="utf-8")
    )
    assert payload[0]["name"] == "crx"
    assert payload[0]["params"] == [{"type": "Parameter", "repr": "theta"}]


def test_remote_op_to_dict_encodes_nonfinite_bytes_sets_and_mapping_collisions() -> (
    None
):
    import json
    import math

    from quport.distributed import RemoteOp

    remote = RemoteOp(
        name="remote_edge_params",
        q0_phys=0,
        q1_phys=1,
        qpu0=0,
        qpu1=1,
        params=(
            float("nan"),
            float("inf"),
            complex(float("-inf"), math.nan),
            b"abc",
            {"1": "string-key", 1: "integer-key"},
            {3, 1, 2},
        ),
        clbits=(),
        index=0,
    )

    payload = remote.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert "NaN" not in encoded
    assert payload["params"][0] == {"type": "float", "value": "nan"}
    assert payload["params"][1] == {"type": "float", "value": "inf"}
    assert payload["params"][2] == {
        "type": "complex",
        "real": {"type": "float", "value": "-inf"},
        "imag": {"type": "float", "value": "nan"},
    }
    assert payload["params"][3] == {
        "type": "bytes",
        "encoding": "base64",
        "data": "YWJj",
    }
    assert payload["params"][4] == {
        "type": "mapping",
        "entries": [["1", "string-key"], [1, "integer-key"]],
    }
    assert payload["params"][5] == {"type": "set", "items": [1, 2, 3]}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "", "remote operation name must be a non-empty string"),
        ("q0_phys", -1, "q0_phys must be non-negative"),
        ("q1_phys", True, "q1_phys must be an integer"),
        ("q1_phys", 0, "physical qubits must be distinct"),
        ("qpu0", 1.5, "qpu0 must be an integer"),
        ("qpu1", -1, "qpu1 must be non-negative"),
        ("qpu1", 0, "QPUs must be distinct"),
        ("params", None, "params must be a sequence"),
        ("params", "not-a-sequence", "params must be a sequence"),
        ("clbits", None, "clbits must be a sequence"),
        ("clbits", (False,), "clbit index must be an integer"),
        ("index", -1, "index must be non-negative"),
    ],
)
def test_remote_op_to_dict_rejects_invalid_manifest_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    from quport.distributed import RemoteOp

    kwargs: dict[str, object] = {
        "name": "remote",
        "q0_phys": 0,
        "q1_phys": 1,
        "qpu0": 0,
        "qpu1": 1,
        "params": (),
        "clbits": (),
        "index": 0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        RemoteOp(**kwargs).to_dict()  # type: ignore[arg-type]


def test_remote_op_to_dict_rejects_cyclic_parameter_containers() -> None:
    from quport.distributed import RemoteOp

    cyclic: list[object] = []
    cyclic.append(cyclic)
    remote = RemoteOp(
        name="remote_cyclic",
        q0_phys=0,
        q1_phys=1,
        qpu0=0,
        qpu1=1,
        params=(cyclic,),
        clbits=(),
        index=0,
    )

    with pytest.raises(ValueError, match="parameters cannot contain cycles"):
        remote.to_dict()


def test_write_remote_ops_json_creates_parent_directories(tmp_path: Path) -> None:
    import json

    from quport.distributed import RemoteOp, write_remote_ops_json

    out = tmp_path / "nested" / "remote_ops.json"
    remote = RemoteOp(
        name="remote",
        q0_phys=0,
        q1_phys=1,
        qpu0=0,
        qpu1=1,
        params=(),
        clbits=(),
        index=0,
    )

    write_remote_ops_json((remote,), out)

    assert json.loads(out.read_text(encoding="utf-8")) == [remote.to_dict()]


def test_write_remote_ops_json_accepts_generators_and_rejects_bad_entries(
    tmp_path: Path,
) -> None:
    from quport.distributed import RemoteOp, write_remote_ops_json

    remote = RemoteOp(
        name="remote",
        q0_phys=0,
        q1_phys=1,
        qpu0=0,
        qpu1=1,
        params=(),
        clbits=(),
        index=0,
    )
    write_remote_ops_json((op for op in (remote,)), tmp_path / "ops.json")

    with pytest.raises(ValueError, match=r"remote_ops\[0\] must be a RemoteOp"):
        write_remote_ops_json((object(),), tmp_path / "bad.json")  # type: ignore[arg-type]


def test_write_distributed_program_exports_qasm_and_remote_manifest(
    tmp_path: Path,
) -> None:
    import json

    from quport.distributed import write_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=1,
        comm_qubits_per_qpu=0,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.h(0)
    mapped.cx(0, 1)

    program = split_into_qpus(mapped, arch)
    written = write_distributed_program(program, tmp_path / "bundle")

    assert set(written) == {"qpu_0", "qpu_1", "remote_ops"}
    assert "OPENQASM" in written["qpu_0"].read_text(encoding="utf-8")
    assert "h" in written["qpu_0"].read_text(encoding="utf-8")
    assert "OPENQASM" in written["qpu_1"].read_text(encoding="utf-8")
    payload = json.loads(written["remote_ops"].read_text(encoding="utf-8"))
    assert payload == [program.remote_ops[0].to_dict()]


def test_write_distributed_program_can_skip_empty_local_circuits(
    tmp_path: Path,
) -> None:
    from quport.distributed import write_distributed_program

    program = DistributedProgram(
        local_circuits={0: QuantumCircuit(1), 1: QuantumCircuit(1)}, remote_ops=[]
    )
    program.local_circuits[0].x(0)

    written = write_distributed_program(
        program, tmp_path / "bundle", include_empty_circuits=False
    )

    assert set(written) == {"qpu_0", "remote_ops"}
    assert not (tmp_path / "bundle" / "qpu_1.qasm").exists()


@pytest.mark.parametrize(
    ("program", "include_empty", "message"),
    [
        (object(), True, "program must be a DistributedProgram"),
        (
            DistributedProgram(local_circuits={}, remote_ops=[]),
            1,
            "include_empty_circuits must be a boolean",
        ),
        (
            DistributedProgram(local_circuits={True: QuantumCircuit(1)}, remote_ops=[]),
            True,
            "local circuit QPU id must be an integer",
        ),
        (
            DistributedProgram(local_circuits={0: object()}, remote_ops=[]),
            True,
            r"local_circuits\[0\] must be a QuantumCircuit",
        ),
    ],
)
def test_write_distributed_program_rejects_invalid_inputs(
    tmp_path: Path, program: object, include_empty: object, message: str
) -> None:
    from quport.distributed import write_distributed_program

    with pytest.raises(ValueError, match=message):
        write_distributed_program(
            cast(Any, program),
            tmp_path / "bundle",
            include_empty_circuits=cast(Any, include_empty),
        )


def test_write_distributed_program_accepts_existing_directory_and_pathlike(
    tmp_path: Path,
) -> None:
    from quport.distributed import write_distributed_program

    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    program = DistributedProgram(local_circuits={0: QuantumCircuit(1)}, remote_ops=[])

    written = write_distributed_program(program, out_dir)

    assert written["qpu_0"] == out_dir / "qpu_0.qasm"
    assert written["remote_ops"] == out_dir / "remote_ops.json"


def test_write_distributed_program_validates_entries_before_sorting_or_writing(
    tmp_path: Path,
) -> None:
    from quport.distributed import write_distributed_program

    program = DistributedProgram(
        local_circuits={"bad": QuantumCircuit(1), 0: QuantumCircuit(1)},  # type: ignore[dict-item]
        remote_ops=[],
    )

    with pytest.raises(ValueError, match="local circuit QPU id must be an integer"):
        write_distributed_program(program, tmp_path / "bundle")

    assert not (tmp_path / "bundle" / "qpu_0.qasm").exists()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (123, "path must be a filesystem path"),
        (None, "path must be a filesystem path"),
    ],
)
def test_write_distributed_program_rejects_non_path_outputs(
    tmp_path: Path, path: object, message: str
) -> None:
    from quport.distributed import write_distributed_program

    program = DistributedProgram(local_circuits={0: QuantumCircuit(1)}, remote_ops=[])

    with pytest.raises(ValueError, match=message):
        write_distributed_program(program, path)  # type: ignore[arg-type]


def test_write_distributed_program_rejects_output_file_path(tmp_path: Path) -> None:
    from quport.distributed import write_distributed_program

    out_file = tmp_path / "not_a_directory"
    out_file.write_text("already here", encoding="utf-8")
    program = DistributedProgram(local_circuits={0: QuantumCircuit(1)}, remote_ops=[])

    with pytest.raises(ValueError, match="path must be a directory"):
        write_distributed_program(program, out_file)


@pytest.mark.parametrize("path", [123, None])
def test_write_remote_ops_json_rejects_non_path_outputs(path: object) -> None:
    from quport.distributed import write_remote_ops_json

    with pytest.raises(ValueError, match="path must be a filesystem path"):
        write_remote_ops_json([], path)  # type: ignore[arg-type]


def test_write_distributed_program_is_exported_from_package() -> None:
    import quport
    from quport.distributed import write_distributed_program

    assert quport.write_distributed_program is write_distributed_program


@pytest.mark.parametrize(
    ("seed", "message"),
    [
        (-1, "seed must be non-negative"),
        (True, "seed must be a non-negative integer"),
        (1.5, "seed must be a non-negative integer"),
    ],
)
def test_compile_distributed_rejects_invalid_seed(seed: object, message: str) -> None:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(n_qpus=1, compute_qubits_per_qpu=1, comm_qubits_per_qpu=0)
    qc = QuantumCircuit(1)

    with pytest.raises(ValueError, match=message):
        compile_distributed(qc, cfg, seed=cast(Any, seed))


@pytest.mark.parametrize("strategy", ["balanced", "cluster", "tpccap", "tpccap_sa"])
def test_compile_distributed_completes_for_every_supported_strategy(
    strategy: str,
) -> None:
    """Every advertised strategy must produce a consistent distributed program.

    ``tpccap_sa`` is both the ``compile_distributed`` default and the
    ``compile-dist`` CLI default, so each strategy needs an end-to-end run
    rather than only the argument-validation paths.
    """
    from quport.architecture import MultiQPUArchitecture
    from quport.compiler import compile_distributed
    from quport.metrics import compute_metrics

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    qc = QuantumCircuit(6)
    for control in range(5):
        qc.cx(control, control + 1)
    qc.cx(0, 5)
    qc.cx(1, 4)

    result = compile_distributed(qc, cfg, strategy=strategy, seed=11)

    assert result.strategy == strategy
    assert len(result.partition) == qc.num_qubits
    loads = [result.partition.count(qpu) for qpu in range(cfg.n_qpus)]
    assert max(loads) <= cfg.capacity_per_qpu()
    assert all(0 <= qpu < cfg.n_qpus for qpu in result.partition)

    # One local program per QPU, and remote-op counts agree across every view.
    assert set(result.local_routed) == set(range(cfg.n_qpus))
    assert set(result.program.local_circuits) == set(range(cfg.n_qpus))
    remote_ops = len(result.program.remote_ops)
    assert compute_metrics(result.physical_circuit, arch).remote_2q == remote_ops
    assert result.schedule.remote_ops == remote_ops
    assert result.schedule_plan.summary == result.schedule

    # Diagnostics are reported exactly for the strategies that compute them.
    assert (result.partition_diagnostics is not None) == (
        strategy in ("tpccap", "tpccap_sa")
    )
    assert (result.anneal_diagnostics is not None) == (strategy == "tpccap_sa")


@pytest.mark.parametrize("strategy", ["balanced", "cluster", "tpccap", "tpccap_sa"])
def test_compile_distributed_keeps_local_programs_inside_their_qpu_block(
    strategy: str,
) -> None:
    """Local routing must never place an operation on another QPU's qubits."""
    from quport.architecture import MultiQPUArchitecture
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="ring",
    )
    arch = MultiQPUArchitecture(cfg)
    qc = QuantumCircuit(7)
    for control in range(6):
        qc.cx(control, control + 1)

    result = compile_distributed(qc, cfg, strategy=strategy, seed=5)

    for qpu_id, circuit in result.local_routed.items():
        block = arch.block_of_qpu(qpu_id)
        allowed = set(block.compute + block.comm)
        index = {qubit: position for position, qubit in enumerate(circuit.qubits)}
        for instruction in circuit.data:
            if getattr(instruction.operation, "_directive", False):
                continue
            for qubit in instruction.qubits:
                assert index[qubit] in allowed


def test_split_preserves_source_classical_registers() -> None:
    """Local circuits must carry the source circuit's clbits and registers.

    Building them with a fresh anonymous register leaves register-valued
    conditions pointing at a register the local circuit does not contain.
    """
    from qiskit.circuit import Clbit

    from quport.distributed import split_into_qpus

    arch = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=2,
            compute_qubits_per_qpu=2,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )
    first = ClassicalRegister(2, "alpha")
    second = ClassicalRegister(3, "beta")
    circuit = QuantumCircuit(QuantumRegister(arch.n_phys, "q"), first, second)
    circuit.add_bits([Clbit(), Clbit()])  # loose clbits outside any register
    circuit.h(0)
    circuit.measure(0, first[0])

    program = split_into_qpus(circuit, arch)

    for local in program.local_circuits.values():
        assert list(local.clbits) == list(circuit.clbits)
        assert [reg.name for reg in local.cregs] == ["alpha", "beta"]
        assert local.num_qubits == arch.n_phys


def test_split_handles_conditional_blocks_without_dangling_registers() -> None:
    """An `if_else` inside one QPU must survive splitting, DAG conversion and export."""
    from qiskit import qasm3
    from qiskit.converters import circuit_to_dag

    from quport.distributed import split_into_qpus

    arch = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=2,
            compute_qubits_per_qpu=2,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )
    creg = ClassicalRegister(1, "c0")
    circuit = QuantumCircuit(QuantumRegister(arch.n_phys, "q"), creg)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((creg, 1)):
        circuit.cx(0, 1)  # both operands live in QPU 0

    program = split_into_qpus(circuit, arch)

    assert program.remote_ops == []
    assert program.local_circuits[0].count_ops()["if_else"] == 1
    for local in program.local_circuits.values():
        circuit_to_dag(local)  # panics on a dangling register reference
        qasm3.dumps(local)


def test_compile_distributed_accepts_circuits_with_control_flow() -> None:
    from qiskit import qasm3

    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
        optimization_level=0,
    )
    creg = ClassicalRegister(1, "c0")
    circuit = QuantumCircuit(QuantumRegister(4, "q"), creg)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((creg, 1)):
        circuit.cx(0, 1)

    result = compile_distributed(circuit, cfg, strategy="balanced", seed=0)

    assert set(result.local_routed) == {0, 1}
    for local in result.local_routed.values():
        qasm3.dumps(local)


def test_split_preserves_measurements_on_their_own_qpu() -> None:
    from quport.distributed import split_into_qpus

    arch = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=3,
            compute_qubits_per_qpu=1,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )
    qreg = QuantumRegister(arch.n_phys, "q")
    creg = ClassicalRegister(arch.n_phys, "c")
    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(qreg, creg)

    program = split_into_qpus(circuit, arch)

    total = 0
    for qpu_id, local in program.local_circuits.items():
        block = set(arch.block_of_qpu(qpu_id).compute + arch.block_of_qpu(qpu_id).comm)
        index = {qubit: position for position, qubit in enumerate(local.qubits)}
        for instruction in local.data:
            if instruction.operation.name == "measure":
                total += 1
                assert index[instruction.qubits[0]] in block
    assert total == arch.n_phys


@pytest.mark.parametrize("intra_topology", ["clique", "line", "grid2d"])
def test_split_conserves_every_operation_exactly_once(intra_topology: str) -> None:
    """Splitting must neither drop nor duplicate work.

    Each non-directive operation of the physical circuit has to reappear exactly
    once, either in one QPU's local program or as a remote operation.
    """
    from collections import Counter

    from quport.compiler import compile_distributed
    from quport.distributed import split_into_qpus
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology=intra_topology,
        inter_topology="ring",
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    physical = compile_distributed(
        random_benchmark_circuit(n_logical=6, depth=4, seed=8),
        cfg,
        seed=8,
        strategy="balanced",
    ).physical_circuit
    program = split_into_qpus(physical, arch)

    def tally(circuit: QuantumCircuit) -> Counter:
        position = {qubit: index for index, qubit in enumerate(circuit.qubits)}
        counts: Counter = Counter()
        for instruction in circuit.data:
            if getattr(instruction.operation, "_directive", False):
                continue
            operands = tuple(position[qubit] for qubit in instruction.qubits)
            counts[(instruction.operation.name, operands)] += 1
        return counts

    expected = tally(physical)
    actual: Counter = Counter()
    for local in program.local_circuits.values():
        actual += tally(local)
    for remote in program.remote_ops:
        actual[(remote.name, (remote.q0_phys, remote.q1_phys))] += 1

    assert actual == expected
    assert sum(expected.values()) > 0


def test_local_routing_preserves_each_local_program_unitary() -> None:
    """Intra-QPU SABRE routing must not change what a local program computes."""
    from qiskit.quantum_info import Operator

    from quport.compiler import compile_distributed
    from quport.distributed import split_into_qpus
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="switch",
        optimization_level=1,
    )
    arch = MultiQPUArchitecture(cfg)
    result = compile_distributed(
        random_benchmark_circuit(n_logical=5, depth=4, seed=6),
        cfg,
        seed=6,
        strategy="balanced",
    )
    program = split_into_qpus(result.physical_circuit, arch)

    compared = 0
    for qpu_id, routed in result.local_routed.items():
        before = program.local_circuits[qpu_id]
        if not before.data:
            continue
        compared += 1
        assert Operator.from_circuit(routed).equiv(Operator(before))
    assert compared > 0


def test_compile_distributed_routes_only_on_per_qpu_coupling_maps() -> None:
    """Local routing must never be handed the global map.

    The no-cross-QPU-SWAP guarantee rests entirely on SABRE seeing only one
    QPU's edges. Passing the global map instead is invisible in the output --
    SABRE simply never finds an inter-QPU route attractive -- so the property is
    asserted structurally rather than behaviourally.
    """
    from unittest.mock import patch

    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="ring",
        optimization_level=1,
    )
    global_calls = 0
    intra_calls: list[int] = []
    build_global = MultiQPUArchitecture.build_coupling_map
    build_intra = MultiQPUArchitecture.build_intra_coupling_map

    def spy_global(self: MultiQPUArchitecture):  # type: ignore[no-untyped-def]
        nonlocal global_calls
        global_calls += 1
        return build_global(self)

    def spy_intra(self: MultiQPUArchitecture, qpu_id: int):  # type: ignore[no-untyped-def]
        intra_calls.append(qpu_id)
        return build_intra(self, qpu_id)

    with (
        patch.object(MultiQPUArchitecture, "build_coupling_map", spy_global),
        patch.object(MultiQPUArchitecture, "build_intra_coupling_map", spy_intra),
    ):
        compile_distributed(
            random_benchmark_circuit(n_logical=6, depth=5, seed=1),
            cfg,
            seed=1,
            strategy="balanced",
        )

    assert global_calls == 0, "distributed compilation must not build the global map"
    assert sorted(intra_calls) == list(range(cfg.n_qpus))


@pytest.mark.parametrize(
    ("strategy", "expected_mode"),
    [
        ("balanced", "topk"),
        ("cluster", "topk"),
        ("tpccap", "diverse"),
        ("tpccap_sa", "diverse"),
    ],
)
def test_compile_distributed_uses_diverse_comm_selection_for_tpccap(
    strategy: str, expected_mode: str
) -> None:
    """Topology-aware strategies pair with diversity-aware port selection."""
    from unittest.mock import patch

    import quport.compiler as compiler
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=2,
        intra_topology="line",
        inter_topology="ring",
        optimization_level=0,
    )
    seen: list[str] = []
    original = compiler.compute_layout_hints

    def spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seen.append(str(kwargs.get("comm_mode")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(compiler, "compute_layout_hints", spy):
        compiler.compile_distributed(
            random_benchmark_circuit(n_logical=6, depth=4, seed=2),
            cfg,
            seed=2,
            strategy=strategy,
        )

    assert seen == [expected_mode]


def test_remote_operation_barriers_both_participating_qpus() -> None:
    """Both sides of a remote operation need an explicit synchronization point."""
    from quport.distributed import split_into_qpus

    arch = MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=2,
            compute_qubits_per_qpu=1,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )
    circuit = QuantumCircuit(arch.n_phys)
    circuit.cx(0, 2)  # QPU0 qubit 0 <-> QPU1 qubit 2

    program = split_into_qpus(circuit, arch)

    assert len(program.remote_ops) == 1
    barriers = {}
    for qpu_id, local in program.local_circuits.items():
        position = {qubit: index for index, qubit in enumerate(local.qubits)}
        barriers[qpu_id] = [
            tuple(position[qubit] for qubit in instruction.qubits)
            for instruction in local.data
            if instruction.operation.name == "barrier"
        ]
    assert barriers[0] == [(0,)]
    assert barriers[1] == [(2,)]


def test_json_safe_set_items_are_sorted_for_reproducible_manifests() -> None:
    """Set ordering must not depend on hash randomisation.

    String sets iterate in a PYTHONHASHSEED-dependent order, so the items are
    sorted before export to keep manifests byte-reproducible across runs.
    """
    from quport.distributed import _json_safe_value

    assert _json_safe_value({"b", "a", "c"}) == {
        "type": "set",
        "items": ["a", "b", "c"],
    }
    assert _json_safe_value(frozenset({3, 1, 2})) == {
        "type": "frozenset",
        "items": [1, 2, 3],
    }


def test_json_safe_mapping_preserves_entries_when_string_keys_repeat() -> None:
    """A mapping that yields a key twice must not silently lose an entry."""
    from collections.abc import Mapping

    from quport.distributed import _json_safe_value

    class RepeatedKeys(Mapping):
        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(["a", "a"])

        def __len__(self) -> int:
            return 2

        def __getitem__(self, key: object) -> int:
            return 1

        def items(self):  # type: ignore[no-untyped-def]
            return [("a", 1), ("a", 2)]

    assert _json_safe_value(RepeatedKeys()) == {
        "type": "mapping",
        "entries": [["a", 1], ["a", 2]],
    }


def test_zero_qubit_barrier_is_broadcast_to_every_qpu() -> None:
    """A barrier carrying no qubits is a global sync point, not a no-op.

    `split_into_qpus` routes ops by the QPUs their qubits live on; an op with no
    qubits has none, so it has to be replicated onto every local circuit or the
    per-QPU programs lose the synchronization the original circuit expressed.
    """
    from qiskit.circuit import Barrier

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        inter_topology="switch",
    )
    arch = MultiQPUArchitecture(cfg)

    circuit = QuantumCircuit(arch.n_phys)
    circuit.append(Barrier(0), [], [])

    program = split_into_qpus(circuit, arch)

    assert len(program.local_circuits) == cfg.n_qpus
    for local in program.local_circuits.values():
        assert [inst.operation.name for inst in local.data] == ["barrier"]
    assert program.remote_ops == []


def test_grouping_rejects_mismatched_qubit_and_qpu_lengths() -> None:
    """The operand-order grouping helper pairs the two lists element-wise."""
    from quport.distributed import _group_qubits_by_qpu_in_operand_order

    with pytest.raises(ValueError, match="same length"):
        _group_qubits_by_qpu_in_operand_order([0, 1, 2], [0, 1])


@pytest.mark.parametrize(
    "inter_topology", ["switch", "mesh", "ring", "degree_d", "clos", "fat_tree"]
)
@pytest.mark.parametrize("strategy", ["balanced", "tpccap"])
def test_local_circuits_only_ever_touch_their_own_qpus_qubits(
    inter_topology: str, strategy: str
) -> None:
    """The headline guarantee, asserted on the emitted program rather than the setup.

    `test_compile_distributed_routes_only_on_per_qpu_coupling_maps` pins the
    structural cause -- SABRE only ever sees one QPU's edges. This pins the effect,
    so a violation arriving by any other route is caught too: every instruction in
    local circuit q must touch only q's physical qubits, and every RemoteOp must
    name two genuinely different QPUs.
    """
    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=2,
        intra_topology="line",
        inter_topology=inter_topology,
        optimization_level=1,
    )
    arch = MultiQPUArchitecture(cfg)
    circuit = random_benchmark_circuit(n_logical=8, depth=8, seed=0)

    program = compile_distributed(circuit, cfg, seed=0, strategy=strategy).program

    owned = {
        qpu: set(arch.block_of_qpu(qpu).compute) | set(arch.block_of_qpu(qpu).comm)
        for qpu in range(cfg.n_qpus)
    }

    for qpu, local in program.local_circuits.items():
        position = {qubit: index for index, qubit in enumerate(local.qubits)}
        for instruction in local.data:
            touched = {position[qubit] for qubit in instruction.qubits}
            assert touched <= owned[qpu], (
                f"local circuit {qpu} touches {sorted(touched - owned[qpu])}, "
                f"which it does not own"
            )

    assert program.remote_ops, "expected this circuit to need remote operations"
    for remote in program.remote_ops:
        assert remote.qpu0 != remote.qpu1
        assert remote.q0_phys in owned[remote.qpu0]
        assert remote.q1_phys in owned[remote.qpu1]


def test_remote_barriers_carry_a_label_naming_their_operation() -> None:
    """The marker label is what lets a routed program be matched to a manifest."""
    from quport.distributed import remote_barrier_label

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)
    mapped.cx(1, 4)

    program = split_into_qpus(mapped, arch)

    labels = [
        instruction.operation.label
        for instruction in program.local_circuits[0].data
        if instruction.operation.name == "barrier"
    ]
    assert labels == [remote_barrier_label(0), remote_barrier_label(1)]


def test_labels_do_not_leak_into_emitted_qasm() -> None:
    """Marker labels are transpiler metadata, not part of the program text."""
    from qiskit import qasm3

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)

    program = split_into_qpus(mapped, arch)
    source = qasm3.dumps(program.local_circuits[0])

    assert "barrier" in source
    assert "quport_remote" not in source


def test_routed_manifest_follows_qubits_that_local_routing_moved() -> None:
    """Regression: the manifest must match the programs it is shipped with.

    ``split_into_qpus`` records physical qubits as they stand *before* local
    routing. Routing permutes qubits inside a QPU whenever the intra-QPU
    topology is not a clique, so pairing the original manifest with routed
    programs would point a consumer at the wrong qubit. Here routing moves the
    remote operation's operand from physical qubit 0 to 2, and the shipped
    manifest has to say 2.
    """
    from qiskit import transpile

    from quport.distributed import remap_remote_ops_to_routed

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        basis_gates=("rz", "sx", "x", "cx", "swap"),
    )
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 2)
    mapped.cx(0, 3)
    mapped.cx(1, 3)
    mapped.cx(0, 2)
    mapped.cx(0, 4)  # the remote operation, on physical qubit 0 before routing

    program = split_into_qpus(mapped, arch)
    assert [(op.q0_phys, op.q1_phys) for op in program.remote_ops] == [(0, 4)]

    local_routed = {
        qpu: transpile(
            circuit,
            coupling_map=arch.build_intra_coupling_map(qpu),
            initial_layout=list(range(arch.n_phys)),
            basis_gates=list(cfg.basis_gates),
            optimization_level=1,
            layout_method="trivial",
            routing_method="sabre",
            seed_transpiler=0,
        )
        for qpu, circuit in program.local_circuits.items()
    }

    routed_ops = remap_remote_ops_to_routed(program.remote_ops, local_routed)

    # The barrier the transpiler carried along names the truth.
    marker = next(
        instruction
        for instruction in local_routed[0].data
        if instruction.operation.name == "barrier"
    )
    moved_to = local_routed[0].find_bit(marker.qubits[0]).index
    assert moved_to == 2
    assert [(op.q0_phys, op.q1_phys) for op in routed_ops] == [(moved_to, 4)]

    # Everything else about the operation is untouched.
    assert routed_ops[0].index == program.remote_ops[0].index
    assert routed_ops[0].qpu0 == 0
    assert routed_ops[0].qpu1 == 1
    assert routed_ops[0].name == program.remote_ops[0].name


def test_clique_topologies_need_no_remapping() -> None:
    """The default intra topology never permutes, which is why this went unseen."""
    from qiskit import transpile

    from quport.distributed import remap_remote_ops_to_routed

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
    )
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 2)
    mapped.cx(0, 4)

    program = split_into_qpus(mapped, arch)
    local_routed = {
        qpu: transpile(
            circuit,
            coupling_map=arch.build_intra_coupling_map(qpu),
            initial_layout=list(range(arch.n_phys)),
            basis_gates=list(cfg.basis_gates),
            optimization_level=3,
            layout_method="trivial",
            routing_method="sabre",
            seed_transpiler=0,
        )
        for qpu, circuit in program.local_circuits.items()
    }

    assert remap_remote_ops_to_routed(program.remote_ops, local_routed) == list(
        program.remote_ops
    )


def test_remap_reports_a_manifest_that_lost_its_markers() -> None:
    from quport.distributed import remap_remote_ops_to_routed

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)

    program = split_into_qpus(mapped, arch)
    stripped = {qpu: QuantumCircuit(arch.n_phys) for qpu in program.local_circuits}

    with pytest.raises(ValueError, match="no marker for remote operation"):
        remap_remote_ops_to_routed(program.remote_ops, stripped)


def test_remap_validates_its_inputs() -> None:
    from quport.distributed import remap_remote_ops_to_routed

    with pytest.raises(ValueError, match="local_routed must be a mapping"):
        remap_remote_ops_to_routed([], [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a QuantumCircuit"):
        remap_remote_ops_to_routed([], {0: object()})  # type: ignore[dict-item]


def test_compile_distributed_ships_a_manifest_matching_its_routed_programs() -> None:
    """End to end: the bundle `compile-dist` writes must be self-consistent."""
    from quport.compiler import compile_distributed
    from quport.pipeline import random_benchmark_circuit

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=4,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        optimization_level=1,
    )
    result = compile_distributed(random_benchmark_circuit(9, 14, 4), cfg, seed=4)

    assert result.program.remote_ops, "the fixture must produce remote operations"
    assert len(result.routed_remote_ops) == len(result.program.remote_ops)
    # A line intra-topology forces routing to permute, so the two manifests
    # must actually differ -- otherwise this test would pass vacuously.
    assert result.routed_remote_ops != result.program.remote_ops

    markers: dict[tuple[int, int], int] = {}
    for qpu, circuit in result.local_routed.items():
        ordinal = 0
        for instruction in circuit.data:
            label = getattr(instruction.operation, "label", None)
            if isinstance(label, str) and label.startswith("quport_remote_"):
                ordinal = int(label.removeprefix("quport_remote_"))
                markers[(ordinal, qpu)] = circuit.find_bit(instruction.qubits[0]).index

    for ordinal, op in enumerate(result.routed_remote_ops):
        assert op.q0_phys == markers[(ordinal, op.qpu0)]
        assert op.q1_phys == markers[(ordinal, op.qpu1)]
        # Both operands still belong to the QPU the operation names.
        arch = MultiQPUArchitecture(cfg)
        assert arch.qpu_of_phys(op.q0_phys) == op.qpu0
        assert arch.qpu_of_phys(op.q1_phys) == op.qpu1


def _wide_gate_arch(n_qpus: int, compute: int) -> MultiQPUArchitecture:
    return MultiQPUArchitecture(
        MultiQPUConfig(
            n_qpus=n_qpus,
            compute_qubits_per_qpu=compute,
            comm_qubits_per_qpu=1,
            intra_topology="clique",
            inter_topology="switch",
        )
    )


@pytest.mark.parametrize(
    ("n_qpus", "compute", "operands"),
    [
        # Three operands on three QPUs: two of the markers sit on QPUs the
        # two-endpoint manifest never names.
        (3, 1, (0, 2, 4)),
        # Three operands on two QPUs: one marker covers two of them, so the
        # merge has to take them in operand order rather than one per QPU.
        (2, 2, (0, 1, 3)),
    ],
)
def test_wide_remote_operations_survive_the_round_trip(
    n_qpus: int, compute: int, operands: tuple[int, int, int]
) -> None:
    """An operation on three qubits spanning QPUs must reassemble exactly.

    ``split_into_qpus`` supports operations wider than two qubits, marking every
    participating QPU. Reassembly therefore has to rebuild the whole operand
    list from those markers -- the manifest names only two endpoints -- and
    retire every marker, including the ones on QPUs the manifest never mentions.
    """
    from quport.distributed import reassemble_distributed_program
    from quport.protocol import verify_distributed_program

    arch = _wide_gate_arch(n_qpus, compute)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.h(operands[0])
    mapped.ccx(*operands)
    mapped.h(operands[2])

    program = split_into_qpus(mapped, arch)
    assert len(program.remote_ops) == 1

    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch, restore_layout=False
    )

    rebuilt = [
        (
            instruction.operation.name,
            [merged.find_bit(q).index for q in instruction.qubits],
        )
        for instruction in merged.data
    ]
    assert ("ccx", list(operands)) in rebuilt
    assert verify_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch
    )


def test_reassembly_rejects_a_manifest_from_other_programs() -> None:
    """The markers are the record of where routing left an operand.

    A manifest paired with programs it was not written for names a qubit the
    markers put elsewhere. Merging it would silently wire the remote gate to the
    wrong state, so it is reported instead.
    """
    from dataclasses import replace

    from quport.distributed import reassemble_distributed_program

    arch = _wide_gate_arch(2, 2)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.cx(0, 3)

    program = split_into_qpus(mapped, arch)
    stale = [replace(program.remote_ops[0], q1_phys=4)]

    with pytest.raises(ValueError, match="does not match these programs"):
        reassemble_distributed_program(
            mapped, program.local_circuits, stale, arch, restore_layout=False
        )


def test_reassembly_accepts_programs_read_back_from_qasm() -> None:
    """A program reloaded from the shipped QASM must not crash the merge.

    OpenQASM 3 writes a routed program over physical qubits, and reading one
    back gives a circuit carrying a synthetic ``layout`` built from loose
    qubits -- Qiskit's own ``final_index_layout`` raises on it. That layout
    records no routing permutation, because the file never held one, so it has
    to be treated like the missing layout of an unrouted program rather than
    bringing the merge down.
    """
    # Reading OpenQASM 3 back needs an importer no extra installs, so the
    # round trip runs only where it is present. Nothing else in this file
    # depends on it.
    pytest.importorskip("qiskit_qasm3_import")
    from qiskit import qasm3

    from quport.compiler import compile_distributed
    from quport.distributed import (
        reassemble_distributed_program,
        remote_barrier_label,
    )
    from quport.pipeline import random_benchmark_circuit
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="clique",
        optimization_level=0,
    )
    arch = MultiQPUArchitecture(cfg)
    result = compile_distributed(random_benchmark_circuit(8, 10, 3), cfg, seed=3)
    assert result.routed_remote_ops, "the fixture must produce remote operations"

    # Emitted QASM carries no barrier labels, so a consumer restores them from
    # the manifest's marker positions -- which is what those fields are for.
    reloaded: dict[int, QuantumCircuit] = {}
    for qpu, circuit in result.local_routed.items():
        wanted = {
            getattr(op, f"qpu{side}_marker"): ordinal
            for ordinal, op in enumerate(result.routed_remote_ops)
            for side in (0, 1)
            if getattr(op, f"qpu{side}") == qpu
        }
        loaded = qasm3.loads(qasm3.dumps(circuit))
        assert loaded.layout is not None, "the fixture must exercise the bad layout"
        rebuilt = loaded.copy_empty_like()
        seen = 0
        for instruction in loaded.data:
            operation = instruction.operation
            if getattr(operation, "_directive", False):
                ordinal = wanted.get(seen)
                seen += 1
                if ordinal is not None:
                    operation = operation.copy()
                    operation.label = remote_barrier_label(ordinal)
            rebuilt.append(operation, instruction.qubits, instruction.clbits)
        reloaded[qpu] = rebuilt

    merged = reassemble_distributed_program(
        result.physical_circuit, reloaded, result.routed_remote_ops, arch
    )
    assert merged.size() > 0

    # A clique permutes nothing, so the reloaded programs still describe the
    # mapped circuit exactly.
    assert verify_distributed_program(
        result.physical_circuit, reloaded, result.routed_remote_ops, arch
    )


def _instruction_trace(
    circuit: QuantumCircuit,
) -> list[tuple[str, list[int], list[int]]]:
    return [
        (
            instruction.operation.name,
            [circuit.find_bit(q).index for q in instruction.qubits],
            [circuit.find_bit(c).index for c in instruction.clbits],
        )
        for instruction in circuit.data
    ]


def test_reassembly_keeps_classical_feedforward_across_qpus_in_order() -> None:
    """A bit measured on one QPU and read on another must keep its order.

    Neither program records the other's accesses to a shared classical bit, so
    qubit dataflow alone lets the merge emit the conditional on QPU 0 before
    the measurement on QPU 1 that it reads -- and the later overwrite of that
    bit on QPU 0 before the measurement as well, changing its final value.
    """
    from quport.distributed import reassemble_distributed_program

    arch = _wide_gate_arch(2, 2)  # QPU 0 holds qubits 0-2, QPU 1 holds 3-5
    creg = ClassicalRegister(1, "c")
    mapped = QuantumCircuit(QuantumRegister(arch.n_phys, "q"), creg)
    mapped.h(3)
    mapped.measure(3, 0)
    with mapped.if_test((creg, 1)):
        mapped.x(0)
    mapped.measure(0, 0)

    program = split_into_qpus(mapped, arch)
    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch, restore_layout=False
    )

    assert _instruction_trace(merged) == _instruction_trace(mapped)


def test_reassembly_orders_remote_operations_by_their_classical_bits() -> None:
    """A conditioned remote gate must sit between the accesses around it.

    The measurement on QPU 0 that follows the remote gate shares no qubit with
    its marker, so only the classical bit holds it back.
    """
    from quport.distributed import reassemble_distributed_program

    arch = _wide_gate_arch(2, 2)
    creg = ClassicalRegister(1, "c")
    mapped = QuantumCircuit(QuantumRegister(arch.n_phys, "q"), creg)
    mapped.h(3)
    mapped.measure(3, 0)
    with mapped.if_test((creg, 1)):
        mapped.cx(0, 4)
    mapped.measure(1, 0)

    program = split_into_qpus(mapped, arch)
    assert len(program.remote_ops) == 1
    assert program.remote_ops[0].clbits == (0,)

    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch, restore_layout=False
    )

    assert _instruction_trace(merged) == _instruction_trace(mapped)


def test_routed_reassembly_keeps_classical_feedforward_in_order() -> None:
    """The order survives local routing, which reorders each program freely."""
    from quport.compiler import compile_distributed
    from quport.distributed import reassemble_distributed_program

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        inter_topology="switch",
        optimization_level=0,
    )
    creg = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(QuantumRegister(6, "q"), creg)
    # Two tight clusters, which the cluster strategy puts on different QPUs,
    # and one gate between them so there is a remote operation to route around.
    for a, b, c in ((0, 1, 2), (3, 4, 5)):
        circuit.h(a)
        circuit.cx(a, b)
        circuit.cx(b, c)
        circuit.cx(a, c)
    circuit.cx(2, 5)
    circuit.measure(3, 0)
    with circuit.if_test((creg, 1)):
        circuit.x(1)
    circuit.measure(1, 0)

    result = compile_distributed(circuit, cfg, strategy="cluster", seed=0)
    arch = MultiQPUArchitecture(cfg)
    physical = result.physical_circuit
    readers = [
        arch.qpu_of_phys(physical.find_bit(instruction.qubits[0]).index)
        for instruction in physical.data
        if instruction.clbits
    ]
    assert readers[0] != readers[1], "the fixture must feed forward across QPUs"
    assert result.routed_remote_ops, "the fixture must produce remote operations"
    merged = reassemble_distributed_program(
        result.physical_circuit,
        result.local_routed,
        result.routed_remote_ops,
        arch,
        restore_layout=False,
    )

    def classical(trace: list[tuple[str, list[int], list[int]]]) -> list[str]:
        return [name for name, _qubits, clbits in trace if clbits]

    expected = classical(_instruction_trace(result.physical_circuit))
    assert expected == ["measure", "if_else", "measure"]
    assert classical(_instruction_trace(merged)) == expected


def test_marker_indices_count_barriers_nested_in_control_flow() -> None:
    """The emitted program inlines an ``if`` body, barriers and all.

    A consumer counting barrier statements in the OpenQASM file must land on
    the marker; skipping the nested barrier pointed it one statement early, at
    a barrier on another qubit.
    """
    from qiskit import qasm3

    from quport.distributed import remap_remote_ops_to_routed

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    mapped = QuantumCircuit(arch.n_phys, 1)
    mapped.h(1)
    mapped.measure(1, 0)
    with mapped.if_test((mapped.clbits[0], 1)):
        mapped.x(1)
        mapped.barrier(1)
    mapped.cx(0, 3)

    program = split_into_qpus(mapped, arch)
    marker = program.remote_ops[0].qpu0_marker
    statements = [
        line.strip()
        for line in qasm3.dumps(program.local_circuits[0]).splitlines()
        if line.strip().startswith("barrier")
    ]

    assert marker == 1
    assert statements[marker] == "barrier q[0];"
    remapped = remap_remote_ops_to_routed(program.remote_ops, program.local_circuits)
    assert remapped[0].qpu0_marker == marker


def test_remap_counts_barriers_and_not_every_directive() -> None:
    """A classical store is a directive too, but it is not a barrier."""
    from qiskit.circuit.classical import expr

    from quport.distributed import remap_remote_ops_to_routed

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    mapped = QuantumCircuit(arch.n_phys, 1)
    mapped.h(0)
    mapped.cx(0, 3)
    program = split_into_qpus(mapped, arch)

    # A store reaches a program only if someone edits it in, since the split
    # refuses one; the marker count must still skip it.
    edited: dict[int, QuantumCircuit] = {}
    for qpu, circuit in program.local_circuits.items():
        rebuilt = circuit.copy_empty_like()
        rebuilt.store(rebuilt.clbits[0], expr.lift(True))
        for instruction in circuit.data:
            rebuilt.append(instruction)
        edited[qpu] = rebuilt

    remapped = remap_remote_ops_to_routed(program.remote_ops, edited)
    assert (remapped[0].qpu0_marker, remapped[0].qpu1_marker) == (0, 0)


def _with_classical_storage(kind: str, n_qubits: int) -> QuantumCircuit:
    from qiskit.circuit.classical import expr

    circuit = QuantumCircuit(n_qubits, 1)
    circuit.h(0)
    circuit.cx(0, 3)
    if kind == "store":
        circuit.store(circuit.clbits[0], expr.lift(True))
    elif kind == "var":
        circuit.add_var("flag", expr.lift(False))
    else:
        circuit.add_stretch("gap")
    return circuit


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("store", "classical store"),
        ("var", "classical variables"),
        ("stretch", "stretches"),
    ],
)
def test_split_and_reassembly_refuse_classical_storage(kind: str, message: str) -> None:
    """A store was broadcast to every program and then dropped by the merge."""
    from quport.distributed import reassemble_distributed_program

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    mapped = _with_classical_storage(kind, arch.n_phys)

    with pytest.raises(ValueError, match=message):
        split_into_qpus(mapped, arch)
    with pytest.raises(ValueError, match=message):
        reassemble_distributed_program(mapped, {}, [], arch)


def test_a_classical_register_called_q_survives_the_round_trip() -> None:
    """The per-QPU programs add a quantum register of their own, also ``q``."""
    from qiskit import qasm3

    from quport.distributed import reassemble_distributed_program

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    mapped = QuantumCircuit(
        QuantumRegister(arch.n_phys, "data"), ClassicalRegister(1, "q")
    )
    mapped.h(0)
    mapped.cx(0, 3)
    mapped.measure(0, 0)

    program = split_into_qpus(mapped, arch)
    for circuit in program.local_circuits.values():
        qasm3.dumps(circuit)
    merged = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch, restore_layout=False
    )

    assert _instruction_trace(merged) == _instruction_trace(mapped)
    assert [register.name for register in merged.cregs] == ["q"]


def test_a_bad_manifest_leaves_an_existing_bundle_untouched(tmp_path: Path) -> None:
    """Every artifact is validated before the first one is written."""
    from quport.distributed import RemoteOp, write_distributed_program

    bundle = tmp_path / "bundle"
    write_distributed_program(
        DistributedProgram(local_circuits={0: QuantumCircuit(2)}, remote_ops=[]),
        bundle,
    )
    before = {path.name: path.read_text() for path in bundle.iterdir()}

    changed = QuantumCircuit(2)
    changed.h(0)
    bad = DistributedProgram(
        local_circuits={0: changed},
        remote_ops=[
            RemoteOp(
                name="cx",
                q0_phys=0,
                q1_phys=0,
                qpu0=0,
                qpu1=1,
                params=(),
                clbits=(),
                index=0,
            )
        ],
    )
    with pytest.raises(ValueError, match="distinct"):
        write_distributed_program(bad, bundle)

    assert {path.name: path.read_text() for path in bundle.iterdir()} == before


def test_routed_reassembly_accepts_a_decomposed_instruction_writing_a_bit_twice() -> (
    None
):
    """Routing turns one ``parity_readout`` into two accesses to its bit.

    The order across QPUs is read from ``mapped``, which lists the instruction
    once; a QPU whose accesses to a bit form a single run owns all of them.
    """
    from qiskit import transpile

    from quport.distributed import (
        reassemble_distributed_program,
        remap_remote_ops_to_routed,
    )

    cfg = MultiQPUConfig(
        n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1, intra_topology="line"
    )
    arch = MultiQPUArchitecture(cfg)
    readout = QuantumCircuit(2, 1, name="parity_readout")
    readout.measure(0, 0)
    readout.measure(1, 0)
    mapped = QuantumCircuit(arch.n_phys, 1)
    mapped.h(0)
    mapped.cx(0, 3)
    mapped.append(readout.to_instruction(), [0, 1], [0])

    program = split_into_qpus(mapped, arch)
    routed = {
        qpu: transpile(
            circuit,
            coupling_map=arch.build_intra_coupling_map(qpu),
            initial_layout=list(range(arch.n_phys)),
            basis_gates=list(cfg.basis_gates),
            optimization_level=0,
            layout_method="trivial",
            seed_transpiler=0,
        )
        for qpu, circuit in program.local_circuits.items()
    }
    assert [i.operation.name for i in routed[0].data].count("measure") == 2

    merged = reassemble_distributed_program(
        mapped, routed, remap_remote_ops_to_routed(program.remote_ops, routed), arch
    )

    assert merged.count_ops()["measure"] == 2
    measured = [
        merged.find_bit(instruction.qubits[0]).index
        for instruction in merged.data
        if instruction.operation.name == "measure"
    ]
    assert sorted(measured) == [0, 1]


def _route_locally(program: DistributedProgram, cfg: MultiQPUConfig) -> dict[int, Any]:
    from qiskit import transpile

    arch = MultiQPUArchitecture(cfg)
    return {
        qpu: transpile(
            circuit,
            coupling_map=arch.build_intra_coupling_map(qpu),
            initial_layout=list(range(arch.n_phys)),
            basis_gates=list(cfg.basis_gates),
            optimization_level=0,
            layout_method="trivial",
            seed_transpiler=0,
        )
        for qpu, circuit in program.local_circuits.items()
    }


def _clbit_trace(circuit: QuantumCircuit) -> list[tuple[str, int]]:
    return [
        (instruction.operation.name, circuit.find_bit(instruction.qubits[0]).index)
        for instruction in circuit.data
        if instruction.clbits
    ]


@pytest.mark.parametrize("later", ["read", "declared_only"])
def test_routed_reassembly_orders_a_decomposed_instruction_among_several_runs(
    later: str,
) -> None:
    """The readout's two measurements both precede QPU 1's read of the bit.

    With a second run of accesses on QPU 0, counting the decomposed readout as
    one access either stalled the merge ("read") or -- when QPU 0's later
    instruction only declares the bit and so loses it in decomposition --
    silently put QPU 1's read between the two measurements.
    """
    from quport.distributed import (
        reassemble_distributed_program,
        remap_remote_ops_to_routed,
    )

    cfg = MultiQPUConfig(
        n_qpus=2,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        basis_gates=("rz", "sx", "x", "cx", "if_else"),
    )
    arch = MultiQPUArchitecture(cfg)
    readout = QuantumCircuit(2, 1, name="parity_readout")
    readout.measure(0, 0)
    readout.measure(1, 0)
    mapped = QuantumCircuit(arch.n_phys, 1)
    mapped.h(0)
    mapped.cx(0, 3)
    mapped.append(readout.to_instruction(), [0, 1], [0])
    with mapped.if_test((mapped.clbits[0], 1)):
        mapped.x(4)
    if later == "read":
        with mapped.if_test((mapped.clbits[0], 1)):
            mapped.x(2)
    else:
        flagged = QuantumCircuit(1, 1, name="flagged_x")
        flagged.x(0)
        mapped.append(flagged.to_instruction(), [2], [0])

    program = split_into_qpus(mapped, arch)
    routed = _route_locally(program, cfg)
    merged = reassemble_distributed_program(
        mapped,
        routed,
        remap_remote_ops_to_routed(program.remote_ops, routed),
        arch,
        restore_layout=False,
    )

    expected = [("measure", 0), ("measure", 1), ("if_else", 4)]
    if later == "read":
        expected.append(("if_else", 2))
    assert _clbit_trace(merged) == expected

    # Unrouted programs keep the instructions whole and merge as before.
    unrouted = reassemble_distributed_program(
        mapped, program.local_circuits, program.remote_ops, arch, restore_layout=False
    )
    assert _instruction_trace(unrouted) == _instruction_trace(mapped)


@pytest.mark.parametrize("field", ["name", "params"])
def test_an_unencodable_manifest_leaves_an_existing_bundle_untouched(
    tmp_path: Path, field: str
) -> None:
    """A lone surrogate passes every check on a str and fails only on encoding."""
    from quport.distributed import RemoteOp, write_distributed_program

    bundle = tmp_path / "bundle"
    write_distributed_program(
        DistributedProgram(local_circuits={0: QuantumCircuit(2)}, remote_ops=[]),
        bundle,
    )
    before = {path.name: path.read_text() for path in bundle.iterdir()}

    changed = QuantumCircuit(2)
    changed.h(0)
    op = RemoteOp(
        name="\ud800" if field == "name" else "cx",
        q0_phys=0,
        q1_phys=1,
        qpu0=0,
        qpu1=1,
        params=("\ud800",) if field == "params" else (),
        clbits=(),
        index=0,
    )
    with pytest.raises(ValueError, match="cannot be encoded as UTF-8"):
        write_distributed_program(
            DistributedProgram(local_circuits={0: changed}, remote_ops=[op]), bundle
        )

    assert {path.name: path.read_text() for path in bundle.iterdir()} == before


def test_reassembly_validates_its_manifest_entries() -> None:
    """A manifest read back from JSON, or naming other instructions, crashed deep inside."""
    import dataclasses
    import json

    from quport.distributed import reassemble_distributed_program

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    mapped = QuantumCircuit(arch.n_phys)
    mapped.h(0)
    mapped.cx(0, 3)
    mapped.cx(1, 4)
    program = split_into_qpus(mapped, arch)
    ops = program.remote_ops

    cases = [
        (json.loads(json.dumps([op.to_dict() for op in ops])), "must be a RemoteOp"),
        ([dataclasses.replace(ops[0], index=10**6), ops[1]], "does not have"),
        ([ops[0], dataclasses.replace(ops[1], index=ops[0].index)], "earlier entry"),
        (iter(ops), "sequence of RemoteOp"),
    ]
    for manifest, message in cases:
        with pytest.raises(ValueError, match=message):
            reassemble_distributed_program(
                mapped, program.local_circuits, manifest, arch  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="QPU id must be an integer"):
        reassemble_distributed_program(
            mapped, {**program.local_circuits, "x": QuantumCircuit(1)}, ops, arch  # type: ignore[dict-item]
        )


def _cx_named(name: str) -> Any:
    body = QuantumCircuit(2)
    body.cx(0, 1)
    gate = body.to_gate()
    gate.name = name
    return gate


def test_a_gate_named_barrier_is_split_as_a_gate() -> None:
    """Split replaced it with a real barrier: the gate vanished from every
    program and its remote operation from the manifest."""
    from quport.compiler import compile_distributed
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    arch = MultiQPUArchitecture(cfg)
    mapped = QuantumCircuit(arch.n_phys)
    mapped.h(0)
    mapped.append(_cx_named("barrier"), [0, 2])

    program = split_into_qpus(mapped, arch)

    assert [op.name for op in program.remote_ops] == ["barrier"]
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(2)
    for _ in range(3):
        qc.cx(0, 1)
        qc.cx(2, 3)
    qc.append(_cx_named("barrier"), [1, 2])
    res = compile_distributed(qc, cfg, seed=0, strategy="balanced")
    assert verify_distributed_program(
        res.physical_circuit, res.local_routed, res.routed_remote_ops, arch
    )


def test_a_gate_named_store_compiles_as_a_gate() -> None:
    """It was refused as a classical store, so a valid OpenQASM 2 program
    defining ``gate store`` failed in compile-dist, ebits, optimal and migrate."""
    from qiskit import qasm2

    from quport.compiler import compile_distributed
    from quport.protocol import verify_distributed_program

    cfg = MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    qc = qasm2.loads(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate store a { x a; }\nqreg q[2];\n'
        "creg c[2];\nh q[0];\ncx q[0], q[1];\nstore q[1];\nmeasure q -> c;\n"
    )

    res = compile_distributed(qc, cfg, seed=0, strategy="balanced")

    assert verify_distributed_program(
        res.physical_circuit,
        res.local_routed,
        res.routed_remote_ops,
        MultiQPUArchitecture(cfg),
    )


def test_a_store_inside_a_control_flow_body_is_refused() -> None:
    """Only top-level stores were refused: a nested toggle was copied into every
    program, so it ran once per QPU, and reassembly then dropped it."""
    from qiskit.circuit.classical import expr

    from quport.distributed import reassemble_distributed_program

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    qc = QuantumCircuit(arch.n_phys, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        with qc.if_test((qc.clbits[0], 1)):
            qc.store(qc.clbits[1], expr.logic_not(qc.clbits[1]))
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(3)

    with pytest.raises(ValueError, match="classical store"):
        split_into_qpus(qc, arch)
    with pytest.raises(ValueError, match="classical store"):
        reassemble_distributed_program(qc, {}, [], arch)


def _cross_qpu_compile(qc: QuantumCircuit, **settings: Any) -> Any:
    from quport.compiler import compile_distributed

    cfg = MultiQPUConfig(**settings)
    return compile_distributed(qc, cfg, seed=0, strategy="balanced"), (
        MultiQPUArchitecture(cfg)
    )


def test_routing_keeps_a_local_access_after_the_remote_read_it_follows() -> None:
    """Routing moved the measurement that overwrites c[0] ahead of the marker
    of the remote ``if`` that reads it, and chained them with its swaps: the
    routed programs demanded the bit be overwritten before it was read."""
    from quport.distributed import reassemble_distributed_program

    qc = QuantumCircuit(6, 1)
    qc.cx(0, 1)
    with qc.if_test((qc.clbits[0], 1)):  # spans both QPUs and reads c[0]
        qc.cx(1, 4)
        qc.x(2)
    qc.measure(5, 0)  # local, and overwrites c[0] after the read
    qc.cx(5, 1)
    res, arch = _cross_qpu_compile(
        qc,
        n_qpus=2,
        compute_qubits_per_qpu=3,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        optimization_level=3,
    )

    reassemble_distributed_program(
        res.physical_circuit, res.local_routed, res.routed_remote_ops, arch
    )
    for routed in res.local_routed.values():
        assert not any(
            instruction.operation.label == "quport_order"
            for instruction in routed.data
            if instruction.operation.name == "barrier"
        )


def test_routing_holds_cross_qpu_events_in_order_and_then_lets_go() -> None:
    from qiskit.circuit import Barrier

    from quport.distributed import _hold_cross_qpu_order, _release_cross_qpu_order

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=3, comm_qubits_per_qpu=1)
    )
    mapped = QuantumCircuit(arch.n_phys, 3)
    mapped.measure(0, 0)  # read on QPU 1 below: shared
    mapped.measure(1, 1)  # read on QPU 1 below: shared
    mapped.measure(2, 2)  # nobody else reads it: private
    with mapped.if_test((mapped.clbits[0], 1)):
        mapped.x(4)
    with mapped.if_test((mapped.clbits[1], 1)):
        mapped.x(5)
    program = split_into_qpus(mapped, arch)

    held = _hold_cross_qpu_order(program, mapped, arch)

    def shape(circuit: QuantumCircuit) -> list[tuple[str, list[int]]]:
        return [
            (i.operation.name, [circuit.find_bit(q).index for q in i.qubits])
            for i in circuit.data
        ]

    assert shape(held[0]) == [
        ("measure", [0]),
        ("barrier", [0, 1]),
        ("measure", [1]),
        ("measure", [2]),
    ]
    assert shape(held[1]) == [("if_else", [4]), ("barrier", [4, 5]), ("if_else", [5])]
    order = [i.operation for i in held[0].data if isinstance(i.operation, Barrier)]
    assert [barrier.label for barrier in order] == ["quport_order"]
    assert shape(_release_cross_qpu_order(held[0])) == shape(program.local_circuits[0])


def test_reassembly_keeps_a_block_routing_emptied_of_qubits() -> None:
    """Routing contracts a block whose body cancels to no qubits; reassembly
    skipped every instruction without qubits, so it dropped the block, or
    stalled when another QPU read the same bit afterwards."""
    from quport.distributed import reassemble_distributed_program

    settings = {"n_qpus": 2, "compute_qubits_per_qpu": 2, "comm_qubits_per_qpu": 1}
    single = QuantumCircuit(4, 2)
    single.h(0)
    single.measure(0, 0)
    with single.if_test((single.clbits[0], 1)):
        single.x(0)
        single.x(0)
    single.cx(0, 3)
    single.cx(1, 2)

    shared = QuantumCircuit(4, 2)
    shared.h(0)
    shared.measure(0, 0)
    with shared.if_test((shared.clbits[0], 1)):
        shared.x(1)
        shared.x(1)
    shared.cx(1, 2)
    shared.cx(2, 3)
    shared.cx(0, 3)
    shared.measure(1, 1)
    with shared.if_test((shared.clbits[0], 1)):
        shared.x(3)

    for qc in (single, shared):
        res, arch = _cross_qpu_compile(qc, optimization_level=3, **settings)
        emptied = [
            instruction
            for routed in res.local_routed.values()
            for instruction in routed.data
            if instruction.operation.name == "if_else" and not instruction.qubits
        ]
        assert emptied, "the fixture must have routing contract a block"

        merged = reassemble_distributed_program(
            res.physical_circuit, res.local_routed, res.routed_remote_ops, arch
        )

        assert merged.count_ops()["if_else"] == qc.count_ops()["if_else"]


def test_reassembly_refuses_a_store_that_routing_unrolled_into_a_program() -> None:
    """A custom instruction can carry a store in its definition; once routing
    unrolled it, reassembly skipped the store and the toggle was lost."""
    from qiskit import transpile
    from qiskit.circuit.classical import expr

    from quport.distributed import reassemble_distributed_program

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=2, compute_qubits_per_qpu=2, comm_qubits_per_qpu=1)
    )
    body = QuantumCircuit(1, 1)
    body.measure(0, 0)
    body.store(body.clbits[0], expr.logic_not(body.clbits[0]))
    readflip = body.to_instruction()
    readflip.name = "readflip"
    mapped = QuantumCircuit(arch.n_phys, 1)
    mapped.h(0)
    mapped.append(readflip, [0], [0])
    mapped.cx(0, 3)
    program = split_into_qpus(mapped, arch)
    routed = {
        qpu: transpile(local, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
        for qpu, local in program.local_circuits.items()
    }

    with pytest.raises(ValueError, match="classical store"):
        reassemble_distributed_program(mapped, routed, program.remote_ops, arch)


def test_a_remote_operation_names_every_operand_and_marker() -> None:
    """The manifest named only two QPUs, so the marker and operand on a third
    could not be found from the files, and nothing held the operation there."""
    import json
    import re

    from qiskit import qasm3

    from quport.compiler import compile_distributed
    from quport.distributed import remap_remote_ops_to_routed

    arch = MultiQPUArchitecture(
        MultiQPUConfig(n_qpus=3, compute_qubits_per_qpu=1, comm_qubits_per_qpu=1)
    )
    mapped = QuantumCircuit(arch.n_phys)
    mapped.h(4)
    mapped.ccx(0, 2, 4)  # QPUs 0, 1 and 2
    entry = split_into_qpus(mapped, arch).remote_ops[0].to_dict()

    assert entry["operands"] == [
        {"qpu": 0, "phys": 0},
        {"qpu": 1, "phys": 2},
        {"qpu": 2, "phys": 4},
    ]
    assert entry["markers"] == [
        {"qpu": 0, "marker": 0},
        {"qpu": 1, "marker": 0},
        {"qpu": 2, "marker": 0},
    ]

    cfg = MultiQPUConfig(
        n_qpus=3,
        compute_qubits_per_qpu=2,
        comm_qubits_per_qpu=1,
        intra_topology="line",
        optimization_level=1,
    )
    qc = QuantumCircuit(6, 1)
    for a, b in ((0, 1), (2, 3), (4, 5)):
        for _ in range(4):
            qc.cx(a, b)
    qc.h(0)
    qc.measure(0, 0)
    qc.barrier(5)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
        qc.x(3)
        qc.x(5)
    res = compile_distributed(qc, cfg, seed=1, strategy="balanced")
    wide = [op for op in res.routed_remote_ops if len(op.markers) == 3]
    assert wide, "the fixture must have a block across three QPUs"
    statements = {
        qpu: [
            [int(n) for n in re.findall(r"\$(\d+)", arguments)]
            for arguments in re.findall(r"barrier ([^;]*);", qasm3.dumps(circuit))
        ]
        for qpu, circuit in res.local_routed.items()
    }
    for op in wide:
        entry = json.loads(json.dumps(op.to_dict()))
        for marker in entry["markers"]:
            mine = [o["phys"] for o in entry["operands"] if o["qpu"] == marker["qpu"]]
            barrier = statements[marker["qpu"]][marker["marker"]]
            assert barrier[: len(mine)] == mine
    assert remap_remote_ops_to_routed(res.routed_remote_ops, res.local_routed) == (
        res.routed_remote_ops
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"operands": ((0, 1), (1, 3))}, "start with"),
        ({"operands": ((0, 0), (0, 0), (1, 3))}, "distinct"),
        ({"operands": ((0, 0), (1, 3, 4))}, "pair"),
        ({"markers": ((0, 0), (0, 1))}, "each QPU once"),
        ({"markers": ((1, 0), (0, 0))}, "order"),
        ({"markers": ((0, 2), (1, 0))}, "qpu0_marker"),
    ],
)
def test_remote_op_refuses_operands_or_markers_that_disagree(
    change: dict[str, Any], message: str
) -> None:
    import dataclasses

    from quport.distributed import RemoteOp

    op = RemoteOp(
        name="cx",
        q0_phys=0,
        q1_phys=3,
        qpu0=0,
        qpu1=1,
        params=(),
        clbits=(),
        index=0,
        qpu0_marker=0,
        qpu1_marker=0,
        operands=((0, 0), (1, 3)),
        markers=((0, 0), (1, 0)),
    )
    op.to_dict()

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(op, **change).to_dict()
