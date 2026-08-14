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
