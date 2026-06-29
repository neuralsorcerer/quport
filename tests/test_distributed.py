# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

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
