# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import warnings

import pytest

pytest.importorskip("qiskit")

import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.library import (
    CCXGate,
    CCZGate,
    CHGate,
    CPhaseGate,
    CRZGate,
    CSwapGate,
    CXGate,
    CZGate,
    ECRGate,
    RZXGate,
    RZZGate,
    SwapGate,
    XGate,
    standard_gates,
)
from qiskit.quantum_info import Operator

from quport.entanglement import (
    DIAGONAL_1Q_GATES,
    acts_diagonally,
    breaks_cat_copy,
    diagonal_operands,
    diagonal_positions,
    is_directive,
)


def _z_operator(position: int, num_qubits: int) -> np.ndarray:
    """Build ``Z`` on one qubit in Qiskit's little-endian operator convention."""
    single = [
        (
            np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
            if index == position
            else np.eye(2, dtype=complex)
        )
        for index in range(num_qubits)
    ]
    out = np.array([[1.0]], dtype=complex)
    for index in reversed(range(num_qubits)):
        out = np.kron(out, single[index])
    return out


def _commuting_positions(gate: Gate) -> set[int]:
    """Positions on which the gate's unitary genuinely commutes with ``Z``."""
    unitary = Operator(gate).data
    out: set[int] = set()
    for position in range(gate.num_qubits):
        z = _z_operator(position, gate.num_qubits)
        if np.allclose(unitary @ z, z @ unitary, atol=1e-10):
            out.add(position)
    return out


def _instantiable_standard_gates() -> list[Gate]:
    """Every standard-library gate that can be built with placeholder angles."""
    gates: list[Gate] = []
    seen: set[str] = set()
    for _name, obj in sorted(vars(standard_gates).items()):
        if not inspect.isclass(obj) or not issubclass(obj, Gate) or obj is Gate:
            continue
        parameters = list(inspect.signature(obj.__init__).parameters.items())[1:]
        kwargs: dict[str, object] = {}
        constructible = True
        for name, parameter in parameters:
            if parameter.default is not inspect.Parameter.empty or parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                continue
            if name in ("theta", "phi", "lam", "gamma", "beta"):
                kwargs[name] = 0.7
            elif name == "num_ctrl_qubits":
                kwargs[name] = 2
            else:
                constructible = False
        if not constructible:
            continue
        try:
            with warnings.catch_warnings():
                # Some multi-controlled X synthesis classes are deprecated but
                # still worth checking while they exist.
                warnings.simplefilter("ignore", DeprecationWarning)
                gate = obj(**kwargs)
            if gate.num_qubits > 4:
                continue
            Operator(gate)
        except Exception:  # pragma: no cover - defensive against library churn
            continue
        if gate.name in seen:
            continue
        seen.add(gate.name)
        gates.append(gate)
    return gates


def test_reported_diagonal_positions_are_never_unsound() -> None:
    """Every claimed diagonal operand must really commute with ``Z``.

    This is the safety property the whole entanglement stack rests on: claiming
    a cat copy survives an operation that in fact flips its root's
    computational-basis label would silently produce wrong circuits. Under-
    reporting is allowed (it only costs extra EPR pairs); over-reporting is not.
    """
    gates = _instantiable_standard_gates()
    assert len(gates) > 30, "expected the standard gate library to be discoverable"

    for gate in gates:
        claimed = set(diagonal_positions(gate))
        actual = _commuting_positions(gate)
        assert claimed <= actual, (
            f"{gate.name}: claimed diagonal positions {sorted(claimed)} "
            f"exceed the commuting set {sorted(actual)}"
        )


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        (CXGate(), {0}),
        (CZGate(), {0, 1}),
        (CPhaseGate(0.3), {0, 1}),
        (CRZGate(0.3), {0, 1}),
        (CHGate(), {0}),
        (CCXGate(), {0, 1}),
        (CCZGate(), {0, 1, 2}),
        (CSwapGate(), {0}),
        (RZZGate(0.3), {0, 1}),
        (RZXGate(0.3), {0}),
        (SwapGate(), set()),
        (ECRGate(), set()),
        (XGate(), set()),
    ],
)
def test_diagonal_positions_match_the_exact_commuting_set(
    gate: Gate, expected: set[int]
) -> None:
    assert set(diagonal_positions(gate)) == expected
    assert _commuting_positions(gate) == expected


def test_every_listed_single_qubit_gate_really_is_diagonal() -> None:
    """The 1Q allow-list is checked against real unitaries, not just trusted."""
    from qiskit.circuit.library import (
        IGate,
        PhaseGate,
        RZGate,
        SdgGate,
        SGate,
        TdgGate,
        TGate,
        U1Gate,
        ZGate,
    )

    concrete: dict[str, Gate] = {
        "id": IGate(),
        "z": ZGate(),
        "s": SGate(),
        "sdg": SdgGate(),
        "t": TGate(),
        "tdg": TdgGate(),
        "rz": RZGate(0.7),
        "p": PhaseGate(0.7),
        "u1": U1Gate(0.7),
    }
    # Names in the table with no directly constructible standard gate ("i",
    # "delay") are covered by the table itself; the rest are verified here.
    assert set(concrete) <= DIAGONAL_1Q_GATES

    for name, gate in concrete.items():
        assert _commuting_positions(gate) == {0}, name
        assert set(diagonal_positions(gate)) == {0}, name


def test_acts_diagonally_and_helpers() -> None:
    assert acts_diagonally(CXGate(), 0)
    assert not acts_diagonally(CXGate(), 1)
    assert diagonal_operands(CXGate(), (7, 9)) == (7,)
    assert diagonal_operands(CZGate(), (7, 9)) == (7, 9)
    assert breaks_cat_copy(CXGate(), [1])
    assert not breaks_cat_copy(CXGate(), [0])
    assert not breaks_cat_copy(CZGate(), [0, 1])


def test_acts_diagonally_rejects_non_integer_positions() -> None:
    with pytest.raises(ValueError, match="position must be an integer"):
        acts_diagonally(CXGate(), True)
    with pytest.raises(ValueError, match="position must be an integer"):
        acts_diagonally(CXGate(), "0")  # type: ignore[arg-type]


def test_barriers_are_transparent_to_cat_copies() -> None:
    from qiskit.circuit import Barrier

    barrier = Barrier(3)
    assert is_directive(barrier)
    assert set(diagonal_positions(barrier)) == {0, 1, 2}
    assert not breaks_cat_copy(barrier, [0, 1, 2])


def test_unknown_operations_are_treated_conservatively() -> None:
    class _Opaque:
        name = "mystery_gate"
        num_qubits = 2

    assert diagonal_positions(_Opaque()) == frozenset()
    assert breaks_cat_copy(_Opaque(), [0])


def test_measure_and_reset_close_cat_copies() -> None:
    from qiskit.circuit import Measure, Reset

    assert diagonal_positions(Measure()) == frozenset()
    assert diagonal_positions(Reset()) == frozenset()
