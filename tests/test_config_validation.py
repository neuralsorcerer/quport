# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

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
    kwargs = {field: value}
    with pytest.raises(ValueError, match=msg):
        MultiQPUConfig(**kwargs)


def test_rejects_non_boolean_async_classical() -> None:
    with pytest.raises(ValueError, match="async_classical must be a boolean"):
        MultiQPUConfig(async_classical=1)  # type: ignore[arg-type]


def test_rejects_non_tuple_basis_gates() -> None:
    with pytest.raises(ValueError, match="basis_gates must be a tuple"):
        MultiQPUConfig(basis_gates=["cx"])  # type: ignore[arg-type]


def test_rejects_empty_gate_name() -> None:
    with pytest.raises(
        ValueError, match="basis_gates entries must be a non-empty string"
    ):
        MultiQPUConfig(basis_gates=("cx", "   "))
