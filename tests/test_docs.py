# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that keep the published documentation honest about the code."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("qiskit")

import quport
import quport.architecture
import quport.compiler
import quport.config
import quport.cost
import quport.distributed
import quport.interaction
import quport.metrics
import quport.pipeline
import quport.schedule

_API_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "api-references.md"
_SIGNATURE_START = re.compile(r"^([A-Za-z_]\w*)\(")


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not nested inside brackets."""
    parts: list[str] = []
    buffer = ""
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(buffer)
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        parts.append(buffer)
    return parts


def _public_callables() -> dict[str, Any]:
    modules = (
        quport,
        quport.architecture,
        quport.compiler,
        quport.config,
        quport.cost,
        quport.distributed,
        quport.interaction,
        quport.metrics,
        quport.pipeline,
        quport.schedule,
    )
    found: dict[str, Any] = {}
    for module in modules:
        for name, obj in vars(module).items():
            if callable(obj) and not name.startswith("_"):
                found.setdefault(name, obj)
    return found


def _documented_signatures() -> list[tuple[str, list[str]]]:
    source = _API_REFERENCE.read_text(encoding="utf-8")
    known = _public_callables()
    out: list[tuple[str, list[str]]] = []
    for block in re.findall(r"```python\n(.*?)```", source, re.S):
        first_line = block.strip().splitlines()[0]
        match = _SIGNATURE_START.match(first_line)
        if match is None or match.group(1) not in known:
            continue
        flattened = " ".join(line.strip() for line in block.strip().splitlines())
        flattened = flattened.split("->")[0]
        inner = flattened[flattened.index("(") + 1 : flattened.rindex(")")]
        params = [
            piece.split("=")[0].split(":")[0].strip()
            for piece in _split_top_level(inner)
        ]
        out.append((match.group(1), [p for p in params if p]))
    return out


@pytest.mark.skipif(
    not _API_REFERENCE.is_file(), reason="docs/ is not present in this checkout"
)
def test_api_reference_signatures_match_the_code() -> None:
    """Every signature listed in the API reference must match the real one.

    Signature listings rot silently: nothing imports them, so a renamed or
    reordered parameter leaves the published reference quietly wrong.
    """
    known = _public_callables()
    documented = _documented_signatures()
    assert len(documented) >= 15, "expected the reference to list many signatures"

    mismatches: list[str] = []
    for name, doc_params in documented:
        actual = [
            parameter
            for parameter in inspect.signature(known[name]).parameters
            if parameter != "self"
        ]
        if doc_params != actual:
            mismatches.append(f"{name}: documented {doc_params} != actual {actual}")

    assert not mismatches, "\n".join(mismatches)
