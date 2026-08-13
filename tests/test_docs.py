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


_DOCS_DIR = _API_REFERENCE.parent
_CONFIGURATION = _DOCS_DIR / "configuration.md"
_CLI_REFERENCE = _DOCS_DIR / "cli.md"
_README = _DOCS_DIR.parent / "README.md"

_TABLE_ROW = re.compile(r"^\|\s*`([A-Za-z_]\w*)`\s*\|\s*`([^`]*)`\s*\|")


def _documented_defaults(path: Path, cls: type) -> dict[str, str]:
    """Collect `| name | default |` rows naming a field of `cls`."""
    import dataclasses

    field_names = {field.name for field in dataclasses.fields(cls)}
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _TABLE_ROW.match(line)
        if match and match.group(1) in field_names:
            found[match.group(1)] = match.group(2)
    return found


def _actual_default(cls: type, name: str) -> Any:
    import dataclasses

    for field in dataclasses.fields(cls):
        if field.name == name:
            if field.default is not dataclasses.MISSING:
                return field.default
            if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                return field.default_factory()  # type: ignore[misc]
    raise AssertionError(f"{cls.__name__} has no field {name}")


@pytest.mark.skipif(
    not _CONFIGURATION.is_file(), reason="docs/ is not present in this checkout"
)
@pytest.mark.parametrize("cls_name", ["MultiQPUConfig", "LatencyModel"])
def test_documented_defaults_match_the_dataclasses(cls_name: str) -> None:
    """Every default printed in the configuration reference must be the real one.

    These tables are what a reader trusts when reproducing a benchmark, and a
    changed default leaves them wrong with nothing failing.
    """
    import ast
    import dataclasses

    cls = getattr(quport.config, cls_name)
    documented = _documented_defaults(_CONFIGURATION, cls)

    field_names = {field.name for field in dataclasses.fields(cls)}
    assert (
        set(documented) == field_names
    ), f"{cls_name}: documented {sorted(documented)} != fields {sorted(field_names)}"

    mismatches: list[str] = []
    for name, text in sorted(documented.items()):
        actual = _actual_default(cls, name)
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text
        if isinstance(actual, tuple) and isinstance(parsed, tuple):
            equal = list(parsed) == list(actual)
        else:
            equal = parsed == actual
        if not equal:
            mismatches.append(f"{name}: documented {text!r} != actual {actual!r}")

    assert not mismatches, "\n".join(mismatches)


def _registered_cli_commands() -> set[str]:
    from quport.cli import app

    names = set()
    for command in app.registered_commands:
        callback = command.callback
        assert callback is not None
        names.add(command.name or callback.__name__.replace("_", "-"))
    return names


@pytest.mark.skipif(
    not _CLI_REFERENCE.is_file(), reason="docs/ is not present in this checkout"
)
def test_every_cli_command_is_documented() -> None:
    """A command nobody documents is a command nobody finds."""
    commands = _registered_cli_commands()
    assert len(commands) >= 5, "expected the CLI to register several commands"

    cli_text = _CLI_REFERENCE.read_text(encoding="utf-8")
    undocumented = sorted(c for c in commands if f"quport {c}" not in cli_text)
    assert not undocumented, f"docs/cli.md omits: {undocumented}"

    if _README.is_file():
        readme = _README.read_text(encoding="utf-8")
        missing = sorted(c for c in commands if f"quport {c}" not in readme)
        assert not missing, f"README.md omits: {missing}"
