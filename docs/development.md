# Development

This page documents the expected contributor workflow. The most important rule is
that code, tests, and docs should move together: public behavior changes should not
land without tests and documentation.

## Environment

```bash
python -m pip install -e '.[yaml,viz,graph,docs]'
python -m pip install pre-commit
pre-commit install
```

If the `pre-commit` console script is not on `PATH`, use `python -m pre_commit ...`.

## Required checks

Run these before committing:

```bash
pytest -q
mypy src tests
python -m compileall -q src tests
pre-commit run --all-files
sphinx-build -W --keep-going -b dirhtml docs docs/_build/dirhtml
```

The pre-commit configuration currently runs merge-conflict checks, YAML checks,
end-of-file fixes, trailing whitespace trimming, isort, mypy on `src`, and black.

## Documentation checklist

When changing public behavior, update docs in the same change:

- Public Python API change: update [API reference](api-references.md).
- New or changed config field: update [Configuration](configuration.md).
- New or changed CLI command/option/artifact: update [CLI reference](cli.md).
- New workflow or recommended usage: update [Getting started](getting-started.md) or [Examples](examples.md).
- Changed conceptual model: update [Concepts](concepts.md).

Use [Index](index.md) as the navigation source of truth. If you add a new docs page,
link it from the index and from the README documentation section.

## Type-checking policy

QuPort ships `py.typed`; public annotations are part of the developer-facing API.
Runtime validation should still guard user-provided values because config files,
notebooks, and CLI workflows can pass malformed data even when type checking is not used.

For tests that intentionally pass invalid types to exercise runtime validation,
prefer `typing.cast(Any, value)` at the call site rather than weakening production
function signatures.

## Runtime validation policy

- Reject booleans for integer/numeric fields unless a field is explicitly boolean.
- Validate public API inputs before passing them to Qiskit where possible so users
  receive QuPort-specific error messages.
- Keep validation errors deterministic and specific enough for tests.
- Validate serialized artifacts before writing files when invalid data could leave
  partial or misleading outputs.

## Testing guidance

Add tests at the level where behavior is guaranteed:

- unit tests for pure helpers and validation;
- smoke tests for Qiskit-dependent mapping/compilation paths;
- regression tests for previously observed edge cases;
- CSV/artifact tests for file-writing behavior.

Prefer small deterministic circuits in tests. Random-circuit tests should always use
a fixed seed.

## Docs site workflow

The documentation is a Sphinx site built from Markdown with MyST. Source files live
in `docs/`, `docs/conf.py` owns the production build configuration, and GitHub
Actions publishes `docs/_build/dirhtml` to GitHub Pages from the `main` branch.

Useful local commands:

```bash
make -C docs html      # standard local HTML build
make -C docs strict    # production-equivalent build with warnings as errors
make -C docs linkcheck # external and internal link validation
```

The repository does not require a dedicated Markdown linter, but contributors
should at least check local links when editing docs. A simple check is:

```bash
python - <<'PY'
from pathlib import Path
import re, sys
errors = []
for p in [Path('README.md'), *Path('docs').glob('*.md')]:
    for target in re.findall(r'\[[^\]]+\]\(([^)#][^)]*)\)', p.read_text()):
        if '://' in target or target.startswith('mailto:'):
            continue
        local = target.split('#', 1)[0]
        if local and not (p.parent / local).exists():
            errors.append(f'{p}: missing {target}')
print('\n'.join(errors) or 'all local markdown links exist')
if errors:
    sys.exit(1)
PY
```

## Pull request expectations

A strong QuPort PR should include:

1. a concise motivation;
2. implementation details;
3. tests/checks run;
4. docs updates for user-visible behavior;
5. notes about limitations or modeling assumptions when applicable.
