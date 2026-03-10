# CI and local verification notes

This repository currently verifies cleanly with the following local sequence:

```bash
.venv/bin/ruff check .
.venv/bin/pyright
.venv/bin/pytest -q
```

## Current workflow review

- `.github/workflows/test.yml`
  - runs on pushes to `main` and pull requests targeting `main` only when code, tests, dependency manifests, or workflow files change
  - cancels superseded in-progress runs for the same branch or PR
  - sets up Python 3.12, installs the project with `uv`, builds the Rust extension with `maturin develop --release`, then runs Ruff, Pyright, and pytest
- `.github/workflows/CI.yml`
  - runs on tags
  - repeats the test job before building wheels/sdist and uploading to PyPI
  - builds wheel artifacts explicitly for Python 3.12, 3.13, and 3.14 across the supported OS/arch matrix
- `.github/workflows/tag.yml`
  - runs on pushes to `main`
  - uses `ubuntu-slim` because it only needs lightweight git/Python release automation
  - serializes release runs with a `concurrency` group so overlapping pushes do not compute the same next tag
  - checks out the latest `main` tip and refreshes tags before calculating a release
  - repairs `main` if an earlier release tag exists without the corresponding version/changelog commit on the branch
  - uses Commitizen directly in shell steps so duplicate-tag/no-op cases are handled explicitly
  - pushes the bump commit and tag together with `git push --atomic`

## Observations

- The checked-in workflows are consistent with the current project layout and passed local equivalents during this session.
- `Cargo.toml` and `pyproject.toml` should stay version-synced; this was corrected in this branch.
- Commitizen is configured to bump both `pyproject.toml` and `Cargo.toml` so Python and Rust releases stay aligned.
- If a tag already exists but `main` is still on an older version, the workflow now syncs `CHANGELOG.md`, `pyproject.toml`, and `Cargo.toml` to that tagged release before attempting any future bump.
- When building locally outside the workflow, PyO3 can pick up a newer system Python if the environment is not pinned. Using the project `.venv` Python avoids that issue.

## Recommended local build commands

For local Rust extension rebuilds, prefer the project virtual environment:

```bash
PYO3_PYTHON=$PWD/.venv/bin/python .venv/bin/maturin develop --release
```

If PyO3 still resolves the wrong interpreter in a custom shell environment, explicitly set:

```bash
env -u CARGO \
  PYO3_ENVIRONMENT_SIGNATURE='cpython-3.12-64bit' \
  PYO3_PYTHON="$PWD/.venv/bin/python" \
  PYTHON_SYS_EXECUTABLE="$PWD/.venv/bin/python" \
  cargo rustc --features pyo3/extension-module --manifest-path Cargo.toml --lib
```

This manual `cargo rustc` form was useful during development for forcing the intended interpreter.