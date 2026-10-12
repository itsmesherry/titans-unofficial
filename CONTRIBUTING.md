# Contributing

Thanks for helping improve `titans-unofficial`. Bug reports, fixes and closer
matches to the paper are all welcome.

## Setup

```bash
git clone https://github.com/itsmesherry/titans-unofficial.git
cd titans-unofficial
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
pytest
```

CI runs the same checks on Python 3.10 to 3.13 and builds the package.

A pull request should:

- include a test for any behaviour it adds or fixes
- add a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md)
- say which section or equation of the [paper](https://arxiv.org/abs/2501.00663)
  it follows when it changes how a model computes its output

## Compatibility

Within a major version:

- do not remove or rename public classes, methods or constructor arguments
- a change that alters the numbers a model produces for the same weights and
  input is a behaviour change: call it out in the changelog, and save changes
  that invalidate trained checkpoints for a major version

## Releasing (maintainers)

1. Open a release pull request that bumps the version in `pyproject.toml`
   (the only place it is defined) and renames `Unreleased` in `CHANGELOG.md` to
   the new version and date (leave a fresh, empty `Unreleased` above it).
2. Merge it once CI is green.
3. Create a GitHub Release on that commit with the tag `v<version>`, for example
   `v1.2.0`, and paste the changelog section as the notes.

Publishing the release runs `.github/workflows/publish.yml`, which builds the
package from the tag and uploads it to PyPI through Trusted Publishing. No API
token is involved, and the workflow refuses to publish if the tag does not match
the version in `pyproject.toml`.
