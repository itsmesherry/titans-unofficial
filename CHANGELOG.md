# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.2] - 2026-10-12

### Added

- GPU support: models and their memory state stay on one device, and
  `to_device()` moves a model together with its memory. This was tagged as
  1.1.1 but never published to PyPI.
- `titans.__version__`.

### Fixed

- `TitansMAC` raised a size-mismatch error for any input longer than
  `chunk_size` (128 tokens by default). Chunks are now processed in order, so
  each chunk reads what earlier chunks stored in memory.
- The first memory update after a reset applied the first memory layer's
  gradient to the last layer when `memory_depth` was 2 or more.

### Changed

- `TitansMAC(max_parallel_chunks=...)` is accepted but ignored.
- The `dev` extra installs `pytest`, `ruff`, `build` and `twine`; the
  `examples` extra no longer installs `einops`, which no example used.

## [1.1.0] - 2025-02-27

Last version published before this changelog was started. See the
[GitHub releases](https://github.com/itsmesherry/titans-unofficial/releases)
for earlier history.

[Unreleased]: https://github.com/itsmesherry/titans-unofficial/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/itsmesherry/titans-unofficial/compare/v1.1.0...v1.1.2
[1.1.0]: https://github.com/itsmesherry/titans-unofficial/releases/tag/v1.1.0
