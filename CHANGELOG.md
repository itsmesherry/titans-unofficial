# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-10-12

A rework of the neural memory. Model outputs change, so models trained with 1.x
need retraining; their checkpoints still load.

### Fixed

- `TitansMAC` and multi-layer `TitansMAL` let later tokens influence the output
  at earlier positions, which made them unusable as language models. All three
  variants are now causal.
- Every sequence in a batch has its own memory. Before, the batch shared one, so
  an output depended on the other sequences in the batch.
- The memory's key and value projections and its learning-rate, momentum and
  forgetting gates now receive gradients and can be trained.
- Persistent tokens stay visible beyond the attention window in `TitansMAG` and
  `TitansMAL`.

### Changed

- Each `forward` call starts from the learned memory and leaves the model's
  parameters untouched, so the same input always gives the same output. To carry
  the memory across calls as before, construct the model with
  `persist_memory=True`.
- `reset_memory()` clears carried-over memory instead of drawing new random
  weights.
- `TitansMAC` queries the memory with every token of a chunk, not the chunk's
  average, and reading the memory no longer writes to it.
- Keys and queries are L2-normalized and the memory loss is the squared L2 norm,
  as in the paper. The memory's gradient is clipped to norm 1 per sequence
  (`max_grad_norm`).
- An untrained forgetting gate now keeps about 98% of the memory per token; it
  used to keep about half.
- `NeuralMemoryModule` no longer holds the memory. `forward(x, state=None,
  is_training=True)` returns `(outputs, state)`, and `init_state`, `retrieve` and
  `update` are the public operations. `reset_memory`, `memory_forward`,
  `compute_loss`, `compute_gradient` and `update_memory` are removed.

### Added

- `persist_memory` argument on all three models.
- `global_prefix` argument on `MultiHeadAttention` and `SlidingWindowAttention`.

## [1.1.4] - 2026-10-12

Versions 1.1.1 to 1.1.3 exist only as git tags from 2025 and were never
published to PyPI.

### Added

- GPU support: models and their memory state stay on one device, and
  `to_device()` moves a model together with its memory.
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

[Unreleased]: https://github.com/itsmesherry/titans-unofficial/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/itsmesherry/titans-unofficial/compare/v1.1.4...v2.0.0
[1.1.4]: https://github.com/itsmesherry/titans-unofficial/compare/v1.1.0...v1.1.4
[1.1.0]: https://github.com/itsmesherry/titans-unofficial/releases/tag/v1.1.0
