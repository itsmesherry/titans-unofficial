<h1 align="center">titans-unofficial</h1>

<p align="center">
  An unofficial PyTorch implementation of<br>
  <a href="https://arxiv.org/abs/2501.00663">Titans: Learning to Memorize at Test Time</a>
  (Behrouz, Zhong and Mirrokni).
</p>

<p align="center">
  <a href="https://pypi.org/project/titans-unofficial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/titans-unofficial"></a>
  <a href="https://github.com/itsmesherry/titans-unofficial/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/itsmesherry/titans-unofficial/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/titans-unofficial/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"></a>
  <a href="https://pypi.org/project/titans-unofficial/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/titans-unofficial"></a>
  <a href="https://github.com/itsmesherry/titans-unofficial/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

Titans pairs attention, which acts as short-term memory, with a small neural
network that serves as long-term memory. That network keeps learning while the
model reads: it is updated by gradient descent on each token, with momentum and
a forgetting term, at inference time as well as during training.

```bash
pip install titans-unofficial
```

```python
import torch
from titans import TitansMAC

model = TitansMAC(d_model=128, n_layers=2, n_heads=4)

x = torch.randn(2, 300, 128)  # (batch, sequence, d_model)
y = model(x)                  # same shape; the memory learns as it reads
```

The models work on embeddings, not token IDs, so they slot in where a
transformer encoder stack would go. Add your own embedding layer and output
head; the [examples](#examples) show how.

## Variants

The paper describes three ways to connect the memory to attention. All three
share the same constructor arguments and the same `forward(x, is_training=True)`.

| Class       | Paper | How the memory is used                                                     |
| ----------- | ----- | -------------------------------------------------------------------------- |
| `TitansMAC` | §4.1  | Memory as context: what the memory recalls is prepended to each chunk      |
| `TitansMAG` | §4.2  | Memory as gate: a learned gate mixes sliding-window attention with memory  |
| `TitansMAL` | §4.3  | Memory as layer: every block runs attention, then memory, then the MLP     |

```python
from titans import TitansMAC, TitansMAG, TitansMAL

mac = TitansMAC(d_model=128, n_layers=2, n_heads=4, chunk_size=128)
mag = TitansMAG(d_model=128, n_layers=2, n_heads=4, window_size=128)
mal = TitansMAL(d_model=128, n_layers=2, n_heads=4, window_size=128)
```

| Argument            | Default | Meaning                                               |
| ------------------- | ------- | ----------------------------------------------------- |
| `d_model`           |         | Width of the input and output embeddings              |
| `n_layers`          |         | Number of attention blocks                            |
| `n_heads`           |         | Attention heads; must divide `d_model`                |
| `memory_depth`      | 2       | Layers in the memory network                          |
| `persistent_tokens` | 16      | Learned, input-independent tokens prepended to input  |
| `window_size`       | 128     | Attention window (MAG, MAL); off by default in MAC    |
| `chunk_size`        | 128     | Tokens per chunk (MAC only)                           |
| `dropout`           | 0.1     | Dropout rate                                          |

## Working with the memory

The memory is state that lives on the model and changes during `forward`.

```python
y = model(x)                     # reads from memory and writes to it
y = model(x, is_training=False)  # reads only; the memory stays as it is

model.reset_memory()             # forget everything, e.g. between documents
model.to_device("cuda")          # move the model together with its memory
```

`is_training` controls the memory only and is independent of `model.train()` and
`model.eval()`, which still switch dropout as usual. Leave it on at inference
time if you want the test-time learning the paper is about.

## Differences from the paper

This is an alpha-stage reimplementation, and it departs from the paper in ways
that matter if you plan to reproduce its results:

- One memory is shared by the whole batch and persists across `forward` calls,
  so outputs depend on batch composition and call order. Use a batch size of 1
  and call `reset_memory()` between sequences for clean behaviour.
- `reset_memory()` draws new random memory weights; it does not return to a
  learned starting point, so results after a reset are not reproducible without
  fixing the random seed.
- The memory's key and value projections and its learning-rate, momentum and
  forgetting gates are not trained; they keep their initial values.
- In MAG and MAL, persistent tokens drop out of the attention window for
  positions beyond `window_size`.
- MAC queries the memory with the average of a chunk, giving one recalled token
  per chunk.

These are tracked in the [roadmap](https://github.com/itsmesherry/titans-unofficial/blob/main/TODO.md).

## Examples

The example scripts are not part of the installed package. Run them from a clone:

```bash
git clone https://github.com/itsmesherry/titans-unofficial.git
cd titans-unofficial
pip install -e ".[examples]"
cd examples
python simple_example.py
```

| Script                   | What it shows                                                  |
| ------------------------ | -------------------------------------------------------------- |
| `simple_example.py`      | A forward pass through each variant                            |
| `language_modeling.py`   | A language model with token embeddings and text generation     |
| `text_classification.py` | A classifier head, with a training and evaluation loop         |
| `fine_tuning.py`         | A longer training loop with evaluation and sample generation   |

## Citation

This repository is not affiliated with the authors. If you use it, cite the
paper:

```bibtex
@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}
```

## Project

[Changelog](https://github.com/itsmesherry/titans-unofficial/blob/main/CHANGELOG.md) · [Contributing](https://github.com/itsmesherry/titans-unofficial/blob/main/CONTRIBUTING.md) · [Roadmap](https://github.com/itsmesherry/titans-unofficial/blob/main/TODO.md) · [MIT License](https://github.com/itsmesherry/titans-unofficial/blob/main/LICENSE)
