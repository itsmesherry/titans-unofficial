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

| Argument            | Default | Meaning                                                |
| ------------------- | ------- | ------------------------------------------------------ |
| `d_model`           |         | Width of the input and output embeddings               |
| `n_layers`          |         | Number of attention blocks                             |
| `n_heads`           |         | Attention heads; must divide `d_model`                 |
| `memory_depth`      | 2       | Layers in the memory network                           |
| `persistent_tokens` | 16      | Learned, input-independent tokens prepended to input   |
| `window_size`       | 128     | Attention window (MAG, MAL); off by default in MAC     |
| `chunk_size`        | 128     | Tokens per chunk (MAC only)                            |
| `dropout`           | 0.1     | Dropout rate                                           |
| `persist_memory`    | False   | Carry the memory over from one `forward` call to the next |

All three variants are causal: the output at a position depends only on that
position and the ones before it, so they can be trained as language models.

## Working with the memory

Every sequence in a batch gets its own memory. It starts from weights the model
has learned, is updated token by token as the sequence is read, and is thrown
away when `forward` returns. Calls are therefore independent: the same input
always gives the same output, and the model's parameters never change at
inference time.

```python
y = model(x)                     # each sequence reads from and writes to its own memory
y = model(x, is_training=False)  # reads only: the memory stays at its learned starting point
```

`is_training` controls the memory only and is independent of `model.train()` and
`model.eval()`, which still switch dropout as usual. Leave it on at inference
time if you want the test-time learning the paper is about.

To feed one long input in several calls, keep the memory between them:

```python
model = TitansMAC(d_model=128, n_layers=2, n_heads=4, persist_memory=True)

for piece in pieces_of_a_long_document:  # each of shape (batch, tokens, d_model)
    y = model(piece)                     # remembers the earlier pieces

model.reset_memory()                     # before the next document or a new batch size
```

The memory updates are differentiable, so training with an ordinary optimizer
also trains the memory's starting weights, its key, value and query projections,
and the gates that set its learning rate, momentum and forgetting.

## Differences from the paper

This is an alpha-stage reimplementation. It follows the paper's equations for
the memory (Section 3.1) and its three architectures (Section 4), with these
departures:

- The gradient the memory learns from is clipped to a norm of 1 per sequence.
  Without it the updates diverge on inputs of ordinary scale. Pass
  `max_grad_norm=None` to `NeuralMemoryModule` to turn it off.
- The forgetting gate is shifted so that an untrained memory keeps about 98% of
  its weights per token.
- The memory is written one token at a time. The chunked, parallel training of
  Section 3.2 is not implemented, so long sequences are slow.
- MAC keeps the model causal by letting a token see what the memory recalled
  for itself and for earlier tokens only.
- MAL shares one set of memory parameters between its layers, each with its own
  memory state, and places the memory after attention in each block.
- The depthwise-separable convolutions and SiLU activations of Section 4.4 are
  not used.

No results from the paper have been reproduced with this code.

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
