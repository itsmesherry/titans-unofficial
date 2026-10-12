# TODO

Planned work, roughly in priority order. Open an issue or pull request if you
want to pick one up.

## Memory module (planned for 2.0)

These change model outputs and how checkpoints behave, so they are grouped into
one major release.

- [ ] `reset_memory()` should restore learned initial memory weights; today it
      draws new random weights, so results differ after every reset
- [ ] Give each sequence in a batch its own memory state; today the batch shares
      one, so a sample's output depends on its batch-mates (MAG, MAL) and on
      earlier forward calls
- [ ] Make the key/value projections and the learning-rate, momentum and
      forgetting gates trainable; they receive no gradient today
- [ ] Keep persistent tokens visible from every position in MAG and MAL; today
      they fall out of the sliding window

## Library

- [ ] Query memory with the whole chunk in MAC instead of its average
- [ ] Ship a `py.typed` marker so users' type checkers see the type hints
