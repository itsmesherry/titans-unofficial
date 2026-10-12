# TODO

Planned work, roughly in priority order. Open an issue or pull request if you
want to pick one up.

- [ ] Reproduce a result from the paper, however small, and publish the script
- [ ] Parallelize the memory updates over chunks (Section 3.2); writing one
      token at a time makes long sequences slow
- [ ] Add the depthwise-separable convolutions and SiLU activations of
      Section 4.4
- [ ] Ship a `py.typed` marker so users' type checkers see the type hints
