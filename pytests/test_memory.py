import pytest
import torch

from titans.utils.memory import FORGET_GATE_SHIFT, NeuralMemoryModule

D_MODEL = 16


def test_neural_memory_module_init():
    """Test initialization of the Neural Memory Module."""
    memory = NeuralMemoryModule(d_model=D_MODEL, memory_depth=2)

    assert len(memory.memory_layers) == 2
    assert memory.W_K.weight.shape == (D_MODEL, D_MODEL)
    assert memory.W_V.weight.shape == (D_MODEL, D_MODEL)
    assert memory.W_Q.weight.shape == (D_MODEL, D_MODEL)


def test_forward_shapes_and_state():
    """forward returns one output per token and a state with one memory per sequence."""
    memory = NeuralMemoryModule(d_model=D_MODEL, memory_depth=2)
    x = torch.randn(3, 10, D_MODEL)

    output, (params, momentum) = memory(x)

    assert output.shape == (3, 10, D_MODEL)
    assert [p.shape for p in params] == [(3, D_MODEL, D_MODEL), (3, D_MODEL)] * 2
    assert [m.shape for m in momentum] == [p.shape for p in params]


def test_forward_does_not_change_the_module():
    """The learned weights are a starting point; reading and writing leaves them alone."""
    memory = NeuralMemoryModule(d_model=D_MODEL)
    before = {name: value.clone() for name, value in memory.state_dict().items()}

    memory(torch.randn(2, 10, D_MODEL))

    for name, value in memory.state_dict().items():
        assert torch.equal(value, before[name])


def test_read_only_keeps_state():
    """With is_training=False the memory is read but not written."""
    memory = NeuralMemoryModule(d_model=D_MODEL)
    state = memory.init_state(batch_size=2)

    _, new_state = memory(torch.randn(2, 5, D_MODEL), state, is_training=False)

    assert all(torch.equal(a, b) for a, b in zip(state[0], new_state[0]))


def test_output_depends_on_the_past_only():
    """Token t is read from the memory as it was before token t was written."""
    memory = NeuralMemoryModule(d_model=D_MODEL)
    x = torch.randn(1, 8, D_MODEL)
    x_changed = x.clone()
    x_changed[:, 5:] = torch.randn(1, 3, D_MODEL)

    output, _ = memory(x)
    output_changed, _ = memory(x_changed)

    assert torch.allclose(output[:, :5], output_changed[:, :5])
    # ...and writing does change what later tokens read
    read_only, _ = memory(x, is_training=False)
    assert not torch.allclose(output[:, 1:], read_only[:, 1:])


def test_sequences_in_a_batch_have_separate_memories():
    """A sequence gives the same output alone and inside a batch."""
    memory = NeuralMemoryModule(d_model=D_MODEL)
    a, b = torch.randn(1, 6, D_MODEL), torch.randn(1, 6, D_MODEL)

    alone, _ = memory(a)
    batched, _ = memory(torch.cat([a, b]))

    assert torch.allclose(alone, batched[:1], atol=1e-6)


@pytest.mark.parametrize("memory_depth", [1, 2, 3])
def test_surprise_matches_autograd(memory_depth):
    """The hand-written gradient of the memory loss equals the one autograd computes."""
    memory = NeuralMemoryModule(d_model=D_MODEL, memory_depth=memory_depth).double()
    params = [p.detach().clone().requires_grad_(True) for p in memory.init_state(3)[0]]
    k = torch.randn(3, D_MODEL, dtype=torch.double)
    v = torch.randn(3, D_MODEL, dtype=torch.double)

    h = k
    for i in range(memory_depth):
        h = torch.einsum("boi,bi->bo", params[2 * i], h) + params[2 * i + 1]
        if i < memory_depth - 1:
            h = torch.relu(h)
    expected = torch.autograd.grad(((h - v) ** 2).sum(), params)

    for grad, expected_grad in zip(memory._surprise(k, v, params), expected):
        assert torch.allclose(grad, expected_grad)


def test_update_follows_the_paper_equations():
    """One write step is S_t = eta*S_{t-1} - theta*grad and M_t = (1 - alpha)*M_{t-1} + S_t."""
    memory = NeuralMemoryModule(d_model=D_MODEL, memory_depth=2, max_grad_norm=None)
    params, _ = memory.init_state(1)
    momentum = [torch.randn_like(p) for p in params]
    x_t = torch.randn(1, D_MODEL)
    theta = torch.sigmoid(memory.theta_proj(x_t))
    eta = torch.sigmoid(memory.eta_proj(x_t))
    alpha = torch.sigmoid(memory.alpha_proj(x_t) - FORGET_GATE_SHIFT)
    key = torch.nn.functional.normalize(memory.W_K(x_t), dim=-1)
    grads = memory._surprise(key, memory.W_V(x_t), params)

    new_params, new_momentum = memory._update_step(x_t, (params, momentum))

    for param, past, grad, new_param, new_past in zip(
        params, momentum, grads, new_params, new_momentum
    ):
        surprise = eta.item() * past - theta.item() * grad
        assert torch.allclose(new_past, surprise, atol=1e-6)
        assert torch.allclose(new_param, (1 - alpha.item()) * param + surprise, atol=1e-6)


def test_all_parameters_are_trainable():
    """A loss on the outputs reaches the projections, the gates and the initial memory."""
    memory = NeuralMemoryModule(d_model=D_MODEL)

    output, _ = memory(torch.randn(2, 6, D_MODEL))
    output.sum().backward()

    for name, param in memory.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_long_sequences_stay_finite():
    """Clipping the surprise keeps the memory stable on large inputs."""
    memory = NeuralMemoryModule(d_model=D_MODEL)

    output, (params, _) = memory(10 * torch.randn(2, 200, D_MODEL))

    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p).all() for p in params)
