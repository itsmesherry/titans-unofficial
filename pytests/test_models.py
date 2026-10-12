import pytest
import torch

from titans import TitansMAC, TitansMAG, TitansMAL
from titans.utils.memory import NeuralMemoryModule

D_MODEL = 32


def make_model(model_class, **kwargs):
    """Small model; MAC chunks and MAG/MAL windows are 8 tokens wide."""
    size = {"chunk_size": 8} if model_class is TitansMAC else {"window_size": 8}
    return model_class(
        d_model=D_MODEL, n_layers=2, n_heads=4, persistent_tokens=4, **size, **kwargs
    )


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
@pytest.mark.parametrize("seq_len", [5, 8, 16, 20])
@pytest.mark.parametrize("is_training", [True, False])
def test_forward_shape(model_class, seq_len, is_training):
    """Output keeps the input shape for sequences shorter and longer than a chunk/window."""
    model = make_model(model_class)
    x = torch.randn(2, seq_len, D_MODEL)

    output = model(x, is_training=is_training)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_mac_default_chunk_size_long_sequence():
    """MAC accepts sequences longer than its default chunk size (regression test)."""
    model = TitansMAC(d_model=D_MODEL, n_layers=1, n_heads=4)
    x = torch.randn(1, 300, D_MODEL)

    assert model(x).shape == x.shape


def test_mac_later_chunks_read_earlier_chunks():
    """Changing the first chunk changes the output for the second chunk, through memory."""
    model = make_model(TitansMAC).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(1, 16, D_MODEL)
    x_changed = x.clone()
    x_changed[:, :8] = torch.randn(1, 8, D_MODEL)

    with torch.no_grad():
        second_chunk = model(x)[:, 8:]
        model.load_state_dict(state)
        model.neural_memory.surprise_momentum = None
        second_chunk_changed = model(x_changed)[:, 8:]

    assert not torch.allclose(second_chunk, second_chunk_changed)


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
def test_backward(model_class):
    """A loss on the output reaches the attention and persistent-memory parameters."""
    model = make_model(model_class)

    model(torch.randn(2, 12, D_MODEL)).sum().backward()

    assert model.attention_layers[0].attention.W_Q.weight.grad.abs().sum() > 0
    assert model.persistent_memory.memory_tokens.grad.abs().sum() > 0


def test_first_memory_update_goes_to_the_right_layers():
    """Each memory layer is updated with its own gradient, including on the first step."""
    memory = NeuralMemoryModule(D_MODEL, memory_depth=2)
    before = [layer.weight.data.clone() for layer in memory.memory_layers]
    x = torch.randn(1, D_MODEL)
    gradients = memory.compute_gradient(memory.W_K(x), memory.W_V(x))
    theta, eta, alpha = torch.tensor([[0.5]]), torch.tensor([[0.0]]), torch.tensor([[0.0]])

    memory.update_memory(x, theta, eta, alpha)

    for layer, weight, (weight_grad, _) in zip(memory.memory_layers, before, gradients):
        assert torch.allclose(layer.weight.data, weight - 0.5 * weight_grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
def test_forward_on_cuda(model_class):
    """Models run on a GPU after to_device()."""
    model = make_model(model_class).to_device("cuda")
    x = torch.randn(2, 16, D_MODEL, device="cuda")

    assert model(x).device.type == "cuda"
