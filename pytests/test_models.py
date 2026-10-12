import pytest
import torch

from titans import TitansMAC, TitansMAG, TitansMAL

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
    x = torch.randn(1, 16, D_MODEL)
    x_changed = x.clone()
    x_changed[:, :8] = torch.randn(1, 8, D_MODEL)

    with torch.no_grad():
        assert not torch.allclose(model(x)[:, 8:], model(x_changed)[:, 8:])


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
def test_backward(model_class):
    """A loss on the output reaches the attention and persistent-memory parameters."""
    model = make_model(model_class)

    model(torch.randn(2, 12, D_MODEL)).sum().backward()

    assert model.attention_layers[0].attention.W_Q.weight.grad.abs().sum() > 0
    assert model.persistent_memory.memory_tokens.grad.abs().sum() > 0


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
@pytest.mark.parametrize("is_training", [True, False])
def test_causal(model_class, is_training):
    """Changing later tokens never changes the output at earlier positions."""
    model = make_model(model_class).eval()
    x = torch.randn(2, 20, D_MODEL)
    x_changed = x.clone()
    x_changed[:, 11:] = torch.randn(2, 9, D_MODEL)

    with torch.no_grad():
        output = model(x, is_training=is_training)
        output_changed = model(x_changed, is_training=is_training)

    assert torch.allclose(output[:, :11], output_changed[:, :11], atol=1e-6)
    assert not torch.allclose(output[:, 11:], output_changed[:, 11:])


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
def test_calls_and_batch_mates_do_not_affect_each_other(model_class):
    """By default every call starts from the learned memory, one memory per sequence."""
    model = make_model(model_class).eval()
    a, b = torch.randn(1, 20, D_MODEL), torch.randn(1, 20, D_MODEL)

    with torch.no_grad():
        alone = model(a)
        batched = model(torch.cat([a, b]))[:1]
        again = model(a)

    assert torch.equal(alone, again)
    assert torch.allclose(alone, batched, atol=1e-5)


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
def test_persist_memory(model_class):
    """With persist_memory the memory carries over between calls until it is reset."""
    model = make_model(model_class, persist_memory=True).eval()
    x = torch.randn(2, 20, D_MODEL)

    with torch.no_grad():
        first = model(x)
        second = model(x)
        model.reset_memory()
        after_reset = model(x)

    assert not torch.allclose(first, second, atol=1e-4)
    assert torch.equal(first, after_reset)


def test_mac_pieces_with_persist_memory_equal_one_call():
    """Feeding a long input chunk by chunk gives the same result as feeding it at once."""
    model = make_model(TitansMAC, persist_memory=True).eval()
    x = torch.randn(2, 24, D_MODEL)

    with torch.no_grad():
        whole = model(x)
        model.reset_memory()
        pieces = torch.cat([model(x[:, :8]), model(x[:, 8:16]), model(x[:, 16:])], dim=1)

    assert torch.allclose(whole, pieces, atol=1e-6)


def test_persist_memory_rejects_a_new_batch_size():
    model = make_model(TitansMAG, persist_memory=True)
    model(torch.randn(2, 8, D_MODEL))

    with pytest.raises(ValueError, match="reset_memory"):
        model(torch.randn(3, 8, D_MODEL))


@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
@pytest.mark.parametrize("persist_memory", [False, True])
def test_training_steps(model_class, persist_memory):
    """Several optimizer steps run, and every parameter receives a gradient."""
    model = make_model(model_class, persist_memory=persist_memory)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(2):
        optimizer.zero_grad()
        model(torch.randn(2, 20, D_MODEL)).pow(2).mean().backward()
        optimizer.step()

    for name, param in model.named_parameters():
        if persist_memory and "memory_layers" in name:
            continue  # the learned starting point is only used right after a reset
        assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_forward_does_not_change_the_weights():
    """Test-time learning happens in the memory state, not in the model's parameters."""
    model = make_model(TitansMAL, persist_memory=True).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        model(torch.randn(2, 20, D_MODEL))

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("model_class", [TitansMAG, TitansMAL])
def test_persistent_tokens_visible_beyond_the_window(model_class):
    """Positions past the attention window still depend on the persistent tokens."""
    model = model_class(
        d_model=D_MODEL, n_layers=1, n_heads=4, persistent_tokens=4, window_size=8
    ).eval()

    output = model(torch.randn(1, 30, D_MODEL), is_training=False)
    (grad,) = torch.autograd.grad(output[0, 25].sum(), model.persistent_memory.memory_tokens)

    assert grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_class", [TitansMAC, TitansMAG, TitansMAL])
def test_forward_on_cuda(model_class):
    """Models run on a GPU after to_device()."""
    model = make_model(model_class).to_device("cuda")
    x = torch.randn(2, 16, D_MODEL, device="cuda")

    assert model(x).device.type == "cuda"
