import torch
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titans.utils.memory import NeuralMemoryModule


def test_neural_memory_module_init():
    """Test initialization of the Neural Memory Module."""
    d_model = 64
    memory_depth = 2
    
    memory = NeuralMemoryModule(
        d_model=d_model,
        memory_depth=memory_depth,
    )
    
    # Check that the memory has the correct number of layers
    assert len(memory.memory_layers) == memory_depth
    
    # Check that the projection matrices have the correct shapes
    assert memory.W_K.weight.shape == (d_model, d_model)
    assert memory.W_V.weight.shape == (d_model, d_model)
    assert memory.W_Q.weight.shape == (d_model, d_model)


def test_neural_memory_module_forward():
    """Test forward pass of the Neural Memory Module."""
    d_model = 64
    memory_depth = 2
    batch_size = 2
    seq_len = 10
    
    memory = NeuralMemoryModule(
        d_model=d_model,
        memory_depth=memory_depth,
    )
    
    # Generate random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, memory_output = memory(x, is_training=True)
    
    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert memory_output.shape == (batch_size, seq_len, d_model)


def test_neural_memory_module_reset():
    """Test resetting the Neural Memory Module."""
    d_model = 64
    memory_depth = 2
    
    memory = NeuralMemoryModule(
        d_model=d_model,
        memory_depth=memory_depth,
    )
    
    # Store initial weights
    initial_weights = []
    for layer in memory.memory_layers:
        initial_weights.append(layer.weight.data.clone())
    
    # Generate random input and perform forward pass to update weights
    x = torch.randn(2, 10, d_model)
    memory(x, is_training=True)
    
    # Reset memory
    memory.reset_memory()
    
    # Check that weights have been reset
    for i, layer in enumerate(memory.memory_layers):
        # Weights should be different after reset
        assert not torch.allclose(layer.weight.data, initial_weights[i])


if __name__ == "__main__":
    # Run tests
    test_neural_memory_module_init()
    test_neural_memory_module_forward()
    test_neural_memory_module_reset()
    print("All tests passed!") 