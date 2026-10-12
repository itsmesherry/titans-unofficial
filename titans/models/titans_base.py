import torch.nn as nn

from typing import Optional

from titans.utils.memory import NeuralMemoryModule
from titans.utils.persistent_memory import PersistentMemory


class TitansBase(nn.Module):
    """
    Base class for Titans models with shared components.
    
    Provides the neural memory and persistent memory components used by all variants.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize the base Titans model."""
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.memory_depth = memory_depth
        self.persistent_tokens = persistent_tokens
        self.window_size = window_size
        
        # Persistent memory from Section 3.3
        self.persistent_memory = PersistentMemory(d_model, persistent_tokens)
        
        # Neural memory module from Section 3.1
        self.neural_memory = NeuralMemoryModule(
            d_model=d_model,
            memory_depth=memory_depth,
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def reset_memory(self):
        """Reset the neural memory module."""
        self.neural_memory.reset_memory()
        
    def to_device(self, device):
        """
        Move the model to the specified device.
        This ensures all components are on the same device.
        """
        self.to(device)
        # Explicitly move the neural memory module
        for layer in self.neural_memory.memory_layers:
            layer.to(device)
        
        # Reset momentum to ensure it will be created on the right device
        self.neural_memory.surprise_momentum = None
        
        return self
