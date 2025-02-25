import torch.nn as nn

from typing import Optional

from titans.utils.memory import NeuralMemoryModule
from titans.utils.persistent_memory import PersistentMemory


class TitansBase(nn.Module):
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
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.memory_depth = memory_depth
        self.persistent_tokens = persistent_tokens
        self.window_size = window_size
        
        self.persistent_memory = PersistentMemory(d_model, persistent_tokens)
        
        self.neural_memory = NeuralMemoryModule(
            d_model=d_model,
            memory_depth=memory_depth,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def reset_memory(self):
        self.neural_memory.reset_memory()
