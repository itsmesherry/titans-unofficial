import torch.nn as nn

from typing import List, Optional

from titans.utils.memory import MemoryState, NeuralMemoryModule, detach_state
from titans.utils.persistent_memory import PersistentMemory


class TitansBase(nn.Module):
    """
    Base class for Titans models with shared components.
    
    Provides the neural memory and persistent memory components used by all variants.
    
    Every sequence in a batch has its own memory. By default each forward call
    starts from the learned memory, so calls are independent of each other. With
    persist_memory=True the memory carries over to the next call until
    reset_memory() is called, which lets a long input be fed in pieces.
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
        persist_memory: bool = False,
    ):
        """Initialize the base Titans model."""
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.memory_depth = memory_depth
        self.persistent_tokens = persistent_tokens
        self.window_size = window_size
        self.persist_memory = persist_memory
        
        # Persistent memory from Section 3.3
        self.persistent_memory = PersistentMemory(d_model, persistent_tokens)
        
        # Neural memory module from Section 3.1
        self.neural_memory = NeuralMemoryModule(
            d_model=d_model,
            memory_depth=memory_depth,
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Memory carried over from the previous forward call (persist_memory only)
        self._memory_states: Optional[List[MemoryState]] = None
        
    def reset_memory(self):
        """Forget what the memory has stored; the next call starts from the learned memory."""
        self._memory_states = None
        
    def _begin_memory(self, batch_size: int, n_states: int = 1) -> List[MemoryState]:
        """Return the memory states to start a forward call with."""
        if self.persist_memory and self._memory_states is not None:
            carried_batch_size = self._memory_states[0][0][0].size(0)
            if carried_batch_size != batch_size:
                raise ValueError(
                    f"The memory holds {carried_batch_size} sequences but the input "
                    f"has {batch_size}. Call reset_memory() before changing batch size."
                )
            return self._memory_states
        return [self.neural_memory.init_state(batch_size) for _ in range(n_states)]
        
    def _end_memory(self, states: List[MemoryState]) -> None:
        """Keep the memory for the next forward call if persist_memory is on."""
        if self.persist_memory:
            # Detached, so the next call does not backpropagate into this one
            self._memory_states = [detach_state(state) for state in states]
        
    def to_device(self, device):
        """
        Move the model to the specified device and reset its memory.
        """
        self.to(device)
        self.reset_memory()
        return self
