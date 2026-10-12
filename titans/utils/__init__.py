"""
Utility modules for the Titans architecture.

This package contains the core components needed to implement the Titans architecture:

- NeuralMemoryModule: Implements the long-term memory described in Section 3.1,
  which learns to memorize at test time using gradient descent with momentum and
  weight decay.

- MultiHeadAttention: Core attention mechanism used in all Titans variants.

- SlidingWindowAttention: Attention with a sliding window constraint, used 
  primarily in the MAG (Memory as Gating) variant as the short-term memory component.

- CausalSelfAttention: Causal attention used primarily in the MAC (Memory as Context)
  variant, where the current token can only attend to itself and previous tokens.

- PersistentMemory: Implements the persistent memory described in Section 3.3,
  which provides learnable but data-independent parameters to store task knowledge.
"""

from .memory import NeuralMemoryModule
from .attention import MultiHeadAttention, SlidingWindowAttention, CausalSelfAttention
from .persistent_memory import PersistentMemory

__all__ = [
    'NeuralMemoryModule',
    'MultiHeadAttention',
    'SlidingWindowAttention',
    'CausalSelfAttention',
    'PersistentMemory',
] 