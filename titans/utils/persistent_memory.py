import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    """
    Persistent Memory module as described in Section 2.3 of the paper.
    
    Provides learnable but data-independent parameters to store task knowledge.
    """
    
    def __init__(
        self,
        d_model: int,
        num_tokens: int,
    ):
        """Initialize the Persistent Memory module."""
        super().__init__()
        
        self.d_model = d_model
        self.num_tokens = num_tokens
        
        # Learnable persistent memory tokens (P matrix in the paper)
        self.memory_tokens = nn.Parameter(torch.randn(1, num_tokens, d_model))
        
        # Initialize with small values for stability
        nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """Return persistent memory tokens expanded to batch size."""
        # Get the device of the memory tokens
        device = self.memory_tokens.device
        
        # Expand for batch dimension
        return self.memory_tokens.expand(batch_size, -1, -1)
    
    def prepend_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend persistent memory tokens to the input sequence."""
        batch_size = x.size(0)
        device = x.device
        
        # Ensure memory tokens are on the same device as x
        memory_tokens = self.forward(batch_size).to(device)
        
        # Concatenate with consistent device handling
        return torch.cat([memory_tokens, x], dim=1) 