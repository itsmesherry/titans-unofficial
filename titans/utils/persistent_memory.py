import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    """
    Persistent Memory module as described in the paper.
    
    This module provides learnable but data-independent parameters that encode
    knowledge about a task.
    """
    
    def __init__(
        self,
        d_model: int,
        num_tokens: int,
    ):
        """
        Initialize the Persistent Memory module.
        
        Args:
            d_model: Dimension of the model
            num_tokens: Number of persistent memory tokens
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_tokens = num_tokens
        
        # Learnable persistent memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(1, num_tokens, d_model))
        
        # Initialize parameters
        nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Forward pass to get persistent memory tokens.
        
        Args:
            batch_size: Batch size to expand the memory tokens
            
        Returns:
            Persistent memory tokens of shape (batch_size, num_tokens, d_model)
        """
        # Expand memory tokens for batch dimension
        return self.memory_tokens.expand(batch_size, -1, -1)
    
    def prepend_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend persistent memory tokens to the input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with prepended persistent memory tokens
        """
        batch_size = x.size(0)

        # Get persistent memory tokens
        memory_tokens = self.forward(batch_size)

        # Concatenate with input sequence
        return torch.cat([memory_tokens, x], dim=1) 