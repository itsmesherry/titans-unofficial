import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional sliding window support.
    Core attention mechanism used in all Titans variants.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
        attention_scale: float = 1.0,
    ):
        """Initialize the multi-head attention."""
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.attention_scale = attention_scale
        
        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self._init_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the multi-head attention."""
        batch_size = q.size(0)
        device = q.device
        
        # Ensure all inputs are on the same device
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        
        # Linear projections and reshape
        q = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scale query
        q = q / self.scale
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply scaling
        scores = scores * self.attention_scale
        
        # Apply sliding window mask if specified
        if self.window_size is not None:
            window_mask = self.create_sliding_window_mask(
                q.size(2), k.size(2), self.window_size, device=device
            )
            if mask is not None:
                mask = mask.to(device) & window_mask
            else:
                mask = window_mask
        elif mask is not None:
            mask = mask.to(device)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)
        
        return output
    
    @staticmethod
    def create_sliding_window_mask(
        seq_len_q: int,
        seq_len_k: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a sliding window mask for attention."""
        # Create position indices
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        
        # Calculate position differences and apply window constraint
        pos_diff = q_pos - k_pos
        window_mask = (pos_diff >= 0) & (pos_diff < window_size)
        
        return window_mask


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention used in MAG and MAL variants as short-term memory.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.1,
    ):
        """Initialize the sliding window attention."""
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the sliding window attention."""
        # Make sure input and mask are on the same device
        device = x.device
        if mask is not None:
            mask = mask.to(device)
            
        return self.attention(x, x, x, mask)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention used in the MAC variant.
    
    Ensures each token can only attend to itself and previous tokens.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize the causal self-attention."""
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
        )
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the causal self-attention."""
        seq_len = x.size(1)
        device = x.device
        
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        
        return self.attention(x, x, x, causal_mask) 