import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional sliding window and numerical stability enhancements.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
        attention_scale: float = 1.0,
    ):
        """
        Initialize the multi-head attention module.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            window_size: Size of the sliding window (None for full attention)
            dropout: Dropout probability
            attention_scale: Additional scaling factor for attention scores
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.attention_scale = attention_scale
        
        # Projection matrices with Xavier initialization
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self._init_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the multi-head attention with enhanced numerical stability.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor of shape (seq_len_q, seq_len_k)
            
        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scale query for numerical stability
        q = q / self.scale
        
        # Compute attention scores with enhanced numerical stability
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply additional stability measures
        scores = scores * self.attention_scale
        scores = scores / scores.abs().max().clamp(min=1)
        
        # Apply sliding window mask if specified
        if self.window_size is not None:
            window_mask = self.create_sliding_window_mask(
                q.size(2), k.size(2), self.window_size, device=q.device
            )
            if mask is not None:
                mask = mask & window_mask
            else:
                mask = window_mask
        
        # Apply mask and scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  # Use -1e4 instead of -inf for stability
        
        # Apply softmax with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Compute output with gradient scaling
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection with residual scaling
        output = self.W_O(output)
        output = output / (output.abs().mean().clamp(min=1))
        
        return output
    
    @staticmethod
    def create_sliding_window_mask(
        seq_len_q: int,
        seq_len_k: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a sliding window mask.
        
        Args:
            seq_len_q: Length of the query sequence
            seq_len_k: Length of the key sequence
            window_size: Size of the sliding window
            device: Device to create the mask on
            
        Returns:
            Boolean mask tensor of shape (seq_len_q, seq_len_k)
        """
        mask = torch.zeros(seq_len_q, seq_len_k, dtype=torch.bool, device=device)
        
        for i in range(seq_len_q):
            start = max(0, i - window_size + 1)
            end = i + 1  # +1 for causal attention
            mask[i, start:end] = True
            
        return mask
    
class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention as described in the paper.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the sliding window attention module.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            window_size: Size of the sliding window
            dropout: Dropout probability
        """
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
        """
        Forward pass through the sliding window attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.attention(x, x, x, mask)

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with optional sliding window.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize the causal self-attention module.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            window_size: Size of the sliding window (None for full attention)
            dropout: Dropout probability
        """
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
        """
        Forward pass through the causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Create causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        
        return self.attention(x, x, x, causal_mask) 