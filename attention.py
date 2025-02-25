import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        
        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply sliding window mask if specified
        if self.window_size is not None:
            window_mask = self.create_sliding_window_mask(
                q.size(2), k.size(2), self.window_size, device=q.device
            )
            if mask is not None:
                mask = mask & window_mask
            else:
                mask = window_mask
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
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
        mask = torch.zeros(seq_len_q, seq_len_k, dtype=torch.bool, device=device)
        
        for i in range(seq_len_q):
            start = max(0, i - window_size + 1)
            end = i + 1  # +1 for causal attention
            mask[i, start:end] = True
            
        return mask
    
class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.1,
    ):
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
        return self.attention(x, x, x, mask)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
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
        seq_len = x.size(1)
        
        # Create causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        
        return self.attention(x, x, x, causal_mask) 