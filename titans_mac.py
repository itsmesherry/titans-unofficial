from typing import Optional

import torch
import torch.nn as nn

from titans_base import TitansBase
from src.memory import CausalSelfAttention

class TitansMAC(TitansBase):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
        chunk_size: int = 128,
    ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            memory_depth=memory_depth,
            persistent_tokens=persistent_tokens,
            window_size=window_size,
            dropout=dropout,
        )
        
        self.chunk_size = chunk_size
        
        self.attention_layers = nn.ModuleList([
            CausalSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        outputs = []
        
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            
            memory_query = chunk.mean(dim=1, keepdim=True)
            memory_output, _ = self.neural_memory(memory_query, is_training=is_training)
            
            persistent_tokens = self.persistent_memory.forward(batch_size)
            augmented_chunk = torch.cat([persistent_tokens, memory_output, chunk], dim=1)
            
            for j in range(self.n_layers):
                residual = augmented_chunk
                augmented_chunk = self.layer_norms1[j](augmented_chunk)
                augmented_chunk = self.attention_layers[j](augmented_chunk)
                augmented_chunk = self.dropout(augmented_chunk)
                augmented_chunk = residual + augmented_chunk
                
                residual = augmented_chunk
                augmented_chunk = self.layer_norms2[j](augmented_chunk)
                augmented_chunk = self.ffns[j](augmented_chunk)
                augmented_chunk = residual + augmented_chunk
            
            chunk_output = augmented_chunk[:, self.persistent_tokens + memory_output.size(1):, :]
            
            if is_training:
                self.neural_memory(chunk_output, is_training=True)
            
            outputs.append(chunk_output)
        
        output = torch.cat(outputs, dim=1)
        
        output = self.output_projection(output)
        
        return output