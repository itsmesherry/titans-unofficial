from titans.models.titans_base import TitansBase
import torch
import torch.nn as nn
from titans.utils.attention import SlidingWindowAttention
class TitansMAL(TitansBase):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        window_size: int = 128,
        dropout: float = 0.1,
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
        
        self.attention_layers = nn.ModuleList([
            SlidingWindowAttention(
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
        
        self.memory_norm = nn.LayerNorm(d_model)
        
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        x_with_persistent = self.persistent_memory.prepend_to_sequence(x)
        
        memory_output, _ = self.neural_memory(x_with_persistent, is_training=is_training)
        memory_output = self.memory_norm(memory_output)
        
        output = memory_output
        for i in range(self.n_layers):
            residual = output
            output = self.layer_norms1[i](output)
            output = self.attention_layers[i](output)
            output = self.dropout(output)
            output = residual + output
            
            residual = output
            output = self.layer_norms2[i](output)
            output = self.ffns[i](output)
            output = residual + output
        
        output = output[:, self.persistent_tokens:, :]
        
        output = self.output_projection(output)
        
        return output 