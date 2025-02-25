from src.attention import SlidingWindowAttention
from titans_base import TitansBase
import torch
import torch.nn as nn

class TitansMAG(TitansBase):
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
        
        self.gate_norm1 = nn.LayerNorm(d_model)
        self.gate_norm2 = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        
        x_with_persistent = self.persistent_memory.prepend_to_sequence(x)
        
        core_output = x_with_persistent
        for i in range(self.n_layers):
            residual = core_output
            core_output = self.layer_norms1[i](core_output)
            core_output = self.attention_layers[i](core_output)
            core_output = self.dropout(core_output)
            core_output = residual + core_output
            
            residual = core_output
            core_output = self.layer_norms2[i](core_output)
            core_output = self.ffns[i](core_output)
            core_output = residual + core_output
        
        core_output = core_output[:, self.persistent_tokens:, :]
        
        memory_output, _ = self.neural_memory(x, is_training=is_training)
        
        core_output_norm = self.gate_norm1(core_output)
        memory_output_norm = self.gate_norm2(memory_output)
        
        gate_input = torch.cat([core_output_norm, memory_output_norm], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        output = gate * core_output + (1 - gate) * memory_output
        
        output = self.output_projection(output)
        
        return output