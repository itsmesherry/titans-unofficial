import torch
import torch.nn as nn

from titans.models.titans_base import TitansBase
from titans.utils.attention import SlidingWindowAttention


class TitansMAG(TitansBase):
    """
    Titans with Memory as a Gate (MAG) as described in the paper.
    
    In this variant, memory is combined with the core branch using a gating mechanism.
    """

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
        """
        Initialize the Titans MAG model.
        
        Args:
            d_model: Dimension of the model
            n_layers: Number of layers
            n_heads: Number of attention heads
            memory_depth: Depth of the memory MLP
            persistent_tokens: Number of persistent memory tokens
            window_size: Size of the sliding window
            dropout: Dropout probability
        """
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            memory_depth=memory_depth,
            persistent_tokens=persistent_tokens,
            window_size=window_size,
            dropout=dropout,
        )
        
        # Sliding window attention layers
        self.attention_layers = nn.ModuleList([
            SlidingWindowAttention(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        
        # Gate mechanism
        self.gate_norm1 = nn.LayerNorm(d_model)
        self.gate_norm2 = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the Titans MAG model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_training: Whether to update the memory parameters
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        
        # Prepend persistent memory tokens
        x_with_persistent = self.persistent_memory.prepend_to_sequence(x)
        
        # Process through attention layers (core branch)
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
        
        # Remove persistent tokens from core output
        core_output = core_output[:, self.persistent_tokens:, :]
        
        # Process through neural memory (memory branch)
        memory_output, _ = self.neural_memory(x, is_training=is_training)
        
        # Apply gating mechanism
        core_output_norm = self.gate_norm1(core_output)
        memory_output_norm = self.gate_norm2(memory_output)
        
        # Concatenate and project for gating
        gate_input = torch.cat([core_output_norm, memory_output_norm], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        # Combine outputs with gating
        output = gate * core_output + (1 - gate) * memory_output
        
        # Final projection
        output = self.output_projection(output)
        
        return output
    