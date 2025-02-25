import torch
import torch.nn as nn

from titans.models.titans_base import TitansBase
from titans.utils.attention import SlidingWindowAttention


class TitansMAL(TitansBase):
    """
    Titans with Memory as a Layer (MAL) as described in the paper.
    
    In this variant, memory is used as a layer in the architecture.
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
        Initialize the Titans MAL model.
        
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
        
        # Memory layer norm
        self.memory_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the Titans MAL model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_training: Whether to update the memory parameters
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Prepend persistent memory tokens
        x_with_persistent = self.persistent_memory.prepend_to_sequence(x)
        
        # Process through neural memory first
        memory_output, _ = self.neural_memory(x_with_persistent, is_training=is_training)
        memory_output = self.memory_norm(memory_output)
        
        # Process through attention layers
        output = memory_output
        for i in range(self.n_layers):
            # Self-attention
            residual = output
            output = self.layer_norms1[i](output)
            output = self.attention_layers[i](output)
            output = self.dropout(output)
            output = residual + output
            
            # Feed-forward
            residual = output
            output = self.layer_norms2[i](output)
            output = self.ffns[i](output)
            output = residual + output
        
        # Remove persistent tokens
        output = output[:, self.persistent_tokens:, :]
        
        # Final projection
        output = self.output_projection(output)
        
        return output 
    