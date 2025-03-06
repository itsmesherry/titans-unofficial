import torch
import torch.nn as nn
import math

from titans.models.titans_base import TitansBase
from titans.utils.attention import SlidingWindowAttention


class TitansMAL(TitansBase):
    """
    Titans with Memory as Layer (MAL) as described in Section 3.3 of the paper.
    
    In this variant, memory is used as an additional transformation layer
    in the transformer architecture between attention and feed-forward layers.
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
        """Initialize the Titans MAL model."""
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            memory_depth=memory_depth,
            persistent_tokens=persistent_tokens,
            window_size=window_size,
            dropout=dropout,
        )
        
        # Sliding window attention
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
            self._create_ffn()
            for _ in range(n_layers)
        ])
        
        # Memory layer norm
        self.memory_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def _create_ffn(self) -> nn.Sequential:
        """Create a feed-forward network."""
        dropout_prob = self.dropout.p if isinstance(self.dropout, nn.Dropout) else self.dropout
        return nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.Dropout(dropout_prob),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the Titans MAL model."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Ensure input is on the correct device
        x = x.to(device)
        
        # Add persistent memory tokens if needed
        if self.persistent_tokens > 0:
            x_with_memory = self.persistent_memory.prepend_to_sequence(x)
        else:
            x_with_memory = x
        
        # Ensure consistent device placement
        output = x_with_memory.to(device)
        
        # Process through transformer layers with memory as a layer
        for i in range(self.n_layers):
            # 1. Self-attention with residual connection and layer norm
            residual = output
            normalized = self.layer_norms1[i](output)
            attended = self.attention_layers[i](normalized)
            output = residual + self.dropout(attended)
            
            # 2. Memory layer (unique to MAL architecture)
            memory_input = self.memory_norm(output).to(device)
            memory_output, _ = self.neural_memory(memory_input, is_training=is_training)
            
            # Ensure memory_output is on the correct device
            memory_output = memory_output.to(device)
            
            # Add memory contribution (residual connection)
            output = output + memory_output
            
            # 3. Feed-forward with residual connection and layer norm
            residual = output
            normalized = self.layer_norms2[i](output)
            ffn_output = self.ffns[i](normalized)
            output = residual + ffn_output
        
        # Remove persistent tokens if they were added
        if self.persistent_tokens > 0:
            output = output[:, self.persistent_tokens:, :]
        
        # Final projection
        output = self.output_projection(output.to(device))
        
        return output 
    