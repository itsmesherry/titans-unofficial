import torch
import torch.nn as nn
import math

from titans.models.titans_base import TitansBase
from titans.utils.attention import SlidingWindowAttention


class TitansMAG(TitansBase):
    """
    Titans with Memory as Gating (MAG) as described in Section 4.2 of the paper.
    
    Uses sliding window attention as short-term memory and neural memory as
    long-term memory, combining them with a gating mechanism.
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
        persist_memory: bool = False,
    ):
        """Initialize the Titans MAG model."""
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            memory_depth=memory_depth,
            persistent_tokens=persistent_tokens,
            window_size=window_size,
            dropout=dropout,
            persist_memory=persist_memory,
        )
        
        # Sliding window attention (short-term memory)
        self.attention_layers = nn.ModuleList([
            SlidingWindowAttention(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout,
                global_prefix=persistent_tokens,
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
        
        # Gating mechanism
        self.gate_norm1 = nn.LayerNorm(d_model)
        self.gate_norm2 = nn.LayerNorm(d_model)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        
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
        """Forward pass through the Titans MAG model."""
        batch_size = x.size(0)
        
        # 1. Process through sliding window attention (short-term memory).
        # Persistent tokens are prepended and stay visible from every position
        short_term_memory = x
        if self.persistent_tokens > 0:
            short_term_memory = self.persistent_memory.prepend_to_sequence(x)
        
        for i in range(self.n_layers):
            # Self-attention with residual connection and layer norm
            residual = short_term_memory
            normalized = self.layer_norms1[i](short_term_memory)
            attended = self.attention_layers[i](normalized)
            short_term_memory = residual + self.dropout(attended)
            
            # Feed-forward with residual connection and layer norm
            residual = short_term_memory
            normalized = self.layer_norms2[i](short_term_memory)
            ffn_output = self.ffns[i](normalized)
            short_term_memory = residual + ffn_output
        
        # Extract original sequence part
        short_term_memory = short_term_memory[:, self.persistent_tokens:, :]
        
        # 2. Process through neural memory (long-term memory)
        (state,) = self._begin_memory(batch_size)
        long_term_memory, state = self.neural_memory(x, state, is_training=is_training)
        self._end_memory([state])
        
        # 3. Apply gating mechanism to combine memories
        normalized_short = self.gate_norm1(short_term_memory)
        normalized_long = self.gate_norm2(long_term_memory)
        
        # Create gate
        combined = torch.cat([normalized_short, normalized_long], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        
        # Combine with gate
        output = gate * short_term_memory + (1 - gate) * long_term_memory
        
        # Final projection
        output = self.output_projection(output)
        
        return output
    