import torch
import torch.nn as nn

from typing import Optional

from titans.models.titans_base import TitansBase
from titans.utils.attention import CausalSelfAttention

class TitansMAC(TitansBase):
    """
    Titans with Memory as a Context (MAC) as described in the paper.
    
    In this variant, memory is used as context for the attention mechanism.
    """
    
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
        """
        Initialize the Titans MAC model.
        
        Args:
            d_model: Dimension of the model
            n_layers: Number of layers
            n_heads: Number of attention heads
            memory_depth: Depth of the memory MLP
            persistent_tokens: Number of persistent memory tokens
            window_size: Size of the sliding window (None for full attention)
            dropout: Dropout probability
            chunk_size: Size of the chunks for processing
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
        
        self.chunk_size = chunk_size
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            CausalSelfAttention(
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
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the Titans MAC model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_training: Whether to update the memory parameters
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process the sequence in chunks
        outputs = []
        
        for i in range(0, seq_len, self.chunk_size):
            # Get current chunk
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            
            # Query the memory for the current chunk
            memory_query = chunk.mean(dim=1, keepdim=True)
            memory_output, _ = self.neural_memory(memory_query, is_training=is_training)
            
            # Prepend persistent memory and memory output to the chunk
            persistent_tokens = self.persistent_memory.forward(batch_size)
            augmented_chunk = torch.cat([persistent_tokens, memory_output, chunk], dim=1)
            
            # Process through attention layers
            for j in range(self.n_layers):
                # Self-attention
                residual = augmented_chunk
                augmented_chunk = self.layer_norms1[j](augmented_chunk)
                augmented_chunk = self.attention_layers[j](augmented_chunk)
                augmented_chunk = self.dropout(augmented_chunk)
                augmented_chunk = residual + augmented_chunk
                
                # Feed-forward
                residual = augmented_chunk
                augmented_chunk = self.layer_norms2[j](augmented_chunk)
                augmented_chunk = self.ffns[j](augmented_chunk)
                augmented_chunk = residual + augmented_chunk
            
            # Extract the chunk output (remove persistent and memory tokens)
            chunk_output = augmented_chunk[:, self.persistent_tokens + memory_output.size(1):, :]
            
            # Update memory with the processed chunk
            if is_training:
                self.neural_memory(chunk_output, is_training=True)
            
            outputs.append(chunk_output)
        
        # Concatenate the chunk outputs
        output = torch.cat(outputs, dim=1)
        
        # Final projection
        output = self.output_projection(output)
        
        return output
    