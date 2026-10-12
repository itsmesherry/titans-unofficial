import torch
import torch.nn as nn
from typing import Optional
import math

from titans.models.titans_base import TitansBase
from titans.utils.attention import CausalSelfAttention
from titans.utils.memory import MemoryState

class TitansMAC(TitansBase):
    """
    Titans with Memory as Context (MAC) as described in Section 4.1 of the paper.
    
    In this variant, memory is treated as context for the attention mechanism,
    concatenating [persistent tokens || memory output || current context].
    
    The sequence is processed in chunks. For each chunk, every token queries the
    memory as it stood after the previous chunks, and the chunk's output is then
    written to memory. A token attends to the persistent tokens, to what the
    memory recalled for itself and for earlier tokens, and to earlier tokens,
    so the model is causal.
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
        max_parallel_chunks: int = 4,
        persist_memory: bool = False,
    ):
        """
        Initialize the Titans MAC model.
        
        max_parallel_chunks is ignored and kept only for backward compatibility:
        chunks are processed in order so each can read what earlier ones stored.
        """
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
        
        self.chunk_size = chunk_size
        self.max_parallel_chunks = max_parallel_chunks
        
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
            self._create_ffn()
            for _ in range(n_layers)
        ])
        
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
        
    def _context_mask(self, n_persistent: int, chunk_len: int, device: torch.device) -> torch.Tensor:
        """
        Attention mask for [persistent tokens || memory output || chunk].
        
        Causal, with one extra restriction: memory token j was recalled using
        chunk token j, so chunk token i may only see memory tokens j <= i.
        """
        total = n_persistent + 2 * chunk_len
        mask = torch.tril(torch.ones(total, total, dtype=torch.bool, device=device))
        
        chunk_start = n_persistent + chunk_len
        mask[chunk_start:, n_persistent:chunk_start] = torch.tril(
            torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=device)
        )
        return mask
        
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        state: MemoryState,
        persistent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Process a single chunk following the MAC architecture."""
        # Query the memory with every token of the chunk, without updating it
        memory_output = self.neural_memory.retrieve(chunk, state)
        
        # Concatenate [persistent tokens || memory output || chunk]
        augmented_chunk = torch.cat([persistent_tokens, memory_output, chunk], dim=1)
        mask = self._context_mask(persistent_tokens.size(1), chunk.size(1), chunk.device)
        
        # Process through transformer layers
        for j in range(self.n_layers):
            # Self-attention with residual connection and layer norm
            residual = augmented_chunk
            normalized = self.layer_norms1[j](augmented_chunk)
            attended = self.attention_layers[j].attention(normalized, normalized, normalized, mask)
            augmented_chunk = residual + self.dropout(attended)
            
            # Feed-forward with residual connection and layer norm
            residual = augmented_chunk
            normalized = self.layer_norms2[j](augmented_chunk)
            ffn_output = self.ffns[j](normalized)
            augmented_chunk = residual + ffn_output
        
        # Extract chunk output (remove persistent and memory tokens)
        memory_size = persistent_tokens.size(1) + memory_output.size(1)
        chunk_output = augmented_chunk[:, memory_size:, :]
        
        return chunk_output
    
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the Titans MAC model."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        persistent_tokens = self.persistent_memory(batch_size)
        (state,) = self._begin_memory(batch_size)
        
        # Process the sequence chunk by chunk, in order: each chunk reads what
        # the earlier chunks wrote to memory
        chunk_outputs = []
        
        for i in range(0, seq_len, self.chunk_size):
            # Get current chunk (the last one may be shorter)
            chunk = x[:, i:i + self.chunk_size, :]
            
            processed_chunk = self._process_chunk(chunk, state, persistent_tokens)
            
            # Update memory with the processed chunk if in training mode
            if is_training:
                state = self.neural_memory.update(processed_chunk, state)
            
            chunk_outputs.append(processed_chunk)
        
        self._end_memory([state])
        
        if chunk_outputs:
            output = torch.cat(chunk_outputs, dim=1)
            
            # Final projection
            output = self.output_projection(output)
            
            return output
        else:
            # Handle empty case
            return torch.zeros(batch_size, 0, self.d_model, device=device)
