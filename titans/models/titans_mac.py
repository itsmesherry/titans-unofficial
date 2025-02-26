import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import math

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
        max_parallel_chunks: int = 4,
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
            max_parallel_chunks: Maximum number of chunks to process in parallel
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
        
        # Feed-forward networks with improved initialization
        self.ffns = nn.ModuleList([
            self._create_ffn()
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Initialize all weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with improved scaling."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def _create_ffn(self) -> nn.Sequential:
        """Create a feed-forward network with improved architecture."""
        dropout_prob = self.dropout.p if isinstance(self.dropout, nn.Dropout) else self.dropout
        return nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.Dropout(dropout_prob),
        )
        
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        memory_query: torch.Tensor,
        persistent_tokens: torch.Tensor,
        is_training: bool,
    ) -> torch.Tensor:
        """
        Process a single chunk of the input sequence.
        
        Args:
            chunk: Input chunk tensor
            memory_query: Memory query tensor
            persistent_tokens: Persistent memory tokens
            is_training: Whether in training mode
            
        Returns:
            Processed chunk tensor
        """
        # Query the memory
        memory_output, _ = self.neural_memory(memory_query, is_training=is_training)
        
        # Prepend tokens
        augmented_chunk = torch.cat([persistent_tokens, memory_output, chunk], dim=1)
        
        # Process through attention layers
        for j in range(self.n_layers):
            # Self-attention with residual connection and layer norm
            residual = augmented_chunk
            augmented_chunk = self.layer_norms1[j](augmented_chunk)
            augmented_chunk = self.attention_layers[j](augmented_chunk)
            augmented_chunk = residual + self.dropout(augmented_chunk)
            
            # Feed-forward with residual connection and layer norm
            residual = augmented_chunk
            augmented_chunk = self.layer_norms2[j](augmented_chunk)
            augmented_chunk = self.ffns[j](augmented_chunk)
            augmented_chunk = residual + augmented_chunk
        
        # Extract chunk output
        chunk_output = augmented_chunk[:, self.persistent_tokens + memory_output.size(1):, :]
        
        return chunk_output
        
    def _process_chunks_parallel(
        self,
        chunks: List[torch.Tensor],
        memory_queries: torch.Tensor,
        persistent_tokens: torch.Tensor,
        is_training: bool,
    ) -> List[torch.Tensor]:
        """
        Process multiple chunks in parallel.
        
        Args:
            chunks: List of input chunk tensors
            memory_queries: Memory query tensors
            persistent_tokens: Persistent memory tokens
            is_training: Whether in training mode
            
        Returns:
            List of processed chunk tensors
        """
        # Concatenate chunks for parallel processing
        batched_chunks = torch.cat(chunks, dim=0)
        batched_queries = memory_queries.repeat(len(chunks), 1, 1)
        batched_persistent = persistent_tokens.repeat(len(chunks), 1, 1)
        
        # Process all chunks at once
        batched_output = self._process_chunk(
            batched_chunks,
            batched_queries,
            batched_persistent,
            is_training,
        )
        
        # Split back into individual chunks
        chunk_size = chunks[0].size(0)
        return list(batched_output.split(chunk_size, dim=0))
    
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the Titans MAC model with optimized chunk processing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_training: Whether to update the memory parameters
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get persistent tokens once
        persistent_tokens = self.persistent_memory.forward(batch_size)
        
        # Process the sequence in chunks
        chunk_outputs = []
        current_chunks = []
        current_queries = []
        
        for i in range(0, seq_len, self.chunk_size):
            # Get current chunk
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            
            # Prepare memory query
            memory_query = chunk.mean(dim=1, keepdim=True)
            
            # Accumulate chunks for parallel processing
            current_chunks.append(chunk)
            current_queries.append(memory_query)
            
            # Process accumulated chunks if we have enough or if this is the last chunk
            if len(current_chunks) >= self.max_parallel_chunks or end_idx == seq_len:
                # Process chunks in parallel
                processed_chunks = self._process_chunks_parallel(
                    current_chunks,
                    torch.cat(current_queries, dim=0),
                    persistent_tokens,
                    is_training,
                )
                
                # Update memory with processed chunks if in training
                if is_training:
                    for processed_chunk in processed_chunks:
                        self.neural_memory(processed_chunk, is_training=True)
                
                # Add to outputs
                chunk_outputs.extend(processed_chunks)
                
                # Clear current chunks
                current_chunks = []
                current_queries = []
        
        # Concatenate all chunk outputs
        output = torch.cat(chunk_outputs, dim=1)
        
        # Final projection with gradient scaling
        output = self.output_projection(output)
        output = output / output.abs().mean().clamp(min=1)
        
        return output
    