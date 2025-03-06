import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import math

from titans.models.titans_base import TitansBase
from titans.utils.attention import CausalSelfAttention

class TitansMAC(TitansBase):
    """
    Titans with Memory as Context (MAC) as described in Section 3.1 of the paper.
    
    In this variant, memory is treated as context for the attention mechanism,
    concatenating [persistent tokens || memory output || current context].
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
        """Initialize the Titans MAC model."""
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
        
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        memory_query: torch.Tensor,
        persistent_tokens: torch.Tensor,
        is_training: bool,
    ) -> torch.Tensor:
        """Process a single chunk following the MAC architecture."""
        device = chunk.device
        
        # Query the memory
        memory_output, _ = self.neural_memory(memory_query, is_training=is_training)
        
        # Ensure all tensors are on the same device
        persistent_tokens = persistent_tokens.to(device)
        memory_output = memory_output.to(device)
        
        # Concatenate [persistent tokens || memory output || chunk]
        augmented_chunk = torch.cat([persistent_tokens, memory_output, chunk], dim=1)
        
        # Process through transformer layers
        for j in range(self.n_layers):
            # Self-attention with residual connection and layer norm
            residual = augmented_chunk
            normalized = self.layer_norms1[j](augmented_chunk)
            attended = self.attention_layers[j](normalized)
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
        
    def _process_chunks_parallel(
        self,
        chunks: List[torch.Tensor],
        memory_queries: torch.Tensor,
        persistent_tokens: torch.Tensor,
        is_training: bool,
    ) -> List[torch.Tensor]:
        """Process multiple chunks in parallel for efficiency."""
        if not chunks:
            return []
            
        device = chunks[0].device
        
        # Concatenate chunks
        batched_chunks = torch.cat(chunks, dim=0)
        batched_queries = memory_queries.repeat(len(chunks), 1, 1)
        batched_persistent = persistent_tokens.repeat(len(chunks), 1, 1)
        
        # Ensure all tensors are on the same device
        batched_chunks = batched_chunks.to(device)
        batched_queries = batched_queries.to(device)
        batched_persistent = batched_persistent.to(device)
        
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
        """Forward pass through the Titans MAC model."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Get persistent tokens and ensure they're on the right device
        persistent_tokens = self.persistent_memory.forward(batch_size).to(device)
        
        # Process the sequence in chunks
        chunk_outputs = []
        current_chunks = []
        current_queries = []
        
        for i in range(0, seq_len, self.chunk_size):
            # Get current chunk
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :].to(device)
            
            # Prepare memory query (average of chunk)
            memory_query = chunk.mean(dim=1, keepdim=True).to(device)
            
            # Accumulate chunks for parallel processing
            current_chunks.append(chunk)
            current_queries.append(memory_query)
            
            # Process accumulated chunks if we have enough or if this is the last chunk
            if len(current_chunks) >= self.max_parallel_chunks or end_idx == seq_len:
                # Process chunks in parallel
                processed_chunks = self._process_chunks_parallel(
                    current_chunks,
                    torch.cat(current_queries, dim=0).to(device),
                    persistent_tokens,
                    is_training,
                )
                
                # Update memory with processed chunks if in training mode
                if is_training:
                    for processed_chunk in processed_chunks:
                        processed_chunk = processed_chunk.to(device)
                        self.neural_memory(processed_chunk, is_training=True)
                
                # Add to outputs
                chunk_outputs.extend(processed_chunks)
                
                # Clear current chunks
                current_chunks = []
                current_queries = []
        
        # Concatenate all chunk outputs and ensure they're on the right device
        if chunk_outputs:
            output = torch.cat([chunk.to(device) for chunk in chunk_outputs], dim=1)
            
            # Final projection
            output = self.output_projection(output)
            
            return output
        else:
            # Handle empty case
            return torch.zeros(batch_size, 0, self.d_model, device=device)
    