import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tokens: int,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tokens = num_tokens
        
        self.memory_tokens = nn.Parameter(torch.randn(1, num_tokens, d_model))
        
        nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)
        
    def forward(self, batch_size: int) -> torch.Tensor:
        return self.memory_tokens.expand(batch_size, -1)
    
    def prepend_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(1)
        memory_tokens = self.forward(batch_size)
        return torch.cat([memory_tokens, x]) 