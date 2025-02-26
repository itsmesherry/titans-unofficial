import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titans import TitansMAC, TitansMAG, TitansMAL


class TitansForLanguageModeling(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
        model_type: str = "mac",  # "mac", "mag", or "mal"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.max_seq_len = 2048
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        
        if model_type.lower() == "mac":
            self.titans = TitansMAC(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                memory_depth=memory_depth,
                persistent_tokens=persistent_tokens,
                window_size=window_size,
                dropout=dropout,
            )
        elif model_type.lower() == "mag":
            self.titans = TitansMAG(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                memory_depth=memory_depth,
                persistent_tokens=persistent_tokens,
                window_size=window_size if window_size is not None else 128,
                dropout=dropout,
            )
        elif model_type.lower() == "mal":
            self.titans = TitansMAL(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                memory_depth=memory_depth,
                persistent_tokens=persistent_tokens,
                window_size=window_size if window_size is not None else 128,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}"
        
        token_emb = self.token_embedding(input_ids)
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(positions)
        
        x = token_emb + position_emb
        x = self.dropout(x)
        
        x = self.titans(x, is_training=is_training)
        
        logits = self.output_projection(x)
        
        return logits
    
    def reset_memory(self):
        self.titans.reset_memory()
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        
        self.reset_memory()
        
        for _ in range(max_length - input_ids.shape[1]):
            logits = self.forward(generated, is_training=False)
            
            next_token_logits = logits[:, -1, :]
            
            next_token_logits = torch.clamp(next_token_logits, -100, 100)

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k is not None:
                indices_to_remove = torch.topk(next_token_logits, k=top_k, dim=-1)[0]
                indices_to_remove = indices_to_remove[:, -1].unsqueeze(-1).expand_as(next_token_logits)
                next_token_logits = torch.where(
                    next_token_logits < indices_to_remove,
                    torch.ones_like(next_token_logits) * -float("Inf"),
                    next_token_logits,
                )
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = torch.where(
                    indices_to_remove,
                    torch.ones_like(next_token_logits) * -float("Inf"),
                    next_token_logits,
                )
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
                
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class SimpleTokenizer:
    def __init__(self):
        self.chars = [chr(i) for i in range(32, 127)]
        self.vocab = {char: i for i, char in enumerate(self.chars)}
        self.vocab["<pad>"] = len(self.chars)
        self.vocab["<unk>"] = len(self.chars) + 1
        
        self.id_to_token = {i: char for char, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.id_to_token.get(token_id, "<unk>") for token_id in token_ids])


def main():
    torch.manual_seed(42)
    
    tokenizer = SimpleTokenizer()
    
    vocab_size = tokenizer.vocab_size
    d_model = 128
    n_layers = 2
    n_heads = 4
    memory_depth = 2
    persistent_tokens = 8
    window_size = 16
    
    print("Creating Titans language model...")
    model = TitansForLanguageModeling(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
        model_type="mac",
    )
    
    text = "The Titans architecture is designed to learn to memorize at test time."
    print(f"\nSample text: {text}")
    
    input_ids = tokenizer.encode(text)
    print(f"Encoded text: {input_ids}")
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    logits = model(input_tensor)
    print(f"Logits shape: {logits.shape}")
    
    print("\nGenerating text...")
    prompt = "The Titans"
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    
    generated_ids = model.generate(
        input_ids=prompt_tensor,
        max_length=50,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    model.reset_memory()
    
    print("\nDone!")


if __name__ == "__main__":
    main() 