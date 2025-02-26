import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from language_modeling import TitansForLanguageModeling


class MemoryEfficientDataset(Dataset):
    """
    Memory efficient dataset that supports both indexing and iteration.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 128,
        buffer_size: int = 1000,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            buffer_size: Size of the buffer for shuffling
        """
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        
        # Pre-tokenize a small batch for length calculation
        self.cached_items = []
        for text in texts[:buffer_size]:
            item = self._safe_tokenize(text)
            if item is not None:
                self.cached_items.append(item)
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item by index."""
        # Use cached item if available
        if idx < len(self.cached_items):
            return self.cached_items[idx]
        
        # Otherwise tokenize on-the-fly
        text = self.texts[idx]
        item = self._safe_tokenize(text)
        if item is None:
            # If tokenization fails, return a default item
            return self._create_default_item()
        return item
    
    def _create_default_item(self) -> Dict[str, torch.Tensor]:
        """Create a minimal default item if tokenization fails."""
        return {
            "input_ids": torch.tensor([0, 0], dtype=torch.long),
            "target_ids": torch.tensor([0, 0], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        }
    
    def _safe_tokenize(
        self,
        text: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Safely tokenize text with error handling.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary of tensors or None if tokenization failed
        """
        try:
            # Add special tokens and truncate
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Extract and verify tensors
            tokens = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            if tokens.size(0) < 2:  # Need at least 2 tokens for input/target
                return None
                
            # Create input and target sequences
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            attention_mask = attention_mask[:-1]
            
            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "attention_mask": attention_mask,
            }
            
        except Exception as e:
            print(f"Tokenization error: {e}")
            return None

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function with improved error handling and memory efficiency.
    
    Args:
        batch: List of dictionaries containing tensors
        
    Returns:
        Dictionary of batched tensors
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    try:
        # Get max length in the batch
        max_input_len = max(len(sample["input_ids"]) for sample in batch)
        max_target_len = max(len(sample["target_ids"]) for sample in batch)
        
        # Pre-allocate tensors
        batch_size = len(batch)
        padded_input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
        padded_target_ids = torch.zeros(batch_size, max_target_len, dtype=torch.long)
        attention_masks = torch.zeros(batch_size, max_input_len, dtype=torch.long)
        
        # Fill tensors
        for i, sample in enumerate(batch):
            input_ids = sample["input_ids"]
            target_ids = sample["target_ids"]
            attention_mask = sample["attention_mask"]
            
            # Copy data
            padded_input_ids[i, :len(input_ids)] = input_ids
            padded_target_ids[i, :len(target_ids)] = target_ids
            attention_masks[i, :len(attention_mask)] = attention_mask
        
        return {
            "input_ids": padded_input_ids,
            "target_ids": padded_target_ids,
            "attention_mask": attention_masks,
        }
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise

def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Compute the cross-entropy loss for language modeling.
    
    Args:
        logits: Logits from the model (batch_size, seq_len, vocab_size)
        target_ids: Target token IDs (batch_size, seq_len)
        ignore_index: Index to ignore in the loss computation (e.g., padding)
        
    Returns:
        Loss tensor
    """
    # Reshape logits
    _, _, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    
    # Reshape targets
    target_ids = target_ids.view(-1)
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(logits, target_ids)
    
    return loss

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    clip_grad_norm: Optional[float] = 1.0,
    accumulation_steps: int = 4,  # Number of steps to accumulate gradients
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        logits = model(input_ids)
        
        loss = compute_loss(logits, target_ids)
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps

        # Check for invalid loss before backward pass
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected: {loss.item()}. Skipping batch.")
            continue
            
        loss.backward()
        
        # Step optimization after accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping with value checking
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                print("Warning: NaN gradients detected. Skipping parameter update.")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate loss (use the scaled loss for logging)
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
    
    # Step the scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    # Return average loss, handling the case where all batches were skipped
    return total_loss / num_batches if num_batches > 0 else float('inf')

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        Average loss on the validation set
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            logits = model(input_ids, is_training=False)
            loss = compute_loss(logits, target_ids)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a sample from the model.
    
    Args:
        model: The model to generate from
        tokenizer: Tokenizer to encode/decode text
        prompt: Prompt to start generation
        max_length: Maximum length of the generated sequence
        temperature: Sampling temperature
        device: Device to generate on
        
    Returns:
        Generated text
    """
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    # Add safety checks in the generation loop
    try:
        generated_ids = model.generate(
            input_ids=prompt_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        return prompt  # Return original prompt if generation fails
    
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    
    return generated_text

def main():
    # Optimizer with warmup and scheduling
    from transformers import get_linear_schedule_with_warmup
    from transformers import BertTokenizerFast
    import random
    import numpy as np

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

    # Create sample dataset
    texts = [
        "The Titans architecture is designed to learn to memorize at test time.",
        "It consists of three hyper-heads: Core, Long-term Memory, and Persistent Memory.",
        "The Core processes the input sequence using self-attention mechanisms.",
        "Long-term Memory allows the model to access information from distant parts of the sequence.",
        "Persistent Memory enables the model to store and retrieve information across different inputs.",
        "Titans can be implemented in three variants: MAC, MAG, and MAL.",
        "MAC (Memory Access Control) uses a gating mechanism to control memory access.",
        "MAG (Memory Augmented Generation) enhances the generation process with memory.",
        "MAL (Memory Augmented Learning) improves learning by leveraging memory.",
        "The architecture is particularly effective for tasks requiring long-term dependencies.",
        "Titans models can adapt their memory during inference without requiring fine-tuning.",
        "This makes them suitable for continual learning scenarios.",
        "The persistent memory module allows for efficient knowledge retention.",
        "By separating computation and memory, Titans achieve better parameter efficiency.",
        "The architecture can be integrated with various transformer-based models.",
        "Experimental results show that Titans outperform standard transformers on memory-intensive tasks.",
        "The ability to learn at test time is a key advantage of the Titans architecture.",
        "Memory parameters are updated during inference to adapt to new information.",
        "This approach bridges the gap between training and deployment in real-world applications.",
        "Titans represent a step forward in developing more adaptive and efficient neural networks.",
    ]

    # Create dataset and dataloader
    dataset = MemoryEfficientDataset(texts, tokenizer, max_length=64)

    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model parameters
    vocab_size = tokenizer.vocab_size
    d_model = 128
    n_layers = 2
    n_heads = 4
    memory_depth = 2
    persistent_tokens = 8
    window_size = 16

    # Create model
    print("Creating Titans language model...")
    model = TitansForLanguageModeling(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
        model_type="mal",  # Use MAL variant
    )
    model.to(device)

    # Calculate number of training steps
    num_epochs = 10
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10  # 10% of training for warmup

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training parameters
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_path = "best_model.pt"

    # Train the model
    print("\nTraining the model...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer,
            scheduler,
            device,
            accumulation_steps=4
        )
        
        # Check for invalid loss
        if np.isnan(train_loss) or np.isinf(train_loss):
            print("Training loss became nan/inf. Stopping training.")
            break

        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"LR: {scheduler.get_last_lr()[0]:.2e} - "
              f"Time: {elapsed_time:.2f}s")
        
        # Generate a sample
        if (epoch + 1) % 5 == 0:
            try:
                prompt = "The Titans"
                generated = generate_sample(model, tokenizer, prompt, device=device)
                print(f"\nSample generation:")
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated}")
                print()
            except Exception as e:
                print(f"Error generating sample: {e}")

    # Final evaluation
    print("\nFinal evaluation:")
    val_loss = evaluate(model, val_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # Generate samples with different prompts
    prompts = [
        "The Titans",
        "Memory allows",
        "Learning at test",
    ]

    print("\nGenerating samples with different prompts:")
    for prompt in prompts:
        generated = generate_sample(model, tokenizer, prompt, device=device)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print()

    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()