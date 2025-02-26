#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np
import time
import random
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titans import TitansMAC, TitansMAG, TitansMAL

# Import SimpleTokenizer directly from the file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from language_modeling import SimpleTokenizer


class TitansForClassification(nn.Module):
    """
    Titans model for text classification tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_classes: int,
        memory_depth: int = 2,
        persistent_tokens: int = 16,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
        model_type: str = "mac",  # "mac", "mag", or "mal"
    ):
        """
        Initialize the classification model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            n_layers: Number of layers
            n_heads: Number of attention heads
            num_classes: Number of classification classes
            memory_depth: Depth of the memory MLP
            persistent_tokens: Number of persistent memory tokens
            window_size: Size of the sliding window (None for full attention)
            dropout: Dropout probability
            model_type: Type of Titans model to use ("mac", "mag", or "mal")
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.max_seq_len = 512
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        
        # Titans model
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
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
        attention_mask: Optional[torch.FloatTensor] = None,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the classification model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            is_training: Whether to update the memory parameters
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}"
        
        # Get token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + position_emb
        x = self.dropout(x)
        
        # Forward pass through Titans model
        x = self.titans(x, is_training=is_training)
        
        # Pool sequence to get sentence representation (use [CLS] token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            cls_representation = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            cls_representation = torch.mean(x, dim=1)
        
        # Project to classes
        logits = self.classifier(cls_representation)
        
        return logits
    
    def reset_memory(self):
        """Reset the memory of the Titans model."""
        self.titans.reset_memory()


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: Tokenizer to encode the texts
            max_length: Maximum sequence length
        """
        assert len(texts) == len(labels), "Number of texts and labels must match"
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for efficiency
        self.encoded_texts = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.encoded_texts.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def collate_fn(batch):
    """
    Collate function for the DataLoader.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary of batched tensors
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    try:
        # Get max length in the batch
        max_len = max(len(sample["input_ids"]) for sample in batch)
        
        # Pre-allocate tensors
        batch_size = len(batch)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill tensors
        for i, sample in enumerate(batch):
            input_ids[i, :len(sample["input_ids"])] = sample["input_ids"]
            attention_mask[i, :len(sample["attention_mask"])] = sample["attention_mask"]
            labels[i] = sample["label"]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad_norm: Optional[float] = 1.0,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer instance
        device: Device to train on
        clip_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Check for invalid loss
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected: {loss.item()}. Skipping batch.")
            continue
        
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, is_training=False)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tokenizer (using BERT tokenizer as an example)
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Create sample dataset (replace with your own data)
    texts = [
        "The Titans architecture excels at memory-intensive tasks.",
        "Neural networks have revolutionized machine learning.",
        "Memory mechanisms improve model performance significantly.",
        "Deep learning models require careful optimization.",
        "The attention mechanism is a key innovation.",
        "Gradient descent optimizes neural network parameters.",
        "Transformers have become the standard in NLP.",
        "Model architecture affects learning capabilities.",
        "Data preprocessing is crucial for good results.",
        "Hyperparameter tuning can be challenging.",
    ]
    
    # Binary classification labels (0 or 1)
    labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # Create dataset
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=64)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
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
    num_classes = 2  # Binary classification
    memory_depth = 2
    persistent_tokens = 8
    window_size = 16
    
    # Create model
    print("Creating Titans classification model...")
    model = TitansForClassification(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        num_classes=num_classes,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
        model_type="mal",  # Use MAL variant
    )
    model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Training parameters
    num_epochs = 10
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    best_model_path = "best_classification_model.pt"
    
    # Train the model
    print("\nTraining the model...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_dataloader, device)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - "
              f"Time: {elapsed_time:.2f}s")
        
        # Reset memory between epochs
        model.reset_memory()
    
    # Final evaluation
    print("\nFinal evaluation:")
    val_loss, val_acc = evaluate(model, val_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main() 