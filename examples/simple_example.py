import torch
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titans import TitansMAC, TitansMAG, TitansMAL


def main():
    torch.manual_seed(42)
    
    # Model parameters
    d_model = 128
    n_layers = 2
    n_heads = 4
    memory_depth = 2
    persistent_tokens = 8
    window_size = 16
    
    # Create models
    print("Creating Titans MAC model...")
    mac_model = TitansMAC(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
    )
    
    print("Creating Titans MAG model...")
    mag_model = TitansMAG(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
    )
    
    print("Creating Titans MAL model...")
    mal_model = TitansMAL(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        memory_depth=memory_depth,
        persistent_tokens=persistent_tokens,
        window_size=window_size,
    )
    
    # Generate random input
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass through models
    print("\nRunning forward pass through MAC model...")
    mac_output = mac_model(x)
    print(f"MAC output shape: {mac_output.shape}")
    
    print("\nRunning forward pass through MAG model...")
    mag_output = mag_model(x)
    print(f"MAG output shape: {mag_output.shape}")
    
    print("\nRunning forward pass through MAL model...")
    mal_output = mal_model(x)
    print(f"MAL output shape: {mal_output.shape}")
    
    # Reset memory
    print("\nResetting memory in all models...")
    mac_model.reset_memory()
    mag_model.reset_memory()
    mal_model.reset_memory()
    
    print("\nDone!")


if __name__ == "__main__":
    main() 