import torch
from titans import TitansMAC, TitansMAG, TitansMAL

def test_model_on_device(model_class, device):
    """Test that a model can be moved to a device and process inputs."""
    print(f"Testing {model_class.__name__} on {device}...")
    
    # Create the model
    model = model_class(
        d_model=128,
        n_layers=2,
        n_heads=4,
        memory_depth=2,
        persistent_tokens=8
    )
    
    # Move the model to device using our special to_device method
    model = model.to_device(device)
    
    # Create a test input on the device
    x = torch.randn(2, 16, 128, device=device)
    
    # Forward pass
    try:
        output = model(x)
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output shape: {output.shape}, device: {output.device}")
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.device_count()} device(s)")
        device = "cuda"
    else:
        print("CUDA not available, testing on CPU")
        device = "cpu"
    
    # Test each model variant
    models = [TitansMAC, TitansMAG, TitansMAL]
    success = True
    
    for model_class in models:
        success = test_model_on_device(model_class, device) and success
    
    if success:
        print("\nAll models passed the test!")
    else:
        print("\nSome models failed the test.")

if __name__ == "__main__":
    main() 