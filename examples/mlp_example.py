"""
Example: Quantized MLP with Hierarchical Nested Lattice Quantization

This example demonstrates how to use CoSet for training a quantized MLP
with efficient gradient communication in distributed training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from coset import LatticeConfig, LatticeType, QuantizedMLP, QuantizedGradientHook


def create_synthetic_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create synthetic dataset for demonstration."""
    # Generate random input data
    X = torch.randn(num_samples, input_dim)
    
    # Generate random labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


def train_quantized_mlp():
    """Train a quantized MLP using CoSet."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create lattice configuration
    config = LatticeConfig(
        type=LatticeType.Z2,  # Use Z2 lattice (2D)
        radix=4,
        num_layers=3,
        beta=1.0,
        alpha=1.0
    )
    
    print(f"Lattice configuration: {config}")
    
    # Create small quantized MLP for quick testing
    model = QuantizedMLP(
        input_dim=32,       # Small 32D input
        hidden_dims=[16, 8], # Just 2 hidden layers
        output_dim=4,       # 4 classes
        config=config,
        activation="ReLU",
        dropout=0.1,
        use_ste=True,
        use_lookup_tables=True
    ).to(device)
    
    print(f"Model: {model}")
    
    # Create synthetic dataset with small dimensions
    X_train, y_train = create_synthetic_data(200, 32, 4)  # Small dataset
    X_test, y_test = create_synthetic_data(50, 32, 4)     # Small test set
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=16,  # Smaller batch size
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=16,  # Smaller batch size
        shuffle=False
    )
    
    # Create quantized gradient hook for distributed training
    hook = QuantizedGradientHook(
        config=config,
        communication_depth=1,
        compression_enabled=True,
        timing_enabled=True
    )
    
    # Register hook (in real distributed training, this would be done automatically)
    # model.register_comm_hook(None, hook)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 2  # Just 2 epochs for quick testing
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
        
        # Print quantization statistics
        stats = model.get_quantization_stats()
        print(f'Quantization error (layer 0): {stats["layer_0"]["quantization_error"]:.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test accuracy: {accuracy:.2f}%')
    
    # Print compression statistics
    compression_stats = hook.get_compression_stats()
    print(f'Compression ratio: {compression_stats["compression_ratio"]:.2f}x')
    
    # Print timing statistics
    timing_stats = hook.get_timing_stats()
    if timing_stats['num_calls'] > 0:
        print(f'Average quantization time: {timing_stats["avg_quantization_time"]:.4f}s')
        print(f'Average communication time: {timing_stats["avg_communication_time"]:.4f}s')
        print(f'Average dequantization time: {timing_stats["avg_dequantization_time"]:.4f}s')


def demonstrate_encoding_decoding():
    """Demonstrate core encoding/decoding operations."""
    
    print("\n=== Encoding/Decoding Demonstration ===")
    
    # Create lattice configuration
    config = LatticeConfig(
        type=LatticeType.Z2,  # Use Z2 lattice (2D)
        radix=4,
        num_layers=3
    )
    
    # Create quantizer
    from coset.quantizers import LatticeQuantizer
    quantizer = LatticeQuantizer(config)
    
    # Create test input with arbitrary dimensions
    input_tensor = torch.randn(32, 8)  # 8D input with 2D lattice (product quantization)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Quantize input
    quantized = quantizer.quantize(input_tensor)
    print(f"Quantized tensor shape: {quantized.shape}")
    
    # Test quantize_to_depth for indices
    quantized_depth, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
    print(f"Indices shape: {indices.shape}")
    print(f"Indices range: [{indices.min()}, {indices.max()}]")
    
    # Decode from indices
    reconstructed = quantizer.decode_from_depth(indices, source_depth=1)
    print(f"Reconstructed tensor shape: {reconstructed.shape}")
    
    # Compute quantization error
    quantization_error = torch.mean(torch.abs(input_tensor - reconstructed))
    print(f"Quantization error: {quantization_error:.4f}")
    
    # Demonstrate packing encoding
    print("\n--- Packing Encoding ---")
    encoded = quantizer.packing_encode(input_tensor, packing_radix=4, depth=2)
    print(f"Encoded tensor shape: {encoded.shape}")
    print(f"Encoded tensor dtype: {encoded.dtype}")
    
    # Decode from packing
    decoded = quantizer.packing_decode(encoded, packing_radix=4, depth=2)
    print(f"Decoded tensor shape: {decoded.shape}")
    
    # Demonstrate lookup table operations
    print("\n--- Lookup Table Operations ---")
    x_indices = torch.randint(0, 256, (32,))
    y_indices = torch.randint(0, 256, (32,))
    
    dot_products = quantizer.lookup_dot_product(x_indices, y_indices)
    print(f"Dot products shape: {dot_products.shape}")
    print(f"Dot products range: [{dot_products.min():.3f}, {dot_products.max():.3f}]")


def demonstrate_distributed_training():
    """Demonstrate distributed training with quantized gradients."""
    
    print("\n=== Distributed Training Demonstration ===")
    
    # Create lattice configuration
    config = LatticeConfig(
        type=LatticeType.Z2,  # Use Z2 lattice (2D)
        radix=4,
        num_layers=3
    )
    
    # Create quantized gradient hook
    hook = QuantizedGradientHook(
        config=config,
        communication_depth=1,
        compression_enabled=True,
        timing_enabled=True
    )
    
    # Simulate gradient communication
    print("Simulating gradient communication...")
    
    # Create mock gradients with arbitrary dimensions
    gradients = torch.randn(1000, 512)  # 512D gradients (product quantization)
    print(f"Original gradients shape: {gradients.shape}")
    print(f"Original gradients size: {gradients.numel() * gradients.element_size()} bytes")
    
    # Quantize gradients
    quantized_gradients = hook.compressor.compress_gradients(gradients, depth=1)
    print(f"Quantized gradients shape: {quantized_gradients.shape}")
    print(f"Quantized gradients size: {quantized_gradients.numel() * quantized_gradients.element_size()} bytes")
    
    # Compute compression ratio
    original_size = gradients.numel() * gradients.element_size()
    compressed_size = quantized_gradients.numel() * quantized_gradients.element_size()
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Dequantize gradients
    reconstructed_gradients = hook.compressor.decompress_gradients(quantized_gradients, depth=1)
    print(f"Reconstructed gradients shape: {reconstructed_gradients.shape}")
    
    # Compute reconstruction error
    reconstruction_error = torch.mean(torch.abs(gradients - reconstructed_gradients))
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # Print compression statistics
    stats = hook.get_compression_stats()
    print(f"Compression statistics: {stats}")


if __name__ == "__main__":
    print("CoSet: Hierarchical Nested Lattice Quantization Example")
    print("=" * 60)
    
    # Demonstrate core operations
    demonstrate_encoding_decoding()
    
    # Demonstrate distributed training
    demonstrate_distributed_training()
    
    # Train quantized MLP
    print("\n=== Training Quantized MLP ===")
    train_quantized_mlp()
    
    print("\nExample completed successfully!")
