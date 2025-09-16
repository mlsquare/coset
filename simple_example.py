#!/usr/bin/env python3
"""
Simple CoSet Example: Hierarchical Nested Lattice Quantization

This example demonstrates the core functionality of CoSet:
1. Basic quantization operations
2. Quantized MLP training
3. Gradient compression
4. Performance comparison

Run this example with: python simple_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from coset import LatticeConfig, LatticeType, LatticeQuantizer, QuantizedMLP, QuantizedGradientHook


def demo_basic_quantization():
    """Demonstrate basic quantization operations."""
    print("🔬 Demo 1: Basic Quantization Operations")
    print("-" * 50)
    
    # Create lattice configuration
    config = LatticeConfig(
        type=LatticeType.HNLQ,
        radix=4,
        num_layers=3,
        lattice_dim=8,
        learnable_scales=True
    )
    print(f"Configuration: {config}")
    
    # Create quantizer
    quantizer = LatticeQuantizer(config)
    
    # Create test input
    input_tensor = torch.randn(32, 8)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Quantize
    quantized, indices = quantizer.quantize(input_tensor, depth=1)
    print(f"Quantized shape: {quantized.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Indices range: [{indices.min()}, {indices.max()}]")
    
    # Dequantize
    reconstructed = quantizer.dequantize(indices, depth=1)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate quantization error
    error = torch.mean(torch.abs(input_tensor - reconstructed))
    print(f"Quantization error: {error:.4f}")
    
    # Test radix-q encoding
    print("\n📦 Radix-Q Encoding:")
    encoded = quantizer.radixq_encode(input_tensor, radix=4, depth=2)
    decoded = quantizer.radixq_decode(encoded, radix=4, depth=2)
    print(f"Encoded shape: {encoded.shape}, dtype: {encoded.dtype}")
    print(f"Decoded shape: {decoded.shape}")
    
    print("✅ Basic quantization demo completed!\n")


def demo_quantized_mlp():
    """Demonstrate quantized MLP training."""
    print("🧠 Demo 2: Quantized MLP Training")
    print("-" * 50)
    
    # Create configuration
    config = LatticeConfig(
        type=LatticeType.HNLQ,
        radix=4,
        num_layers=3,
        lattice_dim=8
    )
    
    # Create quantized MLP
    model = QuantizedMLP(
        input_dim=784,      # 28x28 images
        hidden_dims=[512, 256, 128],
        output_dim=10,      # 10 classes
        config=config,
        activation='ReLU',
        dropout=0.1,
        use_ste=True,
        use_lookup_tables=False
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create synthetic dataset
    X_train = torch.randn(1000, 784)
    y_train = torch.randint(0, 10, (1000,))
    X_test = torch.randn(200, 784)
    y_test = torch.randint(0, 10, (200,))
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=32,
        shuffle=False
    )
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n🚀 Training...")
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Evaluation
    print("\n📊 Evaluation...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # Show quantization statistics
    stats = model.get_quantization_stats()
    print(f"\n📈 Quantization Statistics:")
    for i, layer_stats in stats.items():
        print(f"  {i}: error = {layer_stats['quantization_error']:.4f}")
    
    print("✅ Quantized MLP demo completed!\n")


def demo_gradient_compression():
    """Demonstrate gradient compression for distributed training."""
    print("🗜️ Demo 3: Gradient Compression")
    print("-" * 50)
    
    # Create configuration
    config = LatticeConfig(
        type=LatticeType.HNLQ,
        radix=4,
        num_layers=3,
        lattice_dim=8
    )
    
    # Create quantized gradient hook
    hook = QuantizedGradientHook(
        config=config,
        communication_depth=1,
        compression_enabled=True,
        timing_enabled=True
    )
    
    # Create mock gradients
    gradients = torch.randn(1000, 8)
    print(f"Original gradients shape: {gradients.shape}")
    print(f"Original gradients size: {gradients.numel() * gradients.element_size():,} bytes")
    
    # Compress gradients
    compressed_gradients = hook.compressor.compress_gradients(gradients, depth=1)
    print(f"Compressed gradients shape: {compressed_gradients.shape}")
    print(f"Compressed gradients size: {compressed_gradients.numel() * compressed_gradients.element_size():,} bytes")
    
    # Calculate compression ratio
    original_size = gradients.numel() * gradients.element_size()
    compressed_size = compressed_gradients.numel() * compressed_gradients.element_size()
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Decompress gradients
    decompressed_gradients = hook.compressor.decompress_gradients(compressed_gradients, depth=1)
    print(f"Decompressed gradients shape: {decompressed_gradients.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = torch.mean(torch.abs(gradients - decompressed_gradients))
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # Show compression statistics
    stats = hook.get_compression_stats()
    print(f"\n📊 Compression Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("✅ Gradient compression demo completed!\n")


def demo_performance_comparison():
    """Compare performance between quantized and standard MLPs."""
    print("⚡ Demo 4: Performance Comparison")
    print("-" * 50)
    
    # Create configuration
    config = LatticeConfig(
        type=LatticeType.HNLQ,
        radix=4,
        num_layers=3,
        lattice_dim=8
    )
    
    # Create quantized MLP
    quantized_mlp = QuantizedMLP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        config=config,
        use_lookup_tables=False
    )
    
    # Create standard MLP
    standard_mlp = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Test data
    input_tensor = torch.randn(1000, 784)
    
    # Benchmark quantized MLP
    print("Testing Quantized MLP...")
    start_time = time.time()
    for _ in range(50):
        with torch.no_grad():
            output = quantized_mlp(input_tensor)
    quantized_time = time.time() - start_time
    
    # Benchmark standard MLP
    print("Testing Standard MLP...")
    start_time = time.time()
    for _ in range(50):
        with torch.no_grad():
            output = standard_mlp(input_tensor)
    standard_time = time.time() - start_time
    
    # Results
    print(f"\n📈 Performance Results:")
    print(f"Quantized MLP: {quantized_time:.4f}s (50 iterations)")
    print(f"Standard MLP:  {standard_time:.4f}s (50 iterations)")
    print(f"Speedup: {standard_time/quantized_time:.2f}x")
    
    # Memory usage
    quantized_params = sum(p.numel() for p in quantized_mlp.parameters())
    standard_params = sum(p.numel() for p in standard_mlp.parameters())
    print(f"\n💾 Memory Usage:")
    print(f"Quantized MLP: {quantized_params:,} parameters")
    print(f"Standard MLP:  {standard_params:,} parameters")
    print(f"Memory ratio: {quantized_params/standard_params:.2f}x")
    
    print("✅ Performance comparison demo completed!\n")


def main():
    """Run all demos."""
    print("🚀 CoSet: Hierarchical Nested Lattice Quantization")
    print("=" * 60)
    print("This example demonstrates the core functionality of CoSet.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_basic_quantization()
        demo_quantized_mlp()
        demo_gradient_compression()
        demo_performance_comparison()
        
        print("🎉 All demos completed successfully!")
        print("\n📚 What you've seen:")
        print("  ✅ Basic quantization operations")
        print("  ✅ Quantized MLP training")
        print("  ✅ Gradient compression (up to 128x!)")
        print("  ✅ Performance benchmarking")
        print("\n🔧 Next steps:")
        print("  • Try different lattice configurations")
        print("  • Experiment with different quantization depths")
        print("  • Test on real datasets")
        print("  • Build CUDA extensions for GPU acceleration")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
