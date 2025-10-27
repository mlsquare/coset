"""
E8 GPU Usage Example: Simple demonstration of GPU-accelerated E8 quantization.

This example shows how to use the new GPU-accelerated E8 quantization functions
to process batches of vectors efficiently.
"""

import torch
from coset.lattices import E8Lattice
from coset.quant import QuantizationConfig
from coset.quant import batch_quantize_e8, batch_encode_e8, batch_decode_e8


def basic_usage_example():
    """Basic usage of GPU-accelerated batch quantization."""
    print("="*80)
    print("E8 GPU Quantization - Basic Usage Example")
    print("="*80)
    
    # Setup: Create lattice and configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    lattice = E8Lattice(device=device)
    config = QuantizationConfig(
        lattice_type="E8",
        q=4,           # Quantization parameter
        M=2,           # Number of hierarchical levels
        beta=1.0,      # Scaling factor
        alpha=1.0,     # Overload scaling
        disable_overload_protection=True  # Disable for faster operation
    )
    
    # Create a batch of vectors to quantize
    batch_size = 100
    X = torch.randn(batch_size, 8, device=device)
    print(f"\nCreated batch of {batch_size} vectors")
    
    # Method 1: Complete quantization (encode + decode)
    print("\n[Method 1] Complete quantization...")
    X_quantized = batch_quantize_e8(X, lattice, config, device=device)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_quantized.shape}")
    print(f"Quantization error: {torch.mean((X - X_quantized)**2).item():.6f}")
    
    # Method 2: Encode and decode separately
    print("\n[Method 2] Encode and decode separately...")
    encodings, T_values = batch_encode_e8(X, lattice, config, device=device)
    X_decoded = batch_decode_e8(encodings, T_values, lattice, config, device=device)
    
    print(f"Encodings shape: {encodings.shape}")
    print(f"T values shape: {T_values.shape}")
    print(f"Reconstruction error: {torch.mean((X - X_decoded)**2).item():.6f}")
    
    # Method 3: Round-trip test (quantize twice)
    print("\n[Method 3] Round-trip test...")
    X_quantized_once = batch_quantize_e8(X, lattice, config, device=device)
    X_quantized_twice = batch_quantize_e8(X_quantized_once, lattice, config, device=device)
    
    round_trip_error = torch.mean((X_quantized_once - X_quantized_twice)**2).item()
    print(f"Round-trip error: {round_trip_error:.6f}")
    print(f"Quantization is idempotent: {round_trip_error < 1e-5}")
    
    print("\n" + "="*80)


def performance_comparison_example():
    """Compare performance between single and batch operations."""
    print("\n" + "="*80)
    print("E8 GPU Quantization - Performance Comparison")
    print("="*80)
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lattice = E8Lattice(device=device)
    config = QuantizationConfig(
        lattice_type="E8",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        disable_overload_protection=True
    )
    
    # Single vector operations (for reference)
    print("\n[Reference] Single vector operations...")
    x_single = torch.randn(8, device=device)
    
    start = time.perf_counter()
    from coset.quant.functional import quantize
    for _ in range(100):
        _ = quantize(x_single, lattice, config, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    single_time = time.perf_counter() - start
    print(f"100 single vectors: {single_time*1000:.2f} ms")
    
    # Batch operations
    print("\n[GPU] Batch operations...")
    batch_sizes = [10, 100, 1000]
    
    for batch_size in batch_sizes:
        X = torch.randn(batch_size, 8, device=device)
        
        # Warmup
        _ = batch_quantize_e8(X, lattice, config, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time batch operation
        start = time.perf_counter()
        _ = batch_quantize_e8(X, lattice, config, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - start
        
        time_per_vec = batch_time / batch_size * 1000  # ms per vector
        speedup = (single_time / 100) / (batch_time / batch_size)
        
        print(f"  Batch size {batch_size:5d}: {batch_time*1000:6.2f} ms total, "
              f"{time_per_vec:.3f} ms/vec, {speedup:.2f}x faster")
    
    print("\n" + "="*80)


def device_management_example():
    """Demonstrate proper device management."""
    print("\n" + "="*80)
    print("E8 GPU Quantization - Device Management Example")
    print("="*80)
    
    # Create tensors on CPU
    X_cpu = torch.randn(10, 8)
    print("\nCreated tensor on CPU")
    print(f"  Device: {X_cpu.device}")
    
    # Create lattice on CPU
    lattice_cpu = E8Lattice(device=torch.device('cpu'))
    config = QuantizationConfig(
        lattice_type="E8",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        disable_overload_protection=True
    )
    
    # Quantize on CPU
    X_quantized_cpu = batch_quantize_e8(X_cpu, lattice_cpu, config, device=torch.device('cpu'))
    print(f"\nQuantized on CPU")
    print(f"  Output device: {X_quantized_cpu.device}")
    
    # If GPU is available, move to GPU
    if torch.cuda.is_available():
        device_gpu = torch.device('cuda')
        X_gpu = X_cpu.to(device_gpu)
        lattice_gpu = E8Lattice(device=device_gpu)
        
        print(f"\nMoved tensor to GPU")
        print(f"  Device: {X_gpu.device}")
        
        X_quantized_gpu = batch_quantize_e8(X_gpu, lattice_gpu, config, device=device_gpu)
        print(f"  Output device: {X_quantized_gpu.device}")
        
        # Verify results are similar
        error = torch.mean((X_quantized_cpu - X_quantized_gpu.cpu())**2).item()
        print(f"\nCPU vs GPU error: {error:.6f}")
        print(f"Results match: {error < 1e-5}")
    else:
        print("\nGPU not available, skipping GPU example")
    
    print("\n" + "="*80)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("E8 GPU-Accelerated Quantization - Usage Examples")
    print("="*80)
    
    try:
        # Basic usage
        basic_usage_example()
        
        # Performance comparison
        performance_comparison_example()
        
        # Device management
        device_management_example()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
