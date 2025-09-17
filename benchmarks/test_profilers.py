#!/usr/bin/env python3
"""
Test script for profiling scripts

This script tests the profiling scripts with the current implementation to ensure
they work correctly before running full benchmarks.
"""

import sys
from pathlib import Path
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coset.lattices import D4Lattice
from coset.quant.params import QuantizationConfig
from coset.quant.functional import encode, decode, quantize


def test_basic_functionality():
    """Test basic functionality of the quantization functions."""
    print("Testing basic functionality...")
    
    # Create lattice and config
    lattice = D4Lattice()
    config = QuantizationConfig(
        lattice_type="D4",
        q=4,
        M=2,
        beta=1.0,
        disable_scaling=True,
        disable_overload_protection=True
    )
    
    # Test with a single vector
    x = torch.randn(4) * 0.5
    print(f"Input vector: {x}")
    
    # Test encoding
    b, t = encode(x, lattice, config)
    print(f"Encoded: shape={b.shape}, T={t}")
    
    # Test decoding
    x_hat = decode(b, lattice, config, t)
    print(f"Decoded: {x_hat}")
    
    # Test quantization (encode + decode)
    x_quantized = quantize(x, lattice, config)
    print(f"Quantized: {x_quantized}")
    
    # Test error
    error = torch.norm(x - x_hat)
    print(f"Reconstruction error: {error:.6f}")
    
    print("✓ Basic functionality test passed")
    return True


def test_batch_processing():
    """Test batch processing functionality."""
    print("\nTesting batch processing...")
    
    # Create lattice and config
    lattice = D4Lattice()
    config = QuantizationConfig(
        lattice_type="D4",
        q=4,
        M=2,
        beta=1.0,
        disable_scaling=True,
        disable_overload_protection=True
    )
    
    # Test with batch of vectors
    batch_size = 5
    x = torch.randn(batch_size, 4) * 0.5
    print(f"Input batch: shape={x.shape}")
    
    # Process each vector in the batch
    encoded_batch = []
    decoded_batch = []
    
    for i in range(batch_size):
        b, t = encode(x[i], lattice, config)
        x_hat = decode(b, lattice, config, t)
        encoded_batch.append(b)
        decoded_batch.append(x_hat)
    
    encoded_batch = torch.stack(encoded_batch)
    decoded_batch = torch.stack(decoded_batch)
    
    print(f"Encoded batch: shape={encoded_batch.shape}")
    print(f"Decoded batch: shape={decoded_batch.shape}")
    
    # Test error
    error = torch.norm(x - decoded_batch, dim=1)
    print(f"Reconstruction errors: {error}")
    print(f"Average error: {error.mean():.6f}")
    
    print("✓ Batch processing test passed")
    return True


def test_cuda_availability():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Test moving tensors to GPU
        x = torch.randn(4)
        x_gpu = x.cuda()
        print(f"Tensor on GPU: {x_gpu.device}")
        
        # Test basic operations on GPU
        lattice = D4Lattice()
        config = QuantizationConfig(
            lattice_type="D4",
            q=4,
            M=2,
            beta=1.0,
            disable_scaling=True,
            disable_overload_protection=True
        )
        
        try:
            b, t = encode(x_gpu, lattice, config)
            x_hat = decode(b, lattice, config, t)
            print(f"GPU encoding/decoding successful: {x_hat.device}")
        except Exception as e:
            print(f"GPU encoding/decoding failed: {e}")
            return False
    
    print("✓ CUDA availability test passed")
    return True


def test_profiler_imports():
    """Test that profiler scripts can be imported."""
    print("\nTesting profiler imports...")
    
    try:
        # Test importing the profiler classes
        from profile_encoding import EncodingProfiler
        from profile_decoding import DecodingProfiler
        from profile_combined import CombinedProfiler
        
        print("✓ All profiler classes imported successfully")
        
        # Test creating profiler instances
        encoding_profiler = EncodingProfiler("D4", 4, 2)
        decoding_profiler = DecodingProfiler("D4", 4, 2)
        combined_profiler = CombinedProfiler("D4", 4, 2)
        
        print("✓ All profiler instances created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Profiler import failed: {e}")
        return False


def test_small_benchmark():
    """Run a small benchmark to test the profilers."""
    print("\nRunning small benchmark test...")
    
    try:
        from profile_encoding import EncodingProfiler
        
        profiler = EncodingProfiler("D4", 4, 2)
        
        # Test with small batch size
        result = profiler.profile_single_batch(batch_size=10, num_warmup=2, num_iterations=3)
        
        print(f"Benchmark result: {result['batch_size']} vectors")
        print(f"Baseline time: {result['baseline']['mean_time']*1000:.2f}ms")
        print(f"CUDA time: {result['cuda_optimized']['mean_time']*1000:.2f}ms")
        print(f"Speedup: {result['speedup']:.2f}x")
        
        print("✓ Small benchmark test passed")
        return True
        
    except Exception as e:
        print(f"✗ Small benchmark test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("PROFILER TEST SUITE")
    print("="*50)
    
    tests = [
        test_basic_functionality,
        test_batch_processing,
        test_cuda_availability,
        test_profiler_imports,
        test_small_benchmark
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("✓ All tests passed! Profilers are ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before running full benchmarks.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
