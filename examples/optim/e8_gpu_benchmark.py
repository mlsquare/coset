"""
E8 GPU Benchmark: Compare CPU vs GPU performance for E8 lattice quantization.

This script benchmarks the E8 quantization operations on both CPU and GPU
to demonstrate the performance improvements achieved with GPU acceleration.
"""

import time
import torch
import numpy as np
from coset.lattices import E8Lattice
from coset.quant import QuantizationConfig, batch_encode_e8, batch_decode_e8, batch_quantize_e8
from coset.quant.functional import encode, decode, quantize


def benchmark_single_vector(num_iterations=1000):
    """Benchmark single vector operations on CPU."""
    print("\n" + "="*80)
    print("Single Vector Operations (CPU)")
    print("="*80)
    
    # Setup
    lattice = E8Lattice(device=torch.device('cpu'))
    config = QuantizationConfig(
        lattice_type="E8",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        disable_overload_protection=True
    )
    
    # Create test vector
    x = torch.randn(8)
    
    # Encode benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        enc, T = encode(x, lattice, config)
    encode_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    print(f"Encode: {encode_time:.3f} ms/vector")
    
    # Decode benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        enc, T = encode(x, lattice, config)
        _ = decode(enc, lattice, config, T)
    decode_time = (time.perf_counter() - start) / num_iterations * 1000
    print(f"Decode: {decode_time:.3f} ms/vector")
    
    # Full quantization benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = quantize(x, lattice, config)
    quantize_time = (time.perf_counter() - start) / num_iterations * 1000
    print(f"Quantize: {quantize_time:.3f} ms/vector")
    
    return {
        'encode': encode_time,
        'decode': decode_time,
        'quantize': quantize_time
    }


def benchmark_batch_cpu(batch_sizes=[10, 100, 1000, 10000]):
    """Benchmark batch operations on CPU."""
    print("\n" + "="*80)
    print("Batch Operations (CPU)")
    print("="*80)
    
    lattice = E8Lattice(device=torch.device('cpu'))
    config = QuantizationConfig(
        lattice_type="E8",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        disable_overload_protection=True
    )
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        X = torch.randn(batch_size, 8)
        
        # Encode
        start = time.perf_counter()
        encodings, T_values = batch_encode_e8(X, lattice, config)
        encode_time = (time.perf_counter() - start) * 1000  # ms
        encode_per_vec = encode_time / batch_size
        print(f"  Encode: {encode_time:.3f} ms total, {encode_per_vec:.3f} ms/vec")
        
        # Decode
        start = time.perf_counter()
        _ = batch_decode_e8(encodings, T_values, lattice, config)
        decode_time = (time.perf_counter() - start) * 1000
        decode_per_vec = decode_time / batch_size
        print(f"  Decode: {decode_time:.3f} ms total, {decode_per_vec:.3f} ms/vec")
        
        # Quantize
        start = time.perf_counter()
        _ = batch_quantize_e8(X, lattice, config)
        quantize_time = (time.perf_counter() - start) * 1000
        quantize_per_vec = quantize_time / batch_size
        print(f"  Quantize: {quantize_time:.3f} ms total, {quantize_per_vec:.3f} ms/vec")
        
        results[batch_size] = {
            'encode_per_vec': encode_per_vec,
            'decode_per_vec': decode_per_vec,
            'quantize_per_vec': quantize_per_vec
        }
    
    return results


def benchmark_batch_gpu(batch_sizes=[10, 100, 1000, 10000]):
    """Benchmark batch operations on GPU."""
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("GPU not available, skipping GPU benchmarks")
        print("="*80)
        return None
    
    print("\n" + "="*80)
    print("Batch Operations (GPU)")
    print("="*80)
    
    device = torch.device('cuda')
    lattice = E8Lattice(device=device)
    config = QuantizationConfig(
        lattice_type="E8",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        disable_overload_protection=True
    )
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        X = torch.randn(batch_size, 8, device=device)
        
        # Warmup
        _ = batch_quantize_e8(X, lattice, config)
        torch.cuda.synchronize()
        
        # Encode
        start = time.perf_counter()
        encodings, T_values = batch_encode_e8(X, lattice, config, device=device)
        torch.cuda.synchronize()
        encode_time = (time.perf_counter() - start) * 1000
        encode_per_vec = encode_time / batch_size
        print(f"  Encode: {encode_time:.3f} ms total, {encode_per_vec:.3f} ms/vec")
        
        # Decode
        start = time.perf_counter()
        _ = batch_decode_e8(encodings, T_values, lattice, config, device=device)
        torch.cuda.synchronize()
        decode_time = (time.perf_counter() - start) * 1000
        decode_per_vec = decode_time / batch_size
        print(f"  Decode: {decode_time:.3f} ms total, {decode_per_vec:.3f} ms/vec")
        
        # Quantize
        start = time.perf_counter()
        _ = batch_quantize_e8(X, lattice, config, device=device)
        torch.cuda.synchronize()
        quantize_time = (time.perf_counter() - start) * 1000
        quantize_per_vec = quantize_time / batch_size
        print(f"  Quantize: {quantize_time:.3f} ms total, {quantize_per_vec:.3f} ms/vec")
        
        results[batch_size] = {
            'encode_per_vec': encode_per_vec,
            'decode_per_vec': decode_per_vec,
            'quantize_per_vec': quantize_per_vec
        }
    
    return results


def print_speedup_comparison(cpu_results, gpu_results):
    """Print speedup comparison between CPU and GPU."""
    if gpu_results is None:
        return
    
    print("\n" + "="*80)
    print("Speedup: GPU vs CPU")
    print("="*80)
    
    for batch_size in cpu_results.keys():
        if batch_size not in gpu_results:
            continue
        
        print(f"\nBatch Size: {batch_size}")
        cpu = cpu_results[batch_size]
        gpu = gpu_results[batch_size]
        
        encode_speedup = cpu['encode_per_vec'] / gpu['encode_per_vec']
        decode_speedup = cpu['decode_per_vec'] / gpu['decode_per_vec']
        quantize_speedup = cpu['quantize_per_vec'] / gpu['quantize_per_vec']
        
        print(f"  Encode:   {encode_speedup:.2f}x faster")
        print(f"  Decode:   {decode_speedup:.2f}x faster")
        print(f"  Quantize: {quantize_speedup:.2f}x faster")


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("E8 Lattice Quantization Benchmark")
    print("GPU Acceleration Performance Comparison")
    
    # Single vector on CPU
    single_results = benchmark_single_vector()
    
    # Batch operations on CPU
    cpu_batch_results = benchmark_batch_cpu()
    
    # Batch operations on GPU
    gpu_batch_results = benchmark_batch_gpu()
    
    # Compare speedups
    print_speedup_comparison(cpu_batch_results, gpu_batch_results)
    
    print("\n" + "="*80)
    print("Benchmark Complete")
    print("="*80)


if __name__ == "__main__":
    run_benchmarks()
