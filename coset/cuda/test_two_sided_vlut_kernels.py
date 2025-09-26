"""
Test script for optimized two-sided vLUT kernels.

This script tests the basic functionality of the optimized two-sided vLUT kernels
and compares performance against PyTorch native operations.
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import E8 lattice
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from coset.lattices.e8 import E8Lattice

class E8Config:
    def __init__(self, q: int = 3, M: int = 2):
        self.q = q
        self.M = M

def test_two_sided_vlut_kernels():
    """Test the optimized two-sided vLUT kernels."""
    print("Testing Optimized Two-Sided vLUT Kernels")
    print("=" * 50)
    
    # Initialize
    lattice = E8Lattice()
    config = E8Config(q=3, M=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"E8 Lattice dimension: {lattice.d}")
    print(f"Configuration: q={config.q}, M={config.M}")
    
    # Load optimized two-sided kernels
    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        two_sided_vlut_ext = load(
            name='optimized_two_sided_vlut',
            sources=[os.path.join(current_dir, 'optimized_two_sided_vlut_kernels.cu')],
            verbose=False,
            extra_cuda_cflags=['-O3', '-use_fast_math', '--ptxas-options=-v']
        )
        print("✓ Optimized two-sided kernels loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load optimized two-sided kernels: {e}")
        return
    
    # Test data
    batch_size = 1000
    num_queries = 100
    d = lattice.d
    input_encodings = torch.randint(0, config.q, (batch_size, d), device=device, dtype=torch.float32)
    query_encodings = torch.randint(0, config.q, (num_queries, d), device=device, dtype=torch.float32)
    input_vectors = torch.randn(batch_size, d, device=device, dtype=torch.float32)
    query_vectors = torch.randn(num_queries, d, device=device, dtype=torch.float32)
    
    print(f"\nTest data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of queries: {num_queries}")
    print(f"  Input encodings shape: {input_encodings.shape}")
    print(f"  Query encodings shape: {query_encodings.shape}")
    print(f"  Input vectors shape: {input_vectors.shape}")
    print(f"  Query vectors shape: {query_vectors.shape}")
    
    # Test 1: Two-sided vLUT construction
    print(f"\n1. Testing Two-Sided vLUT Construction:")
    try:
        lut_size = config.q ** d
        
        # Warm up
        for _ in range(3):
            _ = two_sided_vlut_ext.cuda_optimized_build_two_sided_vlut(
                input_encodings.int(), query_encodings.int(), input_vectors, query_vectors,
                batch_size, num_queries, d, config.q, lut_size
            )
        
        # Time the operation
        num_iterations = 10
        start_time = time.time()
        for _ in range(num_iterations):
            vlut = two_sided_vlut_ext.cuda_optimized_build_two_sided_vlut(
                input_encodings.int(), query_encodings.int(), input_vectors, query_vectors,
                batch_size, num_queries, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        throughput = (batch_size * num_queries) / avg_time
        
        print(f"  ✓ Two-sided vLUT construction: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ vLUT shape: {vlut.shape}")
        print(f"  ✓ vLUT dtype: {vlut.dtype}")
        print(f"  ✓ vLUT device: {vlut.device}")
        
    except Exception as e:
        print(f"  ✗ Two-sided vLUT construction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Two-sided vLUT MAC operations
    print(f"\n2. Testing Two-Sided vLUT MAC Operations:")
    try:
        # Create a simple vLUT for testing
        vlut = torch.randn(num_queries, lut_size, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(3):
            _ = two_sided_vlut_ext.cuda_optimized_two_sided_vlut_mac(
                input_encodings.int(), query_encodings.int(), vlut,
                batch_size, num_queries, d, config.q, lut_size
            )
        
        start_time = time.time()
        for _ in range(num_iterations):
            results = two_sided_vlut_ext.cuda_optimized_two_sided_vlut_mac(
                input_encodings.int(), query_encodings.int(), vlut,
                batch_size, num_queries, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        throughput = (batch_size * num_queries) / avg_time
        
        print(f"  ✓ Two-sided vLUT MAC: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Results shape: {results.shape}")
        print(f"  ✓ Results dtype: {results.dtype}")
        print(f"  ✓ Results device: {results.device}")
        
    except Exception as e:
        print(f"  ✗ Two-sided vLUT MAC failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Batch two-sided operations
    print(f"\n3. Testing Batch Two-Sided Operations:")
    try:
        input_dim = 50
        input_encodings_3d = torch.randint(0, config.q, (batch_size, input_dim, d), device=device, dtype=torch.float32)
        batch_vluts = torch.randn(num_queries, input_dim, lut_size, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(3):
            _ = two_sided_vlut_ext.cuda_optimized_batch_two_sided_vlut(
                input_encodings_3d.int(), query_encodings.int(), batch_vluts,
                batch_size, input_dim, num_queries, d, config.q, lut_size
            )
        
        start_time = time.time()
        for _ in range(num_iterations):
            batch_results = two_sided_vlut_ext.cuda_optimized_batch_two_sided_vlut(
                input_encodings_3d.int(), query_encodings.int(), batch_vluts,
                batch_size, input_dim, num_queries, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        total_ops = batch_size * num_queries * input_dim
        throughput = total_ops / avg_time
        
        print(f"  ✓ Batch two-sided operations: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Total operations: {total_ops:,}")
        print(f"  ✓ Results shape: {batch_results.shape}")
        
    except Exception as e:
        print(f"  ✗ Batch two-sided operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Two-sided matrix multiplication
    print(f"\n4. Testing Two-Sided Matrix Multiplication:")
    try:
        input_dim = 100
        output_dim = 50
        input_encodings_3d = torch.randint(0, config.q, (batch_size, input_dim, d), device=device, dtype=torch.float32)
        weight_encodings = torch.randint(0, config.q, (output_dim, input_dim, d), device=device, dtype=torch.float32)
        matmul_vluts = torch.randn(output_dim, input_dim, lut_size, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(3):
            _ = two_sided_vlut_ext.cuda_optimized_two_sided_vlut_matrix_multiply(
                input_encodings_3d.int(), weight_encodings.int(), matmul_vluts,
                batch_size, input_dim, output_dim, d, config.q, lut_size
            )
        
        start_time = time.time()
        for _ in range(num_iterations):
            matmul_results = two_sided_vlut_ext.cuda_optimized_two_sided_vlut_matrix_multiply(
                input_encodings_3d.int(), weight_encodings.int(), matmul_vluts,
                batch_size, input_dim, output_dim, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        matmul_ops = batch_size * input_dim * output_dim
        throughput = matmul_ops / avg_time
        
        print(f"  ✓ Two-sided matrix multiplication: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Matrix operations: {matmul_ops:,}")
        print(f"  ✓ Results shape: {matmul_results.shape}")
        
    except Exception as e:
        print(f"  ✗ Two-sided matrix multiplication failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Compare with PyTorch baseline
    print(f"\n5. Testing PyTorch Baseline:")
    try:
        # Create full precision inputs for fair comparison
        pytorch_inputs = torch.randn(batch_size, d, device=device, dtype=torch.float32)
        pytorch_queries = torch.randn(num_queries, d, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(3):
            _ = torch.matmul(pytorch_inputs, pytorch_queries.t())
        
        start_time = time.time()
        for _ in range(num_iterations):
            pytorch_results = torch.matmul(pytorch_inputs, pytorch_queries.t())
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        throughput = (batch_size * num_queries) / avg_time
        
        print(f"  ✓ PyTorch matrix multiplication: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Results shape: {pytorch_results.shape}")
        print(f"  ✓ Results dtype: {pytorch_results.dtype}")
        print(f"  ✓ Results device: {pytorch_results.device}")
        
    except Exception as e:
        print(f"  ✗ PyTorch baseline failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n6. Performance Summary:")
    print(f"  ✓ Optimized two-sided kernels implemented with:")
    print(f"    - Vectorized encoding-to-index conversion for both input and query")
    print(f"    - Larger thread blocks (32x32) for better GPU utilization")
    print(f"    - Pre-computed powers of q for E8 lattice")
    print(f"    - Optimized memory access patterns")
    print(f"    - Fused operations for better performance")
    
    print(f"\n7. Expected Improvements:")
    print(f"  - Dual encoding optimization: Both input and query encodings optimized")
    print(f"  - Shared vLUT caching: Reduce redundant vLUT construction")
    print(f"  - Batch parallelism: Better utilization of GPU resources")
    print(f"  - Memory layout optimization: Optimal data arrangement for both sides")
    print(f"  - Target: 2-5x faster than PyTorch native operations")
    print(f"  - Target: 1.5-2x faster than one-sided vLUT operations")


if __name__ == "__main__":
    test_two_sided_vlut_kernels()
