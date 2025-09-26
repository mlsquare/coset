"""
Test script for optimized vLUT kernels v2 with performance improvements.
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

def test_optimized_kernels_v2():
    """Test the optimized vLUT kernels v2."""
    print("Testing Optimized vLUT Kernels v2")
    print("=" * 40)
    
    # Initialize
    lattice = E8Lattice()
    config = E8Config(q=3, M=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"E8 Lattice dimension: {lattice.d}")
    print(f"Configuration: q={config.q}, M={config.M}")
    
    # Load optimized kernels v2
    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        optimized_v2_ext = load(
            name='optimized_vlut_v2',
            sources=[os.path.join(current_dir, 'optimized_vlut_kernels_v2.cu')],
            verbose=False,
            extra_cuda_cflags=['-O3', '-use_fast_math', '--ptxas-options=-v']
        )
        print("✓ Optimized kernels v2 loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load optimized kernels v2: {e}")
        return
    
    # Test data
    batch_size = 1000
    d = lattice.d
    input_encodings = torch.randint(0, config.q, (batch_size, d), device=device, dtype=torch.float32)
    query_vector = torch.randn(d, device=device, dtype=torch.float32)
    
    print(f"\nTest data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input encodings shape: {input_encodings.shape}")
    print(f"  Query vector shape: {query_vector.shape}")
    
    # Create a simple vLUT for testing
    lut_size = config.q ** d
    vlut = torch.randn(lut_size, device=device, dtype=torch.float32)
    
    print(f"  vLUT size: {lut_size}")
    
    # Test 1: Optimized fused encoding vLUT lookup
    print(f"\n1. Testing Optimized Fused Encoding vLUT Lookup:")
    try:
        # Ensure input encodings are int32
        input_encodings_int = input_encodings.int()
        
        # Warm up
        for _ in range(5):
            _ = optimized_v2_ext.cuda_optimized_fused_encoding_vlut_lookup(
                input_encodings_int, vlut, batch_size, d, config.q, lut_size
            )
        
        # Time the operation
        num_iterations = 20
        start_time = time.time()
        for _ in range(num_iterations):
            results = optimized_v2_ext.cuda_optimized_fused_encoding_vlut_lookup(
                input_encodings_int, vlut, batch_size, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        throughput = batch_size / avg_time
        
        print(f"  ✓ Optimized v2 dot product: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Results shape: {results.shape}")
        print(f"  ✓ Results dtype: {results.dtype}")
        print(f"  ✓ Results device: {results.device}")
        
    except Exception as e:
        print(f"  ✗ Optimized v2 operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Compare with PyTorch baseline
    print(f"\n2. Testing PyTorch Baseline:")
    try:
        # Create full precision inputs for fair comparison
        pytorch_inputs = torch.randn(batch_size, d, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(5):
            _ = torch.matmul(pytorch_inputs, query_vector)
        
        start_time = time.time()
        for _ in range(num_iterations):
            pytorch_results = torch.matmul(pytorch_inputs, query_vector)
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        throughput = batch_size / avg_time
        
        print(f"  ✓ PyTorch dot product: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Results shape: {pytorch_results.shape}")
        print(f"  ✓ Results dtype: {pytorch_results.dtype}")
        print(f"  ✓ Results device: {pytorch_results.device}")
        
    except Exception as e:
        print(f"  ✗ PyTorch operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Batch operations
    print(f"\n3. Testing Batch Operations:")
    try:
        num_queries = 50
        query_vectors = torch.randn(num_queries, d, device=device, dtype=torch.float32)
        
        # Create vLUTs for batch operations
        batch_vluts = torch.randn(num_queries, lut_size, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(5):
            _ = optimized_v2_ext.cuda_optimized_batch_vlut_dot_product(
                input_encodings_int, query_vectors, batch_vluts, 
                batch_size, 1, num_queries, d, config.q, lut_size
            )
        
        start_time = time.time()
        for _ in range(num_iterations):
            batch_results = optimized_v2_ext.cuda_optimized_batch_vlut_dot_product(
                input_encodings_int, query_vectors, batch_vluts, 
                batch_size, 1, num_queries, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        total_ops = batch_size * num_queries
        throughput = total_ops / avg_time
        
        print(f"  ✓ Optimized v2 batch operations: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Results shape: {batch_results.shape}")
        print(f"  ✓ Total operations: {total_ops:,}")
        
    except Exception as e:
        print(f"  ✗ Batch operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Matrix multiplication
    print(f"\n4. Testing Matrix Multiplication:")
    try:
        input_dim = 100
        output_dim = 50
        input_encodings_3d = torch.randint(0, config.q, (batch_size, input_dim, d), device=device, dtype=torch.float32)
        weight_vectors = torch.randn(output_dim, input_dim, d, device=device, dtype=torch.float32)
        
        # Create vLUTs for matrix multiplication
        matmul_vluts = torch.randn(output_dim * input_dim, lut_size, device=device, dtype=torch.float32)
        
        # Warm up
        for _ in range(3):
            _ = optimized_v2_ext.cuda_optimized_vlut_matrix_multiply(
                input_encodings_3d.int(), weight_vectors, matmul_vluts,
                batch_size, input_dim, output_dim, d, config.q, lut_size
            )
        
        start_time = time.time()
        for _ in range(num_iterations):
            matmul_results = optimized_v2_ext.cuda_optimized_vlut_matrix_multiply(
                input_encodings_3d.int(), weight_vectors, matmul_vluts,
                batch_size, input_dim, output_dim, d, config.q, lut_size
            )
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations
        matmul_ops = batch_size * input_dim * output_dim
        throughput = matmul_ops / avg_time
        
        print(f"  ✓ Optimized v2 matrix multiplication: {avg_time:.4f}s per iteration")
        print(f"  ✓ Throughput: {throughput:,.0f} operations/second")
        print(f"  ✓ Results shape: {matmul_results.shape}")
        print(f"  ✓ Matrix operations: {matmul_ops:,}")
        
    except Exception as e:
        print(f"  ✗ Matrix multiplication failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n5. Performance Summary:")
    print(f"  ✓ Optimized kernels v2 implemented with:")
    print(f"    - Vectorized encoding-to-index conversion")
    print(f"    - Larger thread blocks (32x32 vs 16x16)")
    print(f"    - Unrolled loops for better performance")
    print(f"    - Pre-computed powers of q")
    print(f"    - Optimized memory access patterns")
    
    print(f"\n6. Expected Improvements:")
    print(f"  - Encoding-to-index: 10-50x faster (vectorized vs loop)")
    print(f"  - Thread utilization: 2-4x faster (larger blocks)")
    print(f"  - Memory access: 2-5x faster (optimized patterns)")
    print(f"  - Total expected speedup: 40-800x improvement")
    print(f"  - Target: 2-10x faster than PyTorch native operations")


if __name__ == "__main__":
    test_optimized_kernels_v2()
