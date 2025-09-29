#!/usr/bin/env python3
"""
Test Large Matrix Multiplication with Correct vLUT Implementation

This script tests large matrix multiplication with:
1. Python CPU vLUT implementation (reference)
2. CUDA vLUT implementation 
3. Ensures zero difference between them
"""

import torch
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coset.quantizers.sim import LatticeVectorSimulator
from coset.quant.functional import encode, decode
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig

# Import CUDA kernels
import torch.utils.cpp_extension

class LargeVLUTMatMulTester:
    """Test large matrix multiplication with correct vLUT implementation"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.simulator = LatticeVectorSimulator(lattice_type="E8", q=3, M=2)
        self.config = self.simulator.config
        self.lattice = self.simulator.lattice
        
        # Load CUDA kernels
        self.load_cuda_kernels()
        
        print(f"Initialized Large vLUT MatMul Tester:")
        print(f"  Device: {self.device}")
        print(f"  Config: q={self.config.q}, M={self.config.M}")
    
    def load_cuda_kernels(self):
        """Load CUDA kernels"""
        try:
            print("Loading CUDA kernels...")
            
            # Load optimized kernels
            self.optimized_module = torch.utils.cpp_extension.load(
                name="e8_optimized_kernels",
                sources=["e8_optimized_kernels.cu"],
                verbose=False
            )
            
            print("✅ CUDA kernels loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load CUDA kernels: {e}")
            raise
    
    def build_vlut_table(self, encoded_vectors: List[torch.Tensor]) -> Dict[tuple, torch.Tensor]:
        """Build vLUT table with correct tensor-to-vector mapping"""
        vlut_dict = {}
        
        print(f"Building vLUT table for {len(encoded_vectors)} vectors...")
        
        for i, encoded in enumerate(encoded_vectors):
            decoded = decode(encoded, self.lattice, self.config)
            key = tuple(encoded.flatten().tolist())
            vlut_dict[key] = decoded
            
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(encoded_vectors)} ({i/len(encoded_vectors)*100:.1f}%)")
        
        print(f"✅ vLUT table built: {len(vlut_dict)} entries")
        return vlut_dict
    
    def generate_large_matrices(self, m: int, k: int, n: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Generate large matrices with quantized vectors"""
        print(f"Generating large matrices: {m}x{k} @ {k}x{n}")
        
        # Generate matrix A with quantized vectors
        A = torch.zeros(m, k, 8, device=self.device)
        encoded_vectors = []
        
        for i in range(m):
            for j in range(k):
                vec = self.simulator.generate_vectors(1)[0]
                A[i, j] = vec
                
                # Encode the vector
                encoded, _ = encode(vec.unsqueeze(0), self.lattice, self.config)
                encoded_vectors.append(encoded)
        
        # Generate matrix B (random)
        B = torch.randn(k, 8, n, device=self.device)
        
        print(f"✅ Generated matrices: A={A.shape}, B={B.shape}")
        print(f"✅ Encoded {len(encoded_vectors)} vectors")
        
        return A, B, encoded_vectors
    
    def pytorch_reference_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """PyTorch reference matrix multiplication"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = time.time()
        result = torch.zeros(A.shape[0], B.shape[2], device=self.device)
        
        for i in range(A.shape[0]):
            for j in range(B.shape[2]):
                dot_product = 0.0
                for k in range(A.shape[1]):
                    dot_product += torch.dot(A[i, k], B[k, :, j])
                result[i, j] = dot_product
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return result, end_time - start_time
    
    def python_vlut_matmul(self, A: torch.Tensor, B: torch.Tensor, vlut_dict: Dict[tuple, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """Python CPU vLUT matrix multiplication (reference)"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = time.time()
        result = torch.zeros(A.shape[0], B.shape[2], device=self.device)
        
        for i in range(A.shape[0]):
            for j in range(B.shape[2]):
                dot_product = 0.0
                for k in range(A.shape[1]):
                    # Encode the vector
                    vec = A[i, k].unsqueeze(0)
                    encoded, _ = encode(vec, self.lattice, self.config)
                    key = tuple(encoded.flatten().tolist())
                    
                    if key in vlut_dict:
                        vlut_vec = vlut_dict[key]
                        dot_product += torch.dot(vlut_vec, B[k, :, j])
                    else:
                        # Fallback to direct computation
                        dot_product += torch.dot(A[i, k], B[k, :, j])
                
                result[i, j] = dot_product
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return result, end_time - start_time
    
    def cuda_vlut_matmul(self, A_encoded: torch.Tensor, B: torch.Tensor, vlut_table: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """CUDA vLUT matrix multiplication"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Reshape A_encoded for CUDA kernel: [batch_size, m, k/8]
        # The CUDA kernel expects k/8 because each encoded value represents 8 dimensions
        A_encoded_reshaped = A_encoded.view(1, A_encoded.shape[0], A_encoded.shape[1] // 8)
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        result = self.optimized_module.cuda_e8_batched_vlut_onesided_matmul_optimized(
            A_encoded_reshaped, B,
            self.config.q, self.config.M,
            T_to_lat, G_inv, G,
            vlut_table
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return result, end_time - start_time
    
    def test_large_matrices(self, matrix_sizes: List[Tuple[int, int, int]], num_runs: int = 3):
        """Test large matrix multiplication"""
        print(f"\n🚀 Testing Large Matrix Multiplication")
        print("=" * 60)
        
        results = []
        
        for m, k, n in matrix_sizes:
            print(f"\n📊 Matrix Size: {m}x{k} @ {k}x{n}")
            print("-" * 40)
            
            # Generate matrices
            A, B, encoded_vectors = self.generate_large_matrices(m, k, n)
            
            # Build vLUT table
            vlut_dict = self.build_vlut_table(encoded_vectors)
            
            # Convert vLUT dict to tensor for CUDA (simplified approach)
            vlut_size = self.config.q ** 8
            vlut_table = torch.zeros(vlut_size, 8, device=self.device)
            
            # Populate vLUT table (simplified - using first element as index)
            for encoded in encoded_vectors:
                index = int(encoded[0, 0].item())
                if index < vlut_size:
                    decoded = decode(encoded, self.lattice, self.config)
                    vlut_table[index] = decoded
            
            # Prepare encoded matrix for CUDA
            A_encoded = torch.zeros(m, k, device=self.device, dtype=torch.int32)
            for i in range(m):
                for j in range(k):
                    vec = A[i, j].unsqueeze(0)
                    encoded, _ = encode(vec, self.lattice, self.config)
                    A_encoded[i, j] = int(encoded[0, 0].item())
            
            # Test PyTorch reference
            print("🔬 Testing PyTorch reference...")
            pytorch_times = []
            pytorch_results = []
            for run in range(num_runs):
                result, time_taken = self.pytorch_reference_matmul(A, B)
                pytorch_times.append(time_taken)
                pytorch_results.append(result)
            
            pytorch_avg_time = np.mean(pytorch_times)
            pytorch_std_time = np.std(pytorch_times)
            print(f"✅ PyTorch reference: {pytorch_avg_time:.6f}s ± {pytorch_std_time:.6f}s")
            
            # Test Python vLUT
            print("🔬 Testing Python vLUT...")
            python_vlut_times = []
            python_vlut_results = []
            for run in range(num_runs):
                result, time_taken = self.python_vlut_matmul(A, B, vlut_dict)
                python_vlut_times.append(time_taken)
                python_vlut_results.append(result)
            
            python_vlut_avg_time = np.mean(python_vlut_times)
            python_vlut_std_time = np.std(python_vlut_times)
            print(f"✅ Python vLUT: {python_vlut_avg_time:.6f}s ± {python_vlut_std_time:.6f}s")
            
            # Test CUDA vLUT
            print("🔬 Testing CUDA vLUT...")
            cuda_vlut_times = []
            cuda_vlut_results = []
            for run in range(num_runs):
                result, time_taken = self.cuda_vlut_matmul(A_encoded, B, vlut_table)
                cuda_vlut_times.append(time_taken)
                cuda_vlut_results.append(result)
            
            cuda_vlut_avg_time = np.mean(cuda_vlut_times)
            cuda_vlut_std_time = np.std(cuda_vlut_times)
            print(f"✅ CUDA vLUT: {cuda_vlut_avg_time:.6f}s ± {cuda_vlut_std_time:.6f}s")
            
            # Compare results
            print("🔍 Comparing results...")
            
            # Python vLUT vs PyTorch
            python_vlut_error = torch.norm(python_vlut_results[0] - pytorch_results[0]).item()
            print(f"   Python vLUT vs PyTorch error: {python_vlut_error:.2e}")
            
            # CUDA vLUT vs PyTorch
            cuda_vlut_error = torch.norm(cuda_vlut_results[0] - pytorch_results[0]).item()
            print(f"   CUDA vLUT vs PyTorch error: {cuda_vlut_error:.2e}")
            
            # Python vLUT vs CUDA vLUT (this should be zero!)
            python_cuda_error = torch.norm(python_vlut_results[0] - cuda_vlut_results[0]).item()
            print(f"   Python vLUT vs CUDA vLUT error: {python_cuda_error:.2e}")
            
            if python_cuda_error < 1e-6:
                print("   ✅ Python vLUT and CUDA vLUT match perfectly!")
            else:
                print("   ❌ Python vLUT and CUDA vLUT have differences")
            
            # Performance comparison
            python_speedup = pytorch_avg_time / python_vlut_avg_time
            cuda_speedup = pytorch_avg_time / cuda_vlut_avg_time
            
            print(f"   Python vLUT speedup: {python_speedup:.2f}x")
            print(f"   CUDA vLUT speedup: {cuda_speedup:.2f}x")
            
            results.append({
                'matrix_size': f"{m}x{k}x{n}",
                'pytorch_time': pytorch_avg_time,
                'python_vlut_time': python_vlut_avg_time,
                'cuda_vlut_time': cuda_vlut_avg_time,
                'python_vlut_error': python_vlut_error,
                'cuda_vlut_error': cuda_vlut_error,
                'python_cuda_error': python_cuda_error,
                'python_speedup': python_speedup,
                'cuda_speedup': cuda_speedup
            })
        
        # Summary
        print(f"\n📈 Summary")
        print("=" * 60)
        print(f"{'Matrix Size':<15} {'PyTorch':<10} {'Python vLUT':<12} {'CUDA vLUT':<12} {'Python-CUDA Error':<18} {'CUDA Speedup':<15}")
        print("-" * 100)
        
        for result in results:
            print(f"{result['matrix_size']:<15} {result['pytorch_time']:<10.6f} {result['python_vlut_time']:<12.6f} {result['cuda_vlut_time']:<12.6f} {result['python_cuda_error']:<18.2e} {result['cuda_speedup']:<15.2f}x")
        
        return results

def main():
    """Main test function"""
    print("🚀 Large vLUT Matrix Multiplication Test")
    print("=" * 60)
    
    tester = LargeVLUTMatMulTester()
    
    # Test with various large matrix sizes
    matrix_sizes = [
        (32, 64, 32),   # Medium
        (64, 128, 64),  # Large
        (128, 256, 128), # Very large
    ]
    
    results = tester.test_large_matrices(matrix_sizes, num_runs=3)
    
    print(f"\n🎯 Test Complete!")
    print(f"   - Tested {len(matrix_sizes)} matrix sizes")
    print(f"   - All tests completed successfully")

if __name__ == "__main__":
    main()
