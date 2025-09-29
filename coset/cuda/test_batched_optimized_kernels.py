#!/usr/bin/env python3
"""
Test Batched Optimized E8 HNLQ CUDA Kernels
Testing batched matrix multiplication with fair timing (with/without encoding)
"""

import sys
import os
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coset.quantizers.sim import LatticeVectorSimulator
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig
from coset.quant.functional import decode

# Import CUDA kernels
import torch.utils.cpp_extension

class BatchedOptimizedKernelTester:
    """Comprehensive tester for batched optimized CUDA kernels"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lattice = E8Lattice()
        self.config = QuantizationConfig(q=3, M=2)
        self.simulator = LatticeVectorSimulator(lattice_type="E8", q=3, M=2, device=device)
        
        # Load optimized CUDA kernels
        self.optimized_module = None
        self.encoder_module = None
        self._load_cuda_kernels()
        
        # Performance tracking
        self.results = {}
        
    def _load_cuda_kernels(self):
        """Load CUDA kernels for optimized and encoding operations"""
        try:
            print("Loading CUDA kernels...")
            
            # Load optimized kernels
            self.optimized_module = torch.utils.cpp_extension.load(
                name="e8_optimized_kernels",
                sources=["e8_optimized_kernels.cu"],
                verbose=False
            )
            
            # Load encoder for encoding operations
            self.encoder_module = torch.utils.cpp_extension.load(
                name="e8_hnlq_encoder",
                sources=["e8_hnlq_encoder_kernel.cu"],
                verbose=False
            )
            
            print("✅ CUDA kernels loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load CUDA kernels: {e}")
            raise
    
    def generate_batched_matrices(self, batch_size: int, m: int, k: int, n: int, use_sim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batched test matrices"""
        matrix_size = m * k * n
        print(f"Generating batched matrices: {batch_size}x{m}x{k} @ {batch_size}x{k}x{n} (size: {matrix_size})")
        
        # Clear cache before generating matrices
        torch.cuda.empty_cache()
        
        if use_sim and matrix_size <= 65536:  # Use sim.py for small/medium matrices
            print("  Using sim.py for perfect quantization...")
            # Generate A matrices (batch_size x m x k)
            A_batch = torch.zeros(batch_size, m, k, device=self.device, dtype=torch.float32)
            for b in range(batch_size):
                num_vectors_A = m * k // 8
                A_vectors = self.simulator.generate_vectors(num_vectors_A)
                A_batch[b] = A_vectors.view(m, k)
            
            # Generate B matrices (batch_size x k x n)
            B_batch = torch.zeros(batch_size, k, n, device=self.device, dtype=torch.float32)
            for b in range(batch_size):
                num_vectors_B = k * n // 8
                B_vectors = self.simulator.generate_vectors(num_vectors_B)
                B_batch[b] = B_vectors.view(k, n)
        else:  # Use random matrices for large matrices
            print("  Using random matrices for speed...")
            A_batch = torch.randn(batch_size, m, k, device=self.device, dtype=torch.float32)
            B_batch = torch.randn(batch_size, k, n, device=self.device, dtype=torch.float32)
        
        print(f"✅ Generated batched matrices: A={A_batch.shape}, B={B_batch.shape}")
        return A_batch, B_batch
    
    def encode_batched_matrix(self, matrix_batch: torch.Tensor) -> torch.Tensor:
        """Encode batched matrix using CUDA encoder"""
        batch_size, m, k = matrix_batch.shape
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        
        # Create encoded matrix with proper shape for vLUT: [batch_size, m, k/8]
        k_encoded = k // 8  # Number of 8D vectors per row
        encoded_batch = torch.zeros(batch_size, m, k_encoded, dtype=torch.int, device=self.device)
        
        # Encode all vectors in all batches
        for b in range(batch_size):
            for row in range(m):
                for col in range(k_encoded):
                    vector_idx = row * k_encoded + col
                    vector = matrix_batch[b, row, col*8:(col+1)*8].unsqueeze(0)  # Keep batch dimension
                    encoded = self.encoder_module.cuda_e8_hnlq_encode(
                        vector, self.config.q, self.config.M, T_to_lat, G_inv
                    )
                    encoded_batch[b, row, col] = encoded[0]
        
        return encoded_batch
    
    def build_vlut_table(self, encoded_batch: torch.Tensor) -> torch.Tensor:
        """Build vLUT table with actual decoded vectors"""
        # vLUT table size is q^d (where d=8 for E8 lattice)
        # For E8 lattice with q=3, d=8, the table size is q^d = 3^8 = 6,561
        # This table is reused across all M levels
        vlut_table_size = self.config.q ** 8  # q^d where d=8 for E8
        vlut_table = torch.zeros(vlut_table_size, 8, device=self.device, dtype=torch.float32)
        
        # Fill vLUT table with actual decoded vectors
        for i in range(vlut_table_size):
            decoded = decode(torch.tensor([[i]], device=self.device, dtype=torch.int32), self.config)
            vlut_table[i] = decoded[0]
        
        print(f"   vLUT table size: {vlut_table_size} (based on q={self.config.q}, d=8, reused across M={self.config.M} levels)")
        return vlut_table
    
    def pytorch_batched_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """PyTorch batched matrix multiplication"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        C = torch.bmm(A, B)  # Batched matrix multiplication
        torch.cuda.synchronize()
        end_time = time.time()
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return C, {
            'time': end_time - start_time,
            'memory': memory_info,
            'device': 'pytorch_cuda_batched'
        }
    
    def optimized_batched_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Optimized CUDA batched matrix multiplication"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        vlut_table = torch.randn(1000, 8, device=self.device, dtype=torch.float32)  # Dummy vLUT table
        
        start_time = time.time()
        result = self.optimized_module.cuda_e8_batched_matrix_multiply_optimized(
            A, B, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
        )
        torch.cuda.synchronize()
        end_time = time.time()
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return result, {
            'time': end_time - start_time,
            'memory': memory_info,
            'device': 'optimized_cuda_batched'
        }
    
    def vlut_onesided_batched_matmul_end_to_end(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """vLUT one-sided batched matrix multiplication (END-TO-END: including encoding)"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Encode A matrix (INCLUDED in timing)
        start_time = time.time()
        A_encoded = self.encode_batched_matrix(A)
        encode_time = time.time() - start_time
        
        # Build vLUT table
        start_time = time.time()
        vlut_table = self.build_vlut_table(A_encoded)
        vlut_time = time.time() - start_time
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        # Perform vLUT batched matrix multiplication
        start_time = time.time()
        C = self.optimized_module.cuda_e8_batched_vlut_onesided_matmul_optimized(
            A_encoded, B, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
        )
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return C, {
            'time': encode_time + vlut_time + matmul_time,
            'encode_time': encode_time,
            'vlut_time': vlut_time,
            'matmul_time': matmul_time,
            'memory': memory_info,
            'device': 'cuda_vlut_onesided_batched_end_to_end'
        }
    
    def vlut_onesided_batched_matmul_fair(self, A_encoded: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """vLUT one-sided batched matrix multiplication (FAIR: excluding encoding)"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Build vLUT table (minimal overhead)
        start_time = time.time()
        vlut_table = self.build_vlut_table(A_encoded)
        vlut_time = time.time() - start_time
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        # Perform vLUT batched matrix multiplication (FAIR TIMING)
        start_time = time.time()
        C = self.optimized_module.cuda_e8_batched_vlut_onesided_matmul_optimized(
            A_encoded, B, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
        )
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return C, {
            'time': vlut_time + matmul_time,  # Exclude encoding time
            'vlut_time': vlut_time,
            'matmul_time': matmul_time,
            'memory': memory_info,
            'device': 'cuda_vlut_onesided_batched_fair'
        }
    
    def vlut_twosided_batched_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """vLUT two-sided batched matrix multiplication (both A and B quantized)"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Encode both matrices
        start_time = time.time()
        A_encoded = self.encode_batched_matrix(A)
        B_encoded = self.encode_batched_matrix(B)
        encode_time = time.time() - start_time
        
        # Build vLUT table
        start_time = time.time()
        vlut_table = self.build_vlut_table(A_encoded)
        vlut_time = time.time() - start_time
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        # Perform vLUT batched matrix multiplication
        start_time = time.time()
        C = self.optimized_module.cuda_e8_batched_vlut_twosided_matmul_optimized(
            A_encoded, B_encoded, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
        )
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return C, {
            'time': encode_time + vlut_time + matmul_time,
            'encode_time': encode_time,
            'vlut_time': vlut_time,
            'matmul_time': matmul_time,
            'memory': memory_info,
            'device': 'cuda_vlut_twosided_batched'
        }
    
    def calculate_accuracy_metrics(self, reference: torch.Tensor, result: torch.Tensor) -> Dict:
        """Calculate accuracy metrics between reference and result"""
        if reference.device != result.device:
            result = result.to(reference.device)
        
        abs_error = torch.norm(reference - result).item()
        rel_error = abs_error / (torch.norm(reference).item() + 1e-8)
        
        return {
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'max_error': torch.max(torch.abs(reference - result)).item(),
            'mean_error': torch.mean(torch.abs(reference - result)).item()
        }
    
    def test_batched_matrix_multiplication(self, batch_sizes: List[int], matrix_sizes: List[Tuple[int, int, int]], num_runs: int = 3):
        """Test batched matrix multiplication across different batch and matrix sizes"""
        print(f"\n🔬 Testing Batched Matrix Multiplication")
        print("=" * 80)
        
        for batch_size in batch_sizes:
            for m, k, n in matrix_sizes:
                print(f"\n📊 Batch Size: {batch_size}, Matrix Size: {m}x{k} @ {k}x{n}")
                print("-" * 60)
                
                # Generate batched matrices
                A, B = self.generate_batched_matrices(batch_size, m, k, n)
                
                # Pre-encode A for fair comparison
                A_encoded = self.encode_batched_matrix(A)
                
                # Test implementations
                implementations = [
                    ('PyTorch CUDA Batched', self.pytorch_batched_matmul),
                    ('Optimized CUDA Batched', self.optimized_batched_matmul),
                    ('vLUT One-sided (End-to-End)', self.vlut_onesided_batched_matmul_end_to_end),
                    ('vLUT One-sided (Fair)', lambda A, B: self.vlut_onesided_batched_matmul_fair(A_encoded, B)),
                    ('vLUT Two-sided', self.vlut_twosided_batched_matmul)
                ]
                
                results = {}
                
                for name, func in implementations:
                    print(f"\n🔬 Testing {name}...")
                    
                    times = []
                    memory_usage = []
                    accuracy_metrics = []
                    
                    for run in range(num_runs):
                        try:
                            if 'Fair' in name:
                                result, stats = func(A_encoded, B)
                            else:
                                result, stats = func(A, B)
                            
                            times.append(stats['time'])
                            memory_usage.append(stats['memory']['peak_allocated'])
                            
                            if run == 0:
                                reference_result = result
                            else:
                                accuracy = self.calculate_accuracy_metrics(reference_result, result)
                                accuracy_metrics.append(accuracy)
                            
                        except Exception as e:
                            print(f"❌ Error in {name} run {run}: {e}")
                            continue
                    
                    if times:
                        results[name] = {
                            'time_mean': np.mean(times),
                            'time_std': np.std(times),
                            'memory_mean': np.mean(memory_usage),
                            'accuracy': accuracy_metrics[0] if accuracy_metrics else None,
                            'success_rate': len(times) / num_runs
                        }
                        
                        print(f"✅ {name}: {results[name]['time_mean']:.6f}s ± {results[name]['time_std']:.6f}s")
                        print(f"   Memory: {results[name]['memory_mean']/1024/1024:.1f} MB")
                        if results[name]['accuracy']:
                            print(f"   Error: {results[name]['accuracy']['absolute_error']:.2e}")
                    else:
                        print(f"❌ {name}: All runs failed")
                        results[name] = None
                
                # Store results
                self.results[f"batched_{batch_size}_{m}x{k}x{n}"] = results
                
                # Print comparison
                self._print_comparison(results, f"Batched {batch_size}x{m}x{k}x{n}")
    
    def _print_comparison(self, results: Dict, test_name: str):
        """Print comparison of results"""
        print(f"\n📈 Performance Comparison - {test_name}:")
        print("-" * 70)
        
        # Find reference (PyTorch CUDA Batched)
        reference = results.get('PyTorch CUDA Batched')
        if not reference:
            print("❌ No reference implementation available")
            return
        
        ref_time = reference['time_mean']
        
        for name, result in results.items():
            if result is None:
                print(f"{name:35s}: FAILED")
                continue
            
            time_val = result['time_mean']
            speedup = ref_time / time_val if time_val > 0 else 0
            memory_mb = result['memory_mean'] / 1024 / 1024
            
            print(f"{name:35s}: {time_val:.6f}s ({speedup:.2f}x) | {memory_mb:.1f} MB")
            
            if result['accuracy']:
                error = result['accuracy']['absolute_error']
                print(f"{'':35s}  Error: {error:.2e}")
    
    def print_summary(self):
        """Print comprehensive summary of all results"""
        print(f"\n🎯 BATCHED OPTIMIZED KERNELS PERFORMANCE SUMMARY")
        print("=" * 80)
        
        for test_name, results in self.results.items():
            print(f"\n📊 {test_name}")
            print("-" * 40)
            
            if not results:
                print("No results available")
                continue
            
            # Find best performing implementation
            best_time = float('inf')
            best_name = None
            
            for name, result in results.items():
                if result and result['time_mean'] < best_time:
                    best_time = result['time_mean']
                    best_name = name
            
            if best_name:
                print(f"🏆 Best Performance: {best_name} ({best_time:.6f}s)")
            
            # Print all results
            for name, result in results.items():
                if result:
                    print(f"{name:35s}: {result['time_mean']:.6f}s | {result['memory_mean']/1024/1024:.1f} MB")
                else:
                    print(f"{name:35s}: FAILED")


def main():
    """Main testing function"""
    print("🚀 Batched Optimized E8 HNLQ CUDA Kernels Testing")
    print("Testing batched matrix multiplication with fair timing")
    print("=" * 80)
    
    # Initialize tester
    tester = BatchedOptimizedKernelTester()
    
    # Test batched matrix multiplication
    batch_sizes = [2, 4, 8, 16]
    matrix_sizes = [
        (8, 16, 8),      # Small
        (16, 32, 16),    # Medium
        (32, 64, 32),    # Large
        (64, 128, 64),   # Very Large
    ]
    
    tester.test_batched_matrix_multiplication(batch_sizes, matrix_sizes, num_runs=3)
    
    # Print summary
    tester.print_summary()
    
    print(f"\n✅ Batched optimized kernels testing complete!")


if __name__ == "__main__":
    main()
