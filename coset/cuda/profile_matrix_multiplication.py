#!/usr/bin/env python3
"""
Comprehensive Matrix Multiplication Profiling
Using sim.py for generating perfectly quantized matrices
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import gc

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coset.quantizers.sim import LatticeVectorSimulator
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig

# Import CUDA kernels
import torch.utils.cpp_extension

class MatrixMultiplicationProfiler:
    """Comprehensive profiler for matrix multiplication with different implementations"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lattice = E8Lattice()
        self.config = QuantizationConfig(q=3, M=2)
        self.simulator = LatticeVectorSimulator(lattice_type="E8", q=3, M=2, device=device)
        
        # Load CUDA kernels
        self.encoder_module = None
        self.decoder_module = None
        self.vlut_module = None
        self._load_cuda_kernels()
        
        # Performance tracking
        self.results = {}
        
    def _load_cuda_kernels(self):
        """Load CUDA kernels for encoding, decoding, and vLUT"""
        try:
            print("Loading CUDA kernels...")
            
            # Load encoder
            self.encoder_module = torch.utils.cpp_extension.load(
                name="e8_hnlq_encoder",
                sources=["e8_hnlq_encoder_kernel.cu"],
                verbose=False
            )
            
            # Load decoder
            self.decoder_module = torch.utils.cpp_extension.load(
                name="e8_hnlq_decoder",
                sources=["e8_hnlq_decoder_kernel.cu"],
                verbose=False
            )
            
            # Load vLUT
            self.vlut_module = torch.utils.cpp_extension.load(
                name="e8_vlut",
                sources=["e8_vlut_kernel.cu"],
                verbose=False
            )
            
            print("✅ CUDA kernels loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load CUDA kernels: {e}")
            raise
    
    def generate_quantized_matrices(self, m: int, k: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate matrices - sim.py for small, random for large"""
        matrix_size = m * k * n
        print(f"Generating matrices: {m}x{k} @ {k}x{n} (size: {matrix_size})")
        
        if matrix_size <= 65536:  # Use sim.py for small/medium matrices
            print("  Using sim.py for perfect quantization...")
            # Generate A matrix (m x k)
            num_vectors_A = m * k // 8  # Each vector is 8D
            A_vectors = self.simulator.generate_vectors(num_vectors_A)
            A = A_vectors.view(m, k).to(self.device)
            
            # Generate B matrix (k x n)
            num_vectors_B = k * n // 8  # Each vector is 8D
            B_vectors = self.simulator.generate_vectors(num_vectors_B)
            B = B_vectors.view(k, n).to(self.device)
        else:  # Use random matrices for large matrices
            print("  Using random matrices for speed...")
            A = torch.randn(m, k, device=self.device, dtype=torch.float32)
            B = torch.randn(k, n, device=self.device, dtype=torch.float32)
        
        print(f"✅ Generated matrices: A={A.shape}, B={B.shape}")
        return A, B
    
    def encode_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Encode matrix using CUDA encoder"""
        # Reshape to vectors for encoding
        original_shape = matrix.shape
        vectors = matrix.reshape(-1, 8)  # Each vector is 8D
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        
        # Create encoded matrix with proper shape for vLUT: [M, N/8]
        # where M is the number of rows and N/8 is the number of 8D vectors per row
        m, n = original_shape
        n_vectors_per_row = n // 8  # Number of 8D vectors per row
        encoded_matrix = torch.zeros(m, n_vectors_per_row, dtype=torch.int, device=self.device)
        
        # Encode all vectors
        for row in range(m):
            for col in range(n_vectors_per_row):
                vector_idx = row * n_vectors_per_row + col
                vector = vectors[vector_idx:vector_idx+1]  # Keep batch dimension
                encoded = self.encoder_module.cuda_e8_hnlq_encode(
                    vector, self.config.q, self.config.M, T_to_lat, G_inv
                )
                encoded_matrix[row, col] = encoded[0]
        
        return encoded_matrix
    
    def build_vlut_table(self, encoded_matrix: torch.Tensor) -> torch.Tensor:
        """Build vLUT table for encoded matrix"""
        # For now, create a simple lookup table
        # In a real implementation, this would be more sophisticated
        unique_indices = torch.unique(encoded_matrix)
        vlut_table = torch.randn(len(unique_indices), 8, device=self.device)
        return vlut_table
    
    def pytorch_cpu_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """PyTorch CPU matrix multiplication"""
        # Move to CPU
        A_cpu = A.cpu()
        B_cpu = B.cpu()
        
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        gc.collect()
        
        # Time the operation
        start_time = time.time()
        C = torch.matmul(A_cpu, B_cpu)
        end_time = time.time()
        
        # Get memory usage
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            'peak_reserved': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        }
        
        return C, {
            'time': end_time - start_time,
            'memory': memory_info,
            'device': 'cpu'
        }
    
    def pytorch_cuda_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """PyTorch CUDA matrix multiplication"""
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Time the operation
        start_time = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Get memory usage
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return C, {
            'time': end_time - start_time,
            'memory': memory_info,
            'device': 'cuda'
        }
    
    def vlut_onesided_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """vLUT one-sided matrix multiplication (A quantized, B full-precision)"""
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Encode A matrix
        start_time = time.time()
        A_encoded = self.encode_matrix(A)
        encode_time = time.time() - start_time
        
        # Build vLUT table
        start_time = time.time()
        vlut_table = self.build_vlut_table(A_encoded)
        vlut_time = time.time() - start_time
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        # Perform vLUT matrix multiplication
        start_time = time.time()
        C = self.vlut_module.cuda_e8_vlut_onesided_matmul(
            A_encoded, B, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
        )
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        # Get memory usage
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
            'device': 'cuda_vlut_onesided'
        }
    
    def vlut_twosided_matmul(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """vLUT two-sided matrix multiplication (both A and B quantized)"""
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Encode both matrices
        start_time = time.time()
        A_encoded = self.encode_matrix(A)
        B_encoded = self.encode_matrix(B.T).T.contiguous()  # Transpose B, encode, then transpose back
        encode_time = time.time() - start_time
        
        # Build vLUT table
        start_time = time.time()
        vlut_table = self.build_vlut_table(A_encoded)
        vlut_time = time.time() - start_time
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        # Perform vLUT matrix multiplication
        start_time = time.time()
        C = self.vlut_module.cuda_e8_vlut_twosided_matmul(
            A_encoded, B_encoded, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
        )
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        # Get memory usage
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
            'device': 'cuda_vlut_twosided'
        }
    
    def calculate_accuracy_metrics(self, reference: torch.Tensor, result: torch.Tensor) -> Dict:
        """Calculate accuracy metrics between reference and result"""
        # Ensure both tensors are on the same device
        if reference.device != result.device:
            result = result.to(reference.device)
        
        # Calculate errors
        abs_error = torch.norm(reference - result).item()
        rel_error = abs_error / (torch.norm(reference).item() + 1e-8)
        
        return {
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'max_error': torch.max(torch.abs(reference - result)).item(),
            'mean_error': torch.mean(torch.abs(reference - result)).item()
        }
    
    def profile_single_matrix_size(self, m: int, k: int, n: int, num_runs: int):
        """Profile a single matrix size"""
        print(f"\n📊 Matrix Size: {m}x{k} @ {k}x{n}")
        print("-" * 50)
        
        # Generate quantized matrices
        A, B = self.generate_quantized_matrices(m, k, n)
        
        # Run all implementations
        implementations = [
            ('PyTorch CPU', self.pytorch_cpu_matmul),
            ('PyTorch CUDA', self.pytorch_cuda_matmul),
            ('vLUT One-sided', self.vlut_onesided_matmul),
            ('vLUT Two-sided', self.vlut_twosided_matmul)
        ]
        
        results = {}
        
        for name, func in implementations:
            print(f"\n🔬 Testing {name}...")
            
            # Run multiple times for statistical significance
            times = []
            memory_usage = []
            accuracy_metrics = []
            
            for run in range(num_runs):
                try:
                    # Run the implementation
                    C, stats = func(A, B)
                    
                    # Store timing and memory
                    times.append(stats['time'])
                    memory_usage.append(stats['memory']['peak_allocated'])
                    
                    # Calculate accuracy (use first run as reference)
                    if run == 0:
                        reference_result = C
                    else:
                        accuracy = self.calculate_accuracy_metrics(reference_result, C)
                        accuracy_metrics.append(accuracy)
                    
                except Exception as e:
                    print(f"❌ Error in {name} run {run}: {e}")
                    continue
            
            # Calculate statistics
            if times:
                results[name] = {
                    'time_mean': np.mean(times),
                    'time_std': np.std(times),
                    'time_min': np.min(times),
                    'time_max': np.max(times),
                    'memory_mean': np.mean(memory_usage),
                    'memory_std': np.std(memory_usage),
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
        self.results[f"{m}x{k}x{n}"] = results
        
        # Print comparison
        self._print_comparison(results)

    def profile_matrix_sizes(self, matrix_sizes: List[Tuple[int, int, int]], num_runs: int = 10):
        """Profile different matrix sizes"""
        print(f"\n🚀 Starting comprehensive profiling with {num_runs} runs per test")
        print("=" * 80)
        
        for m, k, n in matrix_sizes:
            print(f"\n📊 Matrix Size: {m}x{k} @ {k}x{n}")
            print("-" * 50)
            
            # Generate quantized matrices
            A, B = self.generate_quantized_matrices(m, k, n)
            
            # Run all implementations
            implementations = [
                ('PyTorch CPU', self.pytorch_cpu_matmul),
                ('PyTorch CUDA', self.pytorch_cuda_matmul),
                ('vLUT One-sided', self.vlut_onesided_matmul),
                ('vLUT Two-sided', self.vlut_twosided_matmul)
            ]
            
            results = {}
            
            for name, func in implementations:
                print(f"\n🔬 Testing {name}...")
                
                # Run multiple times for statistical significance
                times = []
                memory_usage = []
                accuracy_metrics = []
                
                for run in range(num_runs):
                    try:
                        # Run the implementation
                        C, stats = func(A, B)
                        
                        # Store timing and memory
                        times.append(stats['time'])
                        memory_usage.append(stats['memory']['peak_allocated'])
                        
                        # Calculate accuracy (use first run as reference)
                        if run == 0:
                            reference_result = C
                        else:
                            accuracy = self.calculate_accuracy_metrics(reference_result, C)
                            accuracy_metrics.append(accuracy)
                        
                    except Exception as e:
                        print(f"❌ Error in {name} run {run}: {e}")
                        continue
                
                # Calculate statistics
                if times:
                    results[name] = {
                        'time_mean': np.mean(times),
                        'time_std': np.std(times),
                        'time_min': np.min(times),
                        'time_max': np.max(times),
                        'memory_mean': np.mean(memory_usage),
                        'memory_std': np.std(memory_usage),
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
            self.results[f"{m}x{k}x{n}"] = results
            
            # Print comparison
            self._print_comparison(results)
    
    def _print_comparison(self, results: Dict):
        """Print comparison of results"""
        print(f"\n📈 Performance Comparison:")
        print("-" * 50)
        
        # Find reference (PyTorch CUDA)
        reference = results.get('PyTorch CUDA')
        if not reference:
            print("❌ No reference implementation available")
            return
        
        ref_time = reference['time_mean']
        
        for name, result in results.items():
            if result is None:
                print(f"{name:20s}: FAILED")
                continue
            
            time_val = result['time_mean']
            speedup = ref_time / time_val if time_val > 0 else 0
            memory_mb = result['memory_mean'] / 1024 / 1024
            
            print(f"{name:20s}: {time_val:.6f}s ({speedup:.2f}x) | {memory_mb:.1f} MB")
            
            if result['accuracy']:
                error = result['accuracy']['absolute_error']
                print(f"{'':20s}  Error: {error:.2e}")
    
    def print_summary(self):
        """Print comprehensive summary of all results"""
        print(f"\n🎯 COMPREHENSIVE PROFILING SUMMARY")
        print("=" * 80)
        
        for matrix_size, results in self.results.items():
            print(f"\n📊 Matrix Size: {matrix_size}")
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
            
            print(f"🏆 Best Performance: {best_name} ({best_time:.6f}s)")
            
            # Print all results
            for name, result in results.items():
                if result:
                    print(f"{name:20s}: {result['time_mean']:.6f}s | {result['memory_mean']/1024/1024:.1f} MB")
                else:
                    print(f"{name:20s}: FAILED")


def main():
    """Main profiling function"""
    print("🚀 E8 HNLQ Matrix Multiplication Profiling")
    print("Using sim.py for perfectly quantized matrices")
    print("=" * 60)
    
    # Initialize profiler
    profiler = MatrixMultiplicationProfiler()
    
    # Define matrix sizes to test
    matrix_sizes = [
        (4, 8, 2),      # Small
        (8, 16, 4),     # Medium
        (16, 32, 8),    # Large
        (64, 128, 32),  # Very Large
        (256, 512, 128), # Huge
        (1024, 1024, 1024), # Massive
    ]
    
    # Run profiling with dynamic number of runs based on matrix size
    for m, k, n in matrix_sizes:
        matrix_size = m * k * n
        if matrix_size <= 1024:  # Small matrices
            num_runs = 5
        elif matrix_size <= 65536:  # Medium matrices
            num_runs = 3
        else:  # Large matrices
            num_runs = 1
        
        print(f"\n🔬 Testing {m}x{k} @ {k}x{n} (size: {matrix_size}, runs: {num_runs})")
        profiler.profile_single_matrix_size(m, k, n, num_runs)
    
    # Print summary
    profiler.print_summary()
    
    print(f"\n✅ Profiling complete!")


if __name__ == "__main__":
    main()

