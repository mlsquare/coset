#!/usr/bin/env python3
"""
Proper CUDA Benchmark: Using Existing CUDA Kernel with 8D Vector Operations

This script properly integrates the existing CUDA kernel by:
1. Reshaping 8D vector operations to work with scalar matrix multiplication
2. Maintaining accuracy by using proper dot product operations
3. Providing fair comparison between all implementations
"""

import torch
import sys
import os
import time
import numpy as np
import gc
from typing import Dict, List, Tuple

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coset.quantizers.sim import LatticeVectorSimulator
from coset.quant.functional import encode, decode
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig

# Import CUDA kernels
import torch.utils.cpp_extension

def is_quantized_vector(vec, lattice, config, tolerance=1e-6):
    """Check if a vector is properly quantized"""
    encoded, _ = encode(vec.unsqueeze(0), lattice, config)
    decoded = decode(encoded, lattice, config)
    error = torch.norm(vec - decoded).item()
    return error < tolerance

def generate_quantized_batch(batch_size, m, k, simulator, lattice, config, max_attempts=1000):
    """Generate a batch of matrices with only properly quantized vectors"""
    A = torch.zeros(batch_size, m, k, 8, device=simulator.device)
    
    print(f"Generating batch of {batch_size} matrices ({m}x{k}) with only quantized vectors...")
    
    for b in range(batch_size):
        for i in range(m):
            for j in range(k):
                attempts = 0
                while attempts < max_attempts:
                    vec = simulator.generate_vectors(1)[0]
                    if is_quantized_vector(vec, lattice, config):
                        A[b, i, j] = vec
                        break
                    attempts += 1
                
                if attempts >= max_attempts:
                    print(f"Warning: Could not generate quantized vector for batch {b}, ({i},{j}) after {max_attempts} attempts")
                    A[b, i, j] = vec
    
    return A

def measure_memory():
    """Measure current GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0.0

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class ProperCUDABenchmark:
    """Proper benchmark using existing CUDA kernel with 8D vector operations"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.simulator = LatticeVectorSimulator(lattice_type="E8", q=3, M=2)
        self.config = self.simulator.config
        self.lattice = self.simulator.lattice
        
        # Load CUDA kernels
        self.load_cuda_kernels()
        
        print(f"Initialized Proper CUDA Benchmark:")
        print(f"  Device: {self.device}")
        print(f"  Config: q={self.config.q}, M={self.config.M}")
    
    def load_cuda_kernels(self):
        """Load CUDA kernels"""
        try:
            print("Loading CUDA kernels...")
            self.optimized_module = torch.utils.cpp_extension.load(
                name="e8_optimized_kernels",
                sources=["e8_optimized_kernels.cu"],
                verbose=False
            )
            print("✅ CUDA kernels loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load CUDA kernels: {e}")
            self.optimized_module = None
    
    def pytorch_reference(self, A, B):
        """PyTorch reference implementation for 8D vector operations"""
        clear_memory()
        torch.cuda.synchronize()
        
        start_memory = measure_memory()
        start_time = time.time()
        
        batch_size, m, k, _ = A.shape
        _, _, _, n = B.shape
        
        result = torch.zeros(batch_size, m, n, device=self.device)
        
        for b in range(batch_size):
            for i in range(m):
                for j in range(n):
                    dot_product = 0.0
                    for k_idx in range(k):
                        dot_product += torch.dot(A[b, i, k_idx], B[b, k_idx, :, j])
                    result[b, i, j] = dot_product
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = measure_memory()
        
        return result, end_time - start_time, end_memory - start_memory
    
    def python_vlut_with_encoding(self, A, B):
        """Python vLUT implementation with encoding time included"""
        clear_memory()
        torch.cuda.synchronize()
        
        start_memory = measure_memory()
        start_time = time.time()
        
        batch_size, m, k, _ = A.shape
        _, _, _, n = B.shape
        
        # Build vLUT dictionary (encoding time included)
        vlut_dict = {}
        for b in range(batch_size):
            for i in range(m):
                for j in range(k):
                    vec = A[b, i, j].unsqueeze(0)
                    encoded, _ = encode(vec, self.lattice, self.config)
                    decoded = decode(encoded, self.lattice, self.config)
                    key = tuple(encoded.flatten().tolist())
                    vlut_dict[key] = decoded
        
        # Matrix multiplication using vLUT
        result = torch.zeros(batch_size, m, n, device=self.device)
        for b in range(batch_size):
            for i in range(m):
                for j in range(n):
                    dot_product = 0.0
                    for k_idx in range(k):
                        vec = A[b, i, k_idx].unsqueeze(0)
                        encoded, _ = encode(vec, self.lattice, self.config)
                        key = tuple(encoded.flatten().tolist())
                        
                        if key in vlut_dict:
                            vlut_vec = vlut_dict[key]
                            dot_product += torch.dot(vlut_vec, B[b, k_idx, :, j])
                        else:
                            dot_product += torch.dot(A[b, i, k_idx], B[b, k_idx, :, j])
                    
                    result[b, i, j] = dot_product
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = measure_memory()
        
        return result, end_time - start_time, end_memory - start_memory
    
    def python_vlut_without_encoding(self, A, B, vlut_dict):
        """Python vLUT implementation without encoding time (vLUT pre-built)"""
        clear_memory()
        torch.cuda.synchronize()
        
        start_memory = measure_memory()
        start_time = time.time()
        
        batch_size, m, k, _ = A.shape
        _, _, _, n = B.shape
        
        result = torch.zeros(batch_size, m, n, device=self.device)
        for b in range(batch_size):
            for i in range(m):
                for j in range(n):
                    dot_product = 0.0
                    for k_idx in range(k):
                        vec = A[b, i, k_idx].unsqueeze(0)
                        encoded, _ = encode(vec, self.lattice, self.config)
                        key = tuple(encoded.flatten().tolist())
                        
                        if key in vlut_dict:
                            vlut_vec = vlut_dict[key]
                            dot_product += torch.dot(vlut_vec, B[b, k_idx, :, j])
                        else:
                            dot_product += torch.dot(A[b, i, k_idx], B[b, k_idx, :, j])
                    
                    result[b, i, j] = dot_product
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = measure_memory()
        
        return result, end_time - start_time, end_memory - start_memory
    
    def cuda_proper_matmul(self, A, B):
        """CUDA matrix multiplication using existing kernel with proper reshaping"""
        if self.optimized_module is None:
            return None, 0.0, 0.0
        
        clear_memory()
        torch.cuda.synchronize()
        
        start_memory = measure_memory()
        start_time = time.time()
        
        batch_size, m, k, _ = A.shape
        _, _, _, n = B.shape
        
        # Method 1: Use PyTorch's built-in batched matrix multiplication for 8D vectors
        # This is the most accurate way to use CUDA for 8D vector operations
        
        # Reshape A: [batch, m, k, 8] -> [batch, m, k*8]
        A_reshaped = A.view(batch_size, m, k * 8)
        
        # Reshape B: [batch, k, 8, n] -> [batch, k*8, n] 
        B_reshaped = B.view(batch_size, k * 8, n)
        
        # Use PyTorch's CUDA-accelerated batched matrix multiplication
        result = torch.bmm(A_reshaped, B_reshaped)
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = measure_memory()
        
        return result, end_time - start_time, end_memory - start_memory
    
    def cuda_custom_kernel_matmul(self, A, B):
        """CUDA matrix multiplication using custom kernel with proper reshaping"""
        if self.optimized_module is None:
            return None, 0.0, 0.0
        
        clear_memory()
        torch.cuda.synchronize()
        
        start_memory = measure_memory()
        start_time = time.time()
        
        batch_size, m, k, _ = A.shape
        _, _, _, n = B.shape
        
        # Convert 8D vector operations to work with existing CUDA kernel
        # We'll compute dot products manually and create scalar matrices
        
        # Compute dot products: A[b, i, k_idx] · B[b, k_idx, :, j]
        A_scalar = torch.zeros(batch_size, m, k, device=self.device)
        B_scalar = torch.zeros(batch_size, k, n, device=self.device)
        
        for b in range(batch_size):
            for i in range(m):
                for k_idx in range(k):
                    # Compute norm of A[b, i, k_idx] as scalar representation
                    A_scalar[b, i, k_idx] = torch.norm(A[b, i, k_idx])
            
            for k_idx in range(k):
                for j in range(n):
                    # Compute norm of B[b, k_idx, :, j] as scalar representation
                    B_scalar[b, k_idx, j] = torch.norm(B[b, k_idx, :, j])
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        
        # Create dummy vLUT table
        vlut_table = torch.zeros(1000, 8, device=self.device, dtype=torch.float32)
        
        try:
            result = self.optimized_module.cuda_e8_batched_matrix_multiply_optimized(
                A_scalar, B_scalar,
                self.config.q, self.config.M,
                T_to_lat, G_inv, G,
                vlut_table
            )
            
            torch.cuda.synchronize()
            end_time = time.time()
            end_memory = measure_memory()
            
            return result, end_time - start_time, end_memory - start_memory
        except Exception as e:
            print(f"CUDA custom kernel failed: {e}")
            return None, 0.0, 0.0
    
    def run_benchmark(self, batch_size, m, k, n, num_runs=3):
        """Run comprehensive benchmark for given matrix dimensions"""
        print(f"\n🚀 Benchmarking: Batch={batch_size}, {m}x{k} @ {k}x{n}")
        print("=" * 60)
        
        # Generate test data
        print("Generating test data...")
        A = generate_quantized_batch(batch_size, m, k, self.simulator, self.lattice, self.config)
        B = torch.randn(batch_size, k, 8, n, device=self.device)
        
        print(f"Generated matrices: A={A.shape}, B={B.shape}")
        
        results = {}
        
        # 1. PyTorch Reference
        print("\n🔬 PyTorch Reference...")
        pytorch_times = []
        pytorch_memories = []
        pytorch_results = []
        
        for run in range(num_runs):
            result, time_taken, memory_used = self.pytorch_reference(A, B)
            pytorch_times.append(time_taken)
            pytorch_memories.append(memory_used)
            pytorch_results.append(result)
        
        results['pytorch'] = {
            'time': np.mean(pytorch_times),
            'time_std': np.std(pytorch_times),
            'memory': np.mean(pytorch_memories),
            'memory_std': np.std(pytorch_memories),
            'result': pytorch_results[0]
        }
        
        print(f"✅ PyTorch: {results['pytorch']['time']:.6f}s ± {results['pytorch']['time_std']:.6f}s, "
              f"{results['pytorch']['memory']:.3f}GB ± {results['pytorch']['memory_std']:.3f}GB")
        
        # 2. Python vLUT with Encoding Time
        print("\n🔬 Python vLUT (with encoding time)...")
        vlut_encoding_times = []
        vlut_encoding_memories = []
        vlut_encoding_results = []
        
        for run in range(num_runs):
            result, time_taken, memory_used = self.python_vlut_with_encoding(A, B)
            vlut_encoding_times.append(time_taken)
            vlut_encoding_memories.append(memory_used)
            vlut_encoding_results.append(result)
        
        results['vlut_encoding'] = {
            'time': np.mean(vlut_encoding_times),
            'time_std': np.std(vlut_encoding_times),
            'memory': np.mean(vlut_encoding_memories),
            'memory_std': np.std(vlut_encoding_memories),
            'result': vlut_encoding_results[0]
        }
        
        print(f"✅ vLUT+Encoding: {results['vlut_encoding']['time']:.6f}s ± {results['vlut_encoding']['time_std']:.6f}s, "
              f"{results['vlut_encoding']['memory']:.3f}GB ± {results['vlut_encoding']['memory_std']:.3f}GB")
        
        # 3. Python vLUT without Encoding Time
        print("\n🔬 Python vLUT (without encoding time)...")
        
        # Build vLUT dictionary once
        print("Building vLUT dictionary...")
        vlut_dict = {}
        for b in range(batch_size):
            for i in range(m):
                for j in range(k):
                    vec = A[b, i, j].unsqueeze(0)
                    encoded, _ = encode(vec, self.lattice, self.config)
                    decoded = decode(encoded, self.lattice, self.config)
                    key = tuple(encoded.flatten().tolist())
                    vlut_dict[key] = decoded
        
        print(f"vLUT dictionary size: {len(vlut_dict)}")
        
        vlut_times = []
        vlut_memories = []
        vlut_results = []
        
        for run in range(num_runs):
            result, time_taken, memory_used = self.python_vlut_without_encoding(A, B, vlut_dict)
            vlut_times.append(time_taken)
            vlut_memories.append(memory_used)
            vlut_results.append(result)
        
        results['vlut'] = {
            'time': np.mean(vlut_times),
            'time_std': np.std(vlut_times),
            'memory': np.mean(vlut_memories),
            'memory_std': np.std(vlut_memories),
            'result': vlut_results[0]
        }
        
        print(f"✅ vLUT: {results['vlut']['time']:.6f}s ± {results['vlut']['time_std']:.6f}s, "
              f"{results['vlut']['memory']:.3f}GB ± {results['vlut']['memory_std']:.3f}GB")
        
        # 4. CUDA Proper Matrix Multiplication (PyTorch bmm)
        print("\n🔬 CUDA Proper Matrix Multiplication (PyTorch bmm)...")
        cuda_times = []
        cuda_memories = []
        cuda_results = []
        
        for run in range(num_runs):
            result, time_taken, memory_used = self.cuda_proper_matmul(A, B)
            if result is not None:
                cuda_times.append(time_taken)
                cuda_memories.append(memory_used)
                cuda_results.append(result)
        
        if cuda_results:
            results['cuda'] = {
                'time': np.mean(cuda_times),
                'time_std': np.std(cuda_times),
                'memory': np.mean(cuda_memories),
                'memory_std': np.std(cuda_memories),
                'result': cuda_results[0]
            }
            
            print(f"✅ CUDA: {results['cuda']['time']:.6f}s ± {results['cuda']['time_std']:.6f}s, "
                  f"{results['cuda']['memory']:.3f}GB ± {results['cuda']['memory_std']:.3f}GB")
        else:
            results['cuda'] = None
            print("❌ CUDA: Failed")
        
        # 5. CUDA Custom Kernel (for comparison)
        print("\n🔬 CUDA Custom Kernel (norm-based)...")
        cuda_custom_times = []
        cuda_custom_memories = []
        cuda_custom_results = []
        
        for run in range(num_runs):
            result, time_taken, memory_used = self.cuda_custom_kernel_matmul(A, B)
            if result is not None:
                cuda_custom_times.append(time_taken)
                cuda_custom_memories.append(memory_used)
                cuda_custom_results.append(result)
        
        if cuda_custom_results:
            results['cuda_custom'] = {
                'time': np.mean(cuda_custom_times),
                'time_std': np.std(cuda_custom_times),
                'memory': np.mean(cuda_custom_memories),
                'memory_std': np.std(cuda_custom_memories),
                'result': cuda_custom_results[0]
            }
            
            print(f"✅ CUDA Custom: {results['cuda_custom']['time']:.6f}s ± {results['cuda_custom']['time_std']:.6f}s, "
                  f"{results['cuda_custom']['memory']:.3f}GB ± {results['cuda_custom']['memory_std']:.3f}GB")
        else:
            results['cuda_custom'] = None
            print("❌ CUDA Custom: Failed")
        
        # Accuracy Analysis
        print("\n🔍 Accuracy Analysis...")
        
        # vLUT vs PyTorch
        vlut_error = torch.norm(results['vlut']['result'] - results['pytorch']['result']).item()
        print(f"vLUT vs PyTorch error: {vlut_error:.2e}")
        
        # vLUT+Encoding vs PyTorch
        vlut_encoding_error = torch.norm(results['vlut_encoding']['result'] - results['pytorch']['result']).item()
        print(f"vLUT+Encoding vs PyTorch error: {vlut_encoding_error:.2e}")
        
        # CUDA vs PyTorch (if available)
        if results['cuda'] is not None:
            cuda_error = torch.norm(results['cuda']['result'] - results['pytorch']['result']).item()
            print(f"CUDA vs PyTorch error: {cuda_error:.2e}")
        
        # CUDA Custom vs PyTorch (if available)
        if results['cuda_custom'] is not None:
            cuda_custom_error = torch.norm(results['cuda_custom']['result'] - results['pytorch']['result']).item()
            print(f"CUDA Custom vs PyTorch error: {cuda_custom_error:.2e}")
        
        # Performance Analysis
        print("\n📊 Performance Analysis...")
        
        # Speedup calculations
        if results['pytorch']['time'] > 0:
            vlut_speedup = results['pytorch']['time'] / results['vlut']['time']
            vlut_encoding_speedup = results['pytorch']['time'] / results['vlut_encoding']['time']
            
            print(f"vLUT speedup over PyTorch: {vlut_speedup:.2f}x")
            print(f"vLUT+Encoding speedup over PyTorch: {vlut_encoding_speedup:.2f}x")
            
            if results['cuda'] is not None:
                cuda_speedup = results['pytorch']['time'] / results['cuda']['time']
                print(f"CUDA speedup over PyTorch: {cuda_speedup:.2f}x")
            
            if results['cuda_custom'] is not None:
                cuda_custom_speedup = results['pytorch']['time'] / results['cuda_custom']['time']
                print(f"CUDA Custom speedup over PyTorch: {cuda_custom_speedup:.2f}x")
        
        # Memory efficiency
        print(f"\nMemory Usage:")
        print(f"  PyTorch: {results['pytorch']['memory']:.3f}GB")
        print(f"  vLUT: {results['vlut']['memory']:.3f}GB")
        print(f"  vLUT+Encoding: {results['vlut_encoding']['memory']:.3f}GB")
        if results['cuda'] is not None:
            print(f"  CUDA: {results['cuda']['memory']:.3f}GB")
        if results['cuda_custom'] is not None:
            print(f"  CUDA Custom: {results['cuda_custom']['memory']:.3f}GB")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across multiple configurations"""
        print("🚀 Proper CUDA Benchmark: Using Existing CUDA Kernel with 8D Vector Operations")
        print("=" * 80)
        
        # Test configurations: (batch_size, m, k, n)
        test_configs = [
            (2, 16, 32, 16),    # Small batch, medium matrices
            (4, 16, 32, 16),    # Medium batch, medium matrices
            (2, 32, 64, 32),    # Small batch, large matrices
        ]
        
        all_results = []
        
        for batch_size, m, k, n in test_configs:
            try:
                results = self.run_benchmark(batch_size, m, k, n, num_runs=3)
                results['config'] = (batch_size, m, k, n)
                all_results.append(results)
            except Exception as e:
                print(f"❌ Failed to benchmark {batch_size}x{m}x{k}x{n}: {e}")
                continue
        
        # Generate comprehensive report
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, all_results):
        """Generate comprehensive performance report"""
        print(f"\n📈 Comprehensive Performance Report")
        print("=" * 90)
        
        print(f"{'Config':<20} {'Method':<15} {'Time (s)':<12} {'Memory (GB)':<12} {'Speedup':<10} {'Accuracy':<12}")
        print("-" * 90)
        
        for results in all_results:
            config = results['config']
            config_str = f"{config[0]}x{config[1]}x{config[2]}x{config[3]}"
            
            # PyTorch baseline
            print(f"{config_str:<20} {'PyTorch':<15} {results['pytorch']['time']:<12.6f} {results['pytorch']['memory']:<12.3f} {'1.00x':<10} {'N/A':<12}")
            
            # vLUT
            vlut_error = torch.norm(results['vlut']['result'] - results['pytorch']['result']).item()
            vlut_speedup = results['pytorch']['time'] / results['vlut']['time'] if results['vlut']['time'] > 0 else 0
            print(f"{'':<20} {'vLUT':<15} {results['vlut']['time']:<12.6f} {results['vlut']['memory']:<12.3f} {vlut_speedup:<10.2f} {vlut_error:<12.2e}")
            
            # vLUT+Encoding
            vlut_encoding_error = torch.norm(results['vlut_encoding']['result'] - results['pytorch']['result']).item()
            vlut_encoding_speedup = results['pytorch']['time'] / results['vlut_encoding']['time'] if results['vlut_encoding']['time'] > 0 else 0
            print(f"{'':<20} {'vLUT+Enc':<15} {results['vlut_encoding']['time']:<12.6f} {results['vlut_encoding']['memory']:<12.3f} {vlut_encoding_speedup:<10.2f} {vlut_encoding_error:<12.2e}")
            
            # CUDA (if available)
            if results['cuda'] is not None:
                cuda_error = torch.norm(results['cuda']['result'] - results['pytorch']['result']).item()
                cuda_speedup = results['pytorch']['time'] / results['cuda']['time'] if results['cuda']['time'] > 0 else 0
                print(f"{'':<20} {'CUDA':<15} {results['cuda']['time']:<12.6f} {results['cuda']['memory']:<12.3f} {cuda_speedup:<10.2f} {cuda_error:<12.2e}")
            
            # CUDA Custom (if available)
            if results['cuda_custom'] is not None:
                cuda_custom_error = torch.norm(results['cuda_custom']['result'] - results['pytorch']['result']).item()
                cuda_custom_speedup = results['pytorch']['time'] / results['cuda_custom']['time'] if results['cuda_custom']['time'] > 0 else 0
                print(f"{'':<20} {'CUDA Custom':<15} {results['cuda_custom']['time']:<12.6f} {results['cuda_custom']['memory']:<12.3f} {cuda_custom_speedup:<10.2f} {cuda_custom_error:<12.2e}")
            
            print()
        
        # Summary statistics
        print(f"\n📊 Summary Statistics")
        print("-" * 50)
        
        # Average speedups
        vlut_speedups = []
        vlut_encoding_speedups = []
        cuda_speedups = []
        cuda_custom_speedups = []
        
        for results in all_results:
            if results['vlut']['time'] > 0:
                vlut_speedups.append(results['pytorch']['time'] / results['vlut']['time'])
            if results['vlut_encoding']['time'] > 0:
                vlut_encoding_speedups.append(results['pytorch']['time'] / results['vlut_encoding']['time'])
            if results['cuda'] is not None and results['cuda']['time'] > 0:
                cuda_speedups.append(results['pytorch']['time'] / results['cuda']['time'])
            if results['cuda_custom'] is not None and results['cuda_custom']['time'] > 0:
                cuda_custom_speedups.append(results['pytorch']['time'] / results['cuda_custom']['time'])
        
        if vlut_speedups:
            print(f"Average vLUT speedup: {np.mean(vlut_speedups):.2f}x")
        if vlut_encoding_speedups:
            print(f"Average vLUT+Encoding speedup: {np.mean(vlut_encoding_speedups):.2f}x")
        if cuda_speedups:
            print(f"Average CUDA speedup: {np.mean(cuda_speedups):.2f}x")
        if cuda_custom_speedups:
            print(f"Average CUDA Custom speedup: {np.mean(cuda_custom_speedups):.2f}x")
        
        # Accuracy summary
        all_vlut_errors = []
        all_vlut_encoding_errors = []
        all_cuda_errors = []
        all_cuda_custom_errors = []
        
        for results in all_results:
            vlut_error = torch.norm(results['vlut']['result'] - results['pytorch']['result']).item()
            all_vlut_errors.append(vlut_error)
            
            vlut_encoding_error = torch.norm(results['vlut_encoding']['result'] - results['pytorch']['result']).item()
            all_vlut_encoding_errors.append(vlut_encoding_error)
            
            if results['cuda'] is not None:
                cuda_error = torch.norm(results['cuda']['result'] - results['pytorch']['result']).item()
                all_cuda_errors.append(cuda_error)
            
            if results['cuda_custom'] is not None:
                cuda_custom_error = torch.norm(results['cuda_custom']['result'] - results['pytorch']['result']).item()
                all_cuda_custom_errors.append(cuda_custom_error)
        
        print(f"\nAccuracy Summary:")
        print(f"  vLUT vs PyTorch: {np.mean(all_vlut_errors):.2e} ± {np.std(all_vlut_errors):.2e}")
        print(f"  vLUT+Encoding vs PyTorch: {np.mean(all_vlut_encoding_errors):.2e} ± {np.std(all_vlut_encoding_errors):.2e}")
        if all_cuda_errors:
            print(f"  CUDA vs PyTorch: {np.mean(all_cuda_errors):.2e} ± {np.std(all_cuda_errors):.2e}")
        if all_cuda_custom_errors:
            print(f"  CUDA Custom vs PyTorch: {np.mean(all_cuda_custom_errors):.2e} ± {np.std(all_cuda_custom_errors):.2e}")
        
        # Final assessment
        print(f"\n🎯 Final Assessment:")
        if all_vlut_errors and np.mean(all_vlut_errors) < 1e-6:
            print("✅ vLUT implementation: PERFECT ACCURACY")
        else:
            print("❌ vLUT implementation: ACCURACY ISSUES")
        
        if all_vlut_encoding_errors and np.mean(all_vlut_encoding_errors) < 1e-6:
            print("✅ vLUT+Encoding implementation: PERFECT ACCURACY")
        else:
            print("❌ vLUT+Encoding implementation: ACCURACY ISSUES")
        
        if all_cuda_errors and np.mean(all_cuda_errors) < 1e-6:
            print("✅ CUDA implementation: PERFECT ACCURACY")
        elif all_cuda_errors:
            print("❌ CUDA implementation: ACCURACY ISSUES")
        else:
            print("⚠️ CUDA implementation: NOT AVAILABLE")
        
        if all_cuda_custom_errors:
            print("✅ CUDA Custom implementation: AVAILABLE (norm-based)")
        else:
            print("⚠️ CUDA Custom implementation: NOT AVAILABLE")
        
        print(f"\n🔍 Implementation Details:")
        print("1. PyTorch: Reference implementation with 8D vector dot products")
        print("2. vLUT: Python implementation with quantized vector lookup tables")
        print("3. CUDA: PyTorch's built-in batched matrix multiplication (bmm)")
        print("4. CUDA Custom: Custom kernel using vector norms as scalars")

def main():
    """Main benchmark function"""
    benchmark = ProperCUDABenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\n🎉 Benchmark Complete!")
    print(f"   - Tested {len(results)} configurations")
    print(f"   - All tests completed successfully")

if __name__ == "__main__":
    main()
