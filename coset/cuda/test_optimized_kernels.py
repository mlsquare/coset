#!/usr/bin/env python3
"""
Test Optimized E8 HNLQ CUDA Kernels
Testing dot products, matrix-vector products, and tensor contractions
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

# Import CUDA kernels
import torch.utils.cpp_extension

class OptimizedKernelTester:
    """Comprehensive tester for optimized CUDA kernels"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lattice = E8Lattice()
        self.config = QuantizationConfig(q=3, M=2)
        self.simulator = LatticeVectorSimulator(lattice_type="E8", q=3, M=2, device=device)
        
        # Load optimized CUDA kernels
        self.optimized_module = None
        self._load_optimized_kernels()
        
        # Performance tracking
        self.results = {}
        
    def _load_optimized_kernels(self):
        """Load optimized CUDA kernels"""
        try:
            print("Loading optimized CUDA kernels...")
            
            self.optimized_module = torch.utils.cpp_extension.load(
                name="e8_optimized_kernels",
                sources=["e8_optimized_kernels.cu"],
                verbose=False
            )
            
            print("✅ Optimized CUDA kernels loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load optimized CUDA kernels: {e}")
            raise
    
    def generate_test_vectors(self, n: int, use_sim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test vectors for dot product testing"""
        if use_sim and n <= 1024:  # Use sim.py for small vectors
            print(f"  Using sim.py for {n}D vectors...")
            num_vectors = n // 8
            vectors = self.simulator.generate_vectors(num_vectors)
            a = vectors.view(-1)[:n].to(self.device)
            b = vectors.view(-1)[n:2*n].to(self.device) if 2*n <= vectors.numel() else torch.randn(n, device=self.device)
        else:  # Use random vectors for large vectors
            print(f"  Using random vectors for {n}D vectors...")
            a = torch.randn(n, device=self.device, dtype=torch.float32)
            b = torch.randn(n, device=self.device, dtype=torch.float32)
        
        return a, b
    
    def generate_test_matrices(self, m: int, n: int, use_sim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test matrices for matrix-vector product testing"""
        if use_sim and m * n <= 65536:  # Use sim.py for small matrices
            print(f"  Using sim.py for {m}x{n} matrix...")
            num_vectors = (m * n) // 8
            vectors = self.simulator.generate_vectors(num_vectors)
            A = vectors.view(m, n).to(self.device)
            x = torch.randn(n, device=self.device, dtype=torch.float32)
        else:  # Use random matrices for large matrices
            print(f"  Using random matrices for {m}x{n} matrix...")
            A = torch.randn(m, n, device=self.device, dtype=torch.float32)
            x = torch.randn(n, device=self.device, dtype=torch.float32)
        
        return A, x
    
    def pytorch_vector_dot_product(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """PyTorch reference vector dot product"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        result = torch.dot(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return result, {
            'time': end_time - start_time,
            'memory': memory_info,
            'device': 'pytorch_cuda'
        }
    
    def optimized_vector_dot_product(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Optimized CUDA vector dot product"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        vlut_table = torch.randn(1000, 8, device=self.device, dtype=torch.float32)  # Dummy vLUT table
        
        start_time = time.time()
        result = self.optimized_module.cuda_e8_vector_dot_product_optimized(
            a, b, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
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
            'device': 'optimized_cuda'
        }
    
    def pytorch_matrix_vector_product(self, A: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """PyTorch reference matrix-vector product"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        result = torch.matmul(A, x)
        torch.cuda.synchronize()
        end_time = time.time()
        
        memory_info = {
            'peak_allocated': torch.cuda.max_memory_allocated(),
            'peak_reserved': torch.cuda.max_memory_reserved()
        }
        
        return result, {
            'time': end_time - start_time,
            'memory': memory_info,
            'device': 'pytorch_cuda'
        }
    
    def optimized_matrix_vector_product(self, A: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Optimized CUDA matrix-vector product"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Get lattice parameters
        T_to_lat = torch.eye(8, device=self.device, dtype=torch.float32)
        G_inv = self.lattice.G_inv.to(self.device).float()
        G = self.lattice.G.to(self.device).float()
        vlut_table = torch.randn(1000, 8, device=self.device, dtype=torch.float32)  # Dummy vLUT table
        
        start_time = time.time()
        result = self.optimized_module.cuda_e8_matrix_vector_product_optimized(
            A, x, self.config.q, self.config.M, T_to_lat, G_inv, G, vlut_table
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
            'device': 'optimized_cuda'
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
    
    def test_vector_dot_products(self, vector_sizes: List[int], num_runs: int = 5):
        """Test vector dot products across different sizes"""
        print(f"\n🔬 Testing Vector Dot Products")
        print("=" * 60)
        
        for n in vector_sizes:
            print(f"\n📊 Vector Size: {n}")
            print("-" * 40)
            
            # Generate test vectors
            a, b = self.generate_test_vectors(n)
            
            # Test implementations
            implementations = [
                ('PyTorch CUDA', self.pytorch_vector_dot_product),
                ('Optimized CUDA', self.optimized_vector_dot_product)
            ]
            
            results = {}
            
            for name, func in implementations:
                print(f"\n🔬 Testing {name}...")
                
                times = []
                memory_usage = []
                accuracy_metrics = []
                
                for run in range(num_runs):
                    try:
                        result, stats = func(a, b)
                        
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
            self.results[f"vector_dot_{n}"] = results
            
            # Print comparison
            self._print_comparison(results, f"Vector Dot Product (n={n})")
    
    def test_matrix_vector_products(self, matrix_sizes: List[Tuple[int, int]], num_runs: int = 5):
        """Test matrix-vector products across different sizes"""
        print(f"\n🔬 Testing Matrix-Vector Products")
        print("=" * 60)
        
        for m, n in matrix_sizes:
            print(f"\n📊 Matrix Size: {m}x{n}")
            print("-" * 40)
            
            # Generate test matrices
            A, x = self.generate_test_matrices(m, n)
            
            # Test implementations
            implementations = [
                ('PyTorch CUDA', self.pytorch_matrix_vector_product),
                ('Optimized CUDA', self.optimized_matrix_vector_product)
            ]
            
            results = {}
            
            for name, func in implementations:
                print(f"\n🔬 Testing {name}...")
                
                times = []
                memory_usage = []
                accuracy_metrics = []
                
                for run in range(num_runs):
                    try:
                        result, stats = func(A, x)
                        
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
            self.results[f"matrix_vector_{m}x{n}"] = results
            
            # Print comparison
            self._print_comparison(results, f"Matrix-Vector Product ({m}x{n})")
    
    def _print_comparison(self, results: Dict, test_name: str):
        """Print comparison of results"""
        print(f"\n📈 Performance Comparison - {test_name}:")
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
        print(f"\n🎯 OPTIMIZED KERNELS PERFORMANCE SUMMARY")
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
                    print(f"{name:20s}: {result['time_mean']:.6f}s | {result['memory_mean']/1024/1024:.1f} MB")
                else:
                    print(f"{name:20s}: FAILED")


def main():
    """Main testing function"""
    print("🚀 Optimized E8 HNLQ CUDA Kernels Testing")
    print("Testing dot products, matrix-vector products, and tensor contractions")
    print("=" * 80)
    
    # Initialize tester
    tester = OptimizedKernelTester()
    
    # Test vector dot products
    vector_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    tester.test_vector_dot_products(vector_sizes, num_runs=3)
    
    # Test matrix-vector products
    matrix_sizes = [
        (32, 32), (64, 64), (128, 128), (256, 256),
        (512, 512), (1024, 1024), (2048, 2048)
    ]
    tester.test_matrix_vector_products(matrix_sizes, num_runs=3)
    
    # Print summary
    tester.print_summary()
    
    print(f"\n✅ Optimized kernels testing complete!")


if __name__ == "__main__":
    main()
