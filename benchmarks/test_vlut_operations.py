"""
Comprehensive vLUT (Value Lookup Table) Testing Suite

This module tests both one-sided and two-sided vLUT operations for MAC and A&A operations.
vLUTs store actual scalar values in full precision (FP32), enabling fast lookup-based
operations rather than real-time computation.

Key Concepts:
- Two-sided vLUT: vlut[i,j] = ⟨lattice_point_i, lattice_point_j⟩ (both operands encoded)
- One-sided vLUT: vlut[i] = ⟨query_vector, lattice_point_i⟩ (query in FP, input encoded)
- Both store precomputed scalar values, not modulo arithmetic (handled by eLUT)
"""

import torch
import numpy as np
import time
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import coset modules
from coset.quant.vlut import vLUTManager, vlut_mac_operation, vlut_accumulate_operation
from coset.quant.params import QuantizationConfig
from coset.lattices.d4 import D4Lattice
from coset.lattices.e8 import E8Lattice
from coset.quant.functional import encode, decode


class vLUTBenchmark:
    """Comprehensive vLUT testing and benchmarking suite."""
    
    def __init__(self, lattice_type: str = "D4", q: int = 4, M: int = 2, device: str = "cpu"):
        """
        Initialize vLUT benchmark suite.
        
        Args:
            lattice_type: Type of lattice ("D4" or "E8")
            q: Quantization parameter
            M: Number of layers
            device: Device to run tests on
        """
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        self.device = torch.device(device)
        
        # Initialize lattice and config
        if lattice_type == "D4":
            self.lattice = D4Lattice()
        elif lattice_type == "E8":
            self.lattice = E8Lattice()
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")
            
        self.config = QuantizationConfig(q=q, M=M)
        self.vlut_manager = vLUTManager(self.lattice, self.config)
        
        # Results storage
        self.results = {
            "lattice_type": lattice_type,
            "q": q,
            "M": M,
            "device": str(device),
            "tests": {}
        }
        
    def generate_test_data(self, batch_size: int = 100, vector_dim: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate test data for vLUT operations.
        
        Args:
            batch_size: Number of vectors to generate
            vector_dim: Dimension of vectors (defaults to lattice dimension)
            
        Returns:
            Tuple of (input_vectors, query_vectors)
        """
        if vector_dim is None:
            vector_dim = self.lattice.d
            
        # Generate random input vectors - each vector has lattice dimension
        input_vectors = torch.randn(batch_size, vector_dim, device=self.device, dtype=torch.float32)
        
        # Generate random query vectors - each vector has lattice dimension
        query_vectors = torch.randn(batch_size, vector_dim, device=self.device, dtype=torch.float32)
        
        return input_vectors, query_vectors
    
    def test_two_sided_vlut_construction(self, batch_size: int = 100) -> Dict:
        """
        Test two-sided vLUT construction and verify correctness.
        
        Args:
            batch_size: Number of test vectors
            
        Returns:
            Test results dictionary
        """
        print(f"Testing two-sided vLUT construction (batch_size={batch_size})")
        
        # Generate test data
        input_vectors, query_vectors = self.generate_test_data(batch_size)
        
        # Build two-sided vLUT
        start_time = time.time()
        two_sided_vlut = self.vlut_manager.build_two_sided_vlut(self.device)
        construction_time = time.time() - start_time
        
        # Verify vLUT properties
        expected_size = self.q ** self.lattice.d
        assert two_sided_vlut.shape == (expected_size, expected_size), f"Expected shape ({expected_size}, {expected_size}), got {two_sided_vlut.shape}"
        assert two_sided_vlut.dtype == torch.float32, f"Expected FP32, got {two_sided_vlut.dtype}"
        
        # Test a few entries manually
        all_encodings = self.vlut_manager._generate_all_encodings()
        lattice_points = self.vlut_manager._decode_encodings_to_lattice_points(all_encodings).to(self.device)
        
        # Verify first few entries
        manual_checks = []
        for i in range(min(5, expected_size)):
            for j in range(min(5, expected_size)):
                manual_value = torch.dot(lattice_points[i], lattice_points[j])
                lut_value = two_sided_vlut[i, j]
                manual_checks.append({
                    "i": i, "j": j,
                    "manual": manual_value.item(),
                    "lut": lut_value.item(),
                    "diff": abs(manual_value.item() - lut_value.item())
                })
        
        results = {
            "construction_time": construction_time,
            "vluT_size": expected_size,
            "vluT_shape": list(two_sided_vlut.shape),
            "vluT_dtype": str(two_sided_vlut.dtype),
            "manual_checks": manual_checks,
            "memory_usage_mb": two_sided_vlut.numel() * 4 / (1024 * 1024)  # FP32 = 4 bytes
        }
        
        print(f"  ✓ Construction time: {construction_time:.4f}s")
        print(f"  ✓ vLUT size: {expected_size}x{expected_size}")
        print(f"  ✓ Memory usage: {results['memory_usage_mb']:.2f} MB")
        
        return results
    
    def test_one_sided_vlut_construction(self, batch_size: int = 100) -> Dict:
        """
        Test one-sided vLUT construction and verify correctness.
        
        Args:
            batch_size: Number of test vectors
            
        Returns:
            Test results dictionary
        """
        print(f"Testing one-sided vLUT construction (batch_size={batch_size})")
        
        # Generate test data
        input_vectors, query_vectors = self.generate_test_data(batch_size)
        
        # Test with first query vector
        query_vector = query_vectors[0]
        
        # Build one-sided vLUT
        start_time = time.time()
        one_sided_vlut = self.vlut_manager.build_one_sided_vlut(query_vector, self.device)
        construction_time = time.time() - start_time
        
        # Verify vLUT properties
        expected_size = self.q ** self.lattice.d
        assert one_sided_vlut.shape == (expected_size,), f"Expected shape ({expected_size},), got {one_sided_vlut.shape}"
        assert one_sided_vlut.dtype == torch.float32, f"Expected FP32, got {one_sided_vlut.dtype}"
        
        # Test a few entries manually
        all_encodings = self.vlut_manager._generate_all_encodings()
        lattice_points = self.vlut_manager._decode_encodings_to_lattice_points(all_encodings).to(self.device)
        query_vector = query_vector.to(self.device)
        
        # Verify first few entries
        manual_checks = []
        for i in range(min(10, expected_size)):
            manual_value = torch.dot(query_vector, lattice_points[i])
            lut_value = one_sided_vlut[i]
            manual_checks.append({
                "i": i,
                "manual": manual_value.item(),
                "lut": lut_value.item(),
                "diff": abs(manual_value.item() - lut_value.item())
            })
        
        results = {
            "construction_time": construction_time,
            "vluT_size": expected_size,
            "vluT_shape": list(one_sided_vlut.shape),
            "vluT_dtype": str(one_sided_vlut.dtype),
            "manual_checks": manual_checks,
            "memory_usage_mb": one_sided_vlut.numel() * 4 / (1024 * 1024)  # FP32 = 4 bytes
        }
        
        print(f"  ✓ Construction time: {construction_time:.4f}s")
        print(f"  ✓ vLUT size: {expected_size}")
        print(f"  ✓ Memory usage: {results['memory_usage_mb']:.2f} MB")
        
        return results
    
    def test_two_sided_mac_operation(self, batch_size: int = 100) -> Dict:
        """
        Test two-sided vLUT MAC operation and compare with real-time computation.
        
        Args:
            batch_size: Number of test vectors
            
        Returns:
            Test results dictionary
        """
        print(f"Testing two-sided vLUT MAC operation (batch_size={batch_size})")
        
        # Generate test data
        input_vectors, query_vectors = self.generate_test_data(batch_size)
        
        # Build two-sided vLUT
        two_sided_vlut = self.vlut_manager.build_two_sided_vlut(self.device)
        
        # Encode vectors (each vector individually since encode expects 1D vectors)
        input_encodings = []
        query_encodings = []
        
        for j in range(batch_size):
            input_enc, _ = encode(input_vectors[j], self.lattice, self.config)
            query_enc, _ = encode(query_vectors[j], self.lattice, self.config)
            input_encodings.append(input_enc)
            query_encodings.append(query_enc)
        
        # Restructure: from [batch_size] tensors of shape [M, d] to [M] tensors of shape [batch_size, d]
        input_encodings_restructured = []
        query_encodings_restructured = []
        
        for i in range(self.M):
            input_layer = torch.stack([enc[i] for enc in input_encodings])  # [batch_size, d]
            query_layer = torch.stack([enc[i] for enc in query_encodings])  # [batch_size, d]
            input_encodings_restructured.append(input_layer)
            query_encodings_restructured.append(query_layer)
        
        # Test vLUT MAC operation
        start_time = time.time()
        vlut_results = vlut_mac_operation(input_encodings_restructured, query_encodings_restructured, two_sided_vlut)
        vlut_time = time.time() - start_time
        
        # Test real-time computation for comparison
        start_time = time.time()
        real_time_results = []
        for i in range(batch_size):
            # Decode each vector's encodings (input_encodings[i] has shape [M, d])
            input_decoded = decode(input_encodings[i], self.lattice, self.config)
            query_decoded = decode(query_encodings[i], self.lattice, self.config)
            result = torch.dot(input_decoded, query_decoded)
            real_time_results.append(result)
        real_time_results = torch.stack(real_time_results)
        real_time_time = time.time() - start_time
        
        # Compare results
        diff = torch.abs(vlut_results - real_time_results)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        results = {
            "vlut_time": vlut_time,
            "real_time_time": real_time_time,
            "speedup": real_time_time / vlut_time if vlut_time > 0 else float('inf'),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "batch_size": batch_size
        }
        
        print(f"  ✓ vLUT MAC time: {vlut_time:.4f}s")
        print(f"  ✓ Real-time MAC time: {real_time_time:.4f}s")
        print(f"  ✓ Speedup: {results['speedup']:.2f}x")
        print(f"  ✓ Max difference: {max_diff:.2e}")
        print(f"  ✓ Mean difference: {mean_diff:.2e}")
        
        return results
    
    def test_one_sided_mac_operation(self, batch_size: int = 100) -> Dict:
        """
        Test one-sided vLUT MAC operation and compare with real-time computation.
        
        Args:
            batch_size: Number of test vectors
            
        Returns:
            Test results dictionary
        """
        print(f"Testing one-sided vLUT MAC operation (batch_size={batch_size})")
        
        # Generate test data
        input_vectors, query_vectors = self.generate_test_data(batch_size)
        
        # Encode input vectors (each vector individually)
        input_encodings = []
        for j in range(batch_size):
            input_enc, _ = encode(input_vectors[j], self.lattice, self.config)
            input_encodings.append(input_enc)
        
        # Restructure: from [batch_size] tensors of shape [M, d] to [M] tensors of shape [batch_size, d]
        input_encodings_restructured = []
        for i in range(self.M):
            input_layer = torch.stack([enc[i] for enc in input_encodings])  # [batch_size, d]
            input_encodings_restructured.append(input_layer)
        
        # Use first query vector (in full precision, not encoded)
        query_vector = query_vectors[0]
        
        # Build one-sided vLUT
        one_sided_vlut = self.vlut_manager.build_one_sided_vlut(query_vector, self.device)
        
        # Test vLUT MAC operation (manual implementation since it's not in the current API)
        start_time = time.time()
        vlut_results = self._one_sided_mac_operation(input_encodings_restructured, one_sided_vlut)
        vlut_time = time.time() - start_time
        
        # Test real-time computation for comparison
        start_time = time.time()
        real_time_results = []
        for i in range(batch_size):
            # Decode each vector's encodings (input_encodings[i] has shape [M, d])
            input_decoded = decode(input_encodings[i], self.lattice, self.config)
            result = torch.dot(input_decoded, query_vector)
            real_time_results.append(result)
        real_time_results = torch.stack(real_time_results)
        real_time_time = time.time() - start_time
        
        # Compare results
        diff = torch.abs(vlut_results - real_time_results)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        results = {
            "vlut_time": vlut_time,
            "real_time_time": real_time_time,
            "speedup": real_time_time / vlut_time if vlut_time > 0 else float('inf'),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "batch_size": batch_size
        }
        
        print(f"  ✓ vLUT MAC time: {vlut_time:.4f}s")
        print(f"  ✓ Real-time MAC time: {real_time_time:.4f}s")
        print(f"  ✓ Speedup: {results['speedup']:.2f}x")
        print(f"  ✓ Max difference: {max_diff:.2e}")
        print(f"  ✓ Mean difference: {mean_diff:.2e}")
        
        return results
    
    def _one_sided_mac_operation(self, encodings: List[torch.Tensor], vlut: torch.Tensor) -> torch.Tensor:
        """
        Manual implementation of one-sided MAC operation.
        
        Args:
            encodings: List of M encoding tensors
            vlut: One-sided vLUT
            
        Returns:
            MAC result tensor
        """
        batch_size = encodings[0].shape[0]
        device = vlut.device
        M = len(encodings)
        
        # Convert encodings to indices
        d = encodings[0].shape[1]
        q = int(round(vlut.shape[0] ** (1.0 / d)))
        
        # Convert all encodings to indices (vectorized)
        idx_all = torch.stack([self._encoding_to_index(encodings[i], q) for i in range(M)])
        
        # Vectorized vLUT lookup
        lut_values = vlut[idx_all]
        
        # MAC operation: sum across M dimension
        result = torch.sum(lut_values, dim=0)
        
        return result
    
    def _encoding_to_index(self, encoding: torch.Tensor, q: int) -> torch.Tensor:
        """Convert encoding to vLUT index (vectorized)."""
        batch_size, d = encoding.shape
        device = encoding.device
        
        # Vectorized computation: idx = Σⱼ encoding[:, j] * q^(d-1-j)
        powers = torch.pow(q, torch.arange(d-1, -1, -1, dtype=torch.long, device=device))
        indices = torch.sum(encoding * powers, dim=1)
        
        return indices.long()
    
    def test_memory_efficiency(self, batch_sizes: List[int] = [10, 50, 100, 500]) -> Dict:
        """
        Test memory efficiency of one-sided vs two-sided vLUTs.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Memory efficiency results
        """
        print(f"Testing memory efficiency for batch sizes: {batch_sizes}")
        
        # Build vLUTs
        two_sided_vlut = self.vlut_manager.build_two_sided_vlut(self.device)
        
        # Generate test query vector
        _, query_vectors = self.generate_test_data(1)
        query_vector = query_vectors[0]
        one_sided_vlut = self.vlut_manager.build_one_sided_vlut(query_vector, self.device)
        
        # Calculate memory usage
        two_sided_memory = two_sided_vlut.numel() * 4 / (1024 * 1024)  # FP32 = 4 bytes
        one_sided_memory = one_sided_vlut.numel() * 4 / (1024 * 1024)
        
        # Test performance with different batch sizes
        batch_results = []
        for batch_size in batch_sizes:
            input_vectors, _ = self.generate_test_data(batch_size)
            
            # Encode input vectors (each vector individually)
            input_encodings = []
            for j in range(batch_size):
                input_enc, _ = encode(input_vectors[j], self.lattice, self.config)
                input_encodings.append(input_enc)
            
            # Restructure: from [batch_size] tensors of shape [M, d] to [M] tensors of shape [batch_size, d]
            input_encodings_restructured = []
            for i in range(self.M):
                input_layer = torch.stack([enc[i] for enc in input_encodings])  # [batch_size, d]
                input_encodings_restructured.append(input_layer)
            
            # Two-sided performance
            start_time = time.time()
            vlut_mac_operation(input_encodings_restructured, input_encodings_restructured, two_sided_vlut)
            two_sided_time = time.time() - start_time
            
            # One-sided performance
            start_time = time.time()
            self._one_sided_mac_operation(input_encodings_restructured, one_sided_vlut)
            one_sided_time = time.time() - start_time
            
            batch_results.append({
                "batch_size": batch_size,
                "two_sided_time": two_sided_time,
                "one_sided_time": one_sided_time
            })
        
        results = {
            "two_sided_memory_mb": two_sided_memory,
            "one_sided_memory_mb": one_sided_memory,
            "memory_ratio": two_sided_memory / one_sided_memory,
            "batch_results": batch_results
        }
        
        print(f"  ✓ Two-sided vLUT memory: {two_sided_memory:.2f} MB")
        print(f"  ✓ One-sided vLUT memory: {one_sided_memory:.2f} MB")
        print(f"  ✓ Memory ratio: {results['memory_ratio']:.2f}x")
        
        return results
    
    def run_comprehensive_benchmark(self, batch_sizes: List[int] = [10, 50, 100]) -> Dict:
        """
        Run comprehensive vLUT benchmark suite.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Complete benchmark results
        """
        print(f"Running comprehensive vLUT benchmark for {self.lattice_type} lattice (q={self.q}, M={self.M})")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        # Run all tests
        self.results["tests"]["two_sided_construction"] = self.test_two_sided_vlut_construction()
        self.results["tests"]["one_sided_construction"] = self.test_one_sided_vlut_construction()
        self.results["tests"]["two_sided_mac"] = self.test_two_sided_mac_operation()
        self.results["tests"]["one_sided_mac"] = self.test_one_sided_mac_operation()
        self.results["tests"]["memory_efficiency"] = self.test_memory_efficiency(batch_sizes)
        
        print("-" * 60)
        print("✓ All vLUT benchmarks completed successfully!")
        
        return self.results
    
    def save_results(self, output_dir: str = "benchmarks/results"):
        """Save benchmark results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"vlut_results_{self.lattice_type}_q{self.q}_M{self.M}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
        return filepath


def main():
    """Main function to run vLUT benchmarks."""
    
    # Test configurations
    configs = [
        {"lattice_type": "D4", "q": 3, "M": 2},
        {"lattice_type": "D4", "q": 4, "M": 2},
        {"lattice_type": "E8", "q": 3, "M": 2},
    ]
    
    batch_sizes = [10, 50, 100]
    
    # Test both CPU and GPU
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"Testing on device: {device}")
        print(f"{'='*80}")
        
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Running vLUT benchmarks for {config} on {device}")
            print(f"{'='*60}")
            
            # Create benchmark instance
            benchmark = vLUTBenchmark(**config, device=device)
            
            # Run comprehensive benchmark
            results = benchmark.run_comprehensive_benchmark(batch_sizes)
            
            # Save results
            benchmark.save_results()
            
            print(f"\nCompleted benchmarks for {config} on {device}")


if __name__ == "__main__":
    main()
