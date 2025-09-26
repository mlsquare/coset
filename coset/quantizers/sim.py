"""
Lattice Vector Simulation System

This module provides tools for simulating vectors from a given lattice with specified
quantization parameters (q, M). It samples encodings, decodes them to generate vectors,
and validates quantizer performance.

Key Features:
- Sample encodings from lattice encoding space
- Decode encodings to generate test vectors
- Validate exact reconstruction for quantized inputs
- Assess quantizer performance on generated datasets
- Generate matrices of specified dimensions
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import time
import statistics
from pathlib import Path

# Import lattice and quantization components
from coset.lattices import Z2Lattice, D4Lattice, E8Lattice
from coset.quant.params import QuantizationConfig
from coset.quant.functional import encode, decode, quantize


class LatticeVectorSimulator:
    """
    Simulator for generating vectors from lattice encodings.
    
    This class provides methods to:
    1. Sample encodings from the lattice's encoding space
    2. Decode encodings to generate test vectors
    3. Validate exact reconstruction for quantized inputs
    4. Assess quantizer performance on generated datasets
    """
    
    def __init__(self, lattice_type: str = "E8", q: int = 3, M: int = 2, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the lattice vector simulator.
        
        Args:
            lattice_type: Type of lattice ("Z2", "D4", "E8")
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            device: Device to use for computations
        """
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        self.device = torch.device(device)
        
        # Initialize lattice
        if lattice_type == "Z2":
            self.lattice = Z2Lattice()
        elif lattice_type == "D4":
            self.lattice = D4Lattice()
        elif lattice_type == "E8":
            self.lattice = E8Lattice()
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")
        
        # Initialize quantization config (no overloading)
        self.config = QuantizationConfig(
            lattice_type=lattice_type,
            q=q,
            M=M,
            beta=1.0,
            alpha=1.0,
            max_scaling_iterations=10,
            with_tie_dither=True,
            with_dither=False,
            disable_scaling=True,  # Disable scaling to prevent overloading
            disable_overload_protection=True  # Disable overload protection
        )
        
        # Calculate encoding space size
        self.encoding_space_size = q ** (M * self.lattice.d)
        self.single_level_size = q ** self.lattice.d
        
        print(f"Initialized {lattice_type} Lattice Simulator:")
        print(f"  Dimension: {self.lattice.d}")
        print(f"  Quantization: q={q}, M={M}")
        print(f"  Encoding space size: {self.encoding_space_size:,}")
        print(f"  Single level size: {self.single_level_size:,}")
        print(f"  Configuration: No overloading (t_values=0, scaling disabled)")
        print(f"  Device: {self.device}")
    
    def sample_encodings(self, batch_size: int, 
                        t_values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random encodings from the lattice's encoding space.
        
        Args:
            batch_size: Number of encodings to sample
            t_values: Optional scaling counts (if None, set to zero - no overloading)
            
        Returns:
            Tuple of (encodings, t_values) where:
            - encodings: [batch_size, M, d] tensor of encoding values
            - t_values: [batch_size] tensor of scaling counts (all zeros)
        """
        # Sample encodings from {0, 1, ..., q-1}
        encodings = torch.randint(0, self.q, (batch_size, self.M, self.lattice.d), 
                                 device=self.device, dtype=torch.int32)
        
        # Set t_values to zero (no overloading)
        if t_values is None:
            t_values = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        else:
            t_values = t_values.to(self.device)
        
        return encodings, t_values
    
    def decode_encodings(self, encodings: torch.Tensor, t_values: torch.Tensor) -> torch.Tensor:
        """
        Decode encodings to generate reconstructed vectors.
        
        Args:
            encodings: [batch_size, M, d] tensor of encoding values
            t_values: [batch_size] tensor of scaling counts
            
        Returns:
            [batch_size, d] tensor of reconstructed vectors
        """
        batch_size = encodings.shape[0]
        vectors = torch.zeros((batch_size, self.lattice.d), device=self.device, dtype=torch.float32)
        
        # Decode each vector
        for i in range(batch_size):
            try:
                vector = decode(encodings[i], self.lattice, self.config, t_values[i])
                vectors[i] = vector
            except Exception as e:
                print(f"Warning: Failed to decode vector {i}: {e}")
                # Use zero vector as fallback
                vectors[i] = torch.zeros(self.lattice.d, device=self.device, dtype=torch.float32)
        
        return vectors
    
    def generate_vectors(self, batch_size: int, 
                        t_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Complete pipeline: sample encodings and decode to generate vectors.
        
        Args:
            batch_size: Number of vectors to generate
            t_values: Optional scaling counts
            
        Returns:
            [batch_size, d] tensor of generated vectors
        """
        # Sample encodings
        encodings, t_values = self.sample_encodings(batch_size, t_values)
        
        # Decode to get vectors
        vectors = self.decode_encodings(encodings, t_values)
        
        return vectors
    
    def validate_reconstruction(self, original_vectors: torch.Tensor, 
                              tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Validate exact reconstruction for quantized inputs.
        
        For properly quantized inputs, the quantizer should reconstruct them exactly.
        
        Args:
            original_vectors: [batch_size, d] tensor of input vectors
            tolerance: Numerical tolerance for exact reconstruction
            
        Returns:
            Dictionary with validation metrics
        """
        batch_size = original_vectors.shape[0]
        original_vectors = original_vectors.to(self.device)
        
        # Encode the original vectors
        encodings_list = []
        t_values_list = []
        
        for i in range(batch_size):
            try:
                enc, t_val = encode(original_vectors[i], self.lattice, self.config)
                encodings_list.append(enc)
                t_values_list.append(t_val)
            except Exception as e:
                print(f"Warning: Failed to encode vector {i}: {e}")
                # Use zero encoding as fallback
                encodings_list.append(torch.zeros(self.M, self.lattice.d, device=self.device, dtype=torch.int32))
                t_values_list.append(0)
        
        # Stack encodings and t_values
        encodings = torch.stack(encodings_list)
        t_values = torch.tensor(t_values_list, device=self.device, dtype=torch.int32)
        
        # Decode to get reconstructed vectors
        reconstructed = self.decode_encodings(encodings, t_values)
        
        # Calculate reconstruction errors
        errors = torch.norm(original_vectors - reconstructed, dim=1)
        
        # Calculate metrics
        max_error = torch.max(errors).item()
        mean_error = torch.mean(errors).item()
        std_error = torch.std(errors).item()
        
        # Count exact reconstructions
        exact_reconstructions = torch.sum(errors < tolerance).item()
        exact_rate = exact_reconstructions / batch_size
        
        return {
            'max_error': max_error,
            'mean_error': mean_error,
            'std_error': std_error,
            'exact_reconstructions': exact_reconstructions,
            'exact_rate': exact_rate,
            'tolerance': tolerance
        }
    
    def assess_quantizer_performance(self, test_vectors: torch.Tensor, 
                                   num_iterations: int = 10) -> Dict[str, float]:
        """
        Assess quantizer performance on generated test vectors.
        
        Args:
            test_vectors: [batch_size, d] tensor of test vectors
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with performance metrics
        """
        batch_size = test_vectors.shape[0]
        test_vectors = test_vectors.to(self.device)
        
        # Warm up
        for _ in range(3):
            _ = self.validate_reconstruction(test_vectors[:min(100, batch_size)])
        
        # Time encoding
        encode_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            for i in range(batch_size):
                _ = encode(test_vectors[i], self.lattice, self.config)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            encode_times.append(time.perf_counter() - start_time)
        
        # Time decoding
        decode_times = []
        encodings_list = []
        t_values_list = []
        
        # First encode all vectors
        for i in range(batch_size):
            enc, t_val = encode(test_vectors[i], self.lattice, self.config)
            encodings_list.append(enc)
            t_values_list.append(t_val)
        
        encodings = torch.stack(encodings_list)
        t_values = torch.tensor(t_values_list, device=self.device, dtype=torch.int32)
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            _ = self.decode_encodings(encodings, t_values)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            decode_times.append(time.perf_counter() - start_time)
        
        # Calculate throughput
        encode_throughput = batch_size / statistics.mean(encode_times)
        decode_throughput = batch_size / statistics.mean(decode_times)
        
        return {
            'encode_time_mean': statistics.mean(encode_times),
            'encode_time_std': statistics.stdev(encode_times) if len(encode_times) > 1 else 0,
            'decode_time_mean': statistics.mean(decode_times),
            'decode_time_std': statistics.stdev(decode_times) if len(decode_times) > 1 else 0,
            'encode_throughput': encode_throughput,
            'decode_throughput': decode_throughput,
            'total_throughput': batch_size / (statistics.mean(encode_times) + statistics.mean(decode_times))
        }
    
    def generate_matrix(self, rows: int, cols: int) -> torch.Tensor:
        """
        Generate a matrix by tiling vectors to match lattice dimension.
        
        Args:
            rows: Number of rows in the matrix
            cols: Number of columns in the matrix
            
        Returns:
            [rows, cols] tensor representing the generated matrix
        """
        # Calculate how many vectors we need
        d = self.lattice.d
        vectors_needed = (rows * cols + d - 1) // d  # Ceiling division
        
        # Generate vectors
        vectors = self.generate_vectors(vectors_needed)
        
        # Reshape to matrix
        matrix = vectors.view(-1)[:rows * cols].view(rows, cols)
        
        return matrix
    
    def benchmark_simulation(self, batch_sizes: List[int] = None) -> Dict[str, Dict]:
        """
        Benchmark the simulation system across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [100, 1000, 10000]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size: {batch_size}")
            
            # Generate vectors
            start_time = time.perf_counter()
            vectors = self.generate_vectors(batch_size)
            generation_time = time.perf_counter() - start_time
            
            # Validate reconstruction
            validation_results = self.validate_reconstruction(vectors)
            
            # Assess performance
            performance_results = self.assess_quantizer_performance(vectors)
            
            results[batch_size] = {
                'generation_time': generation_time,
                'generation_throughput': batch_size / generation_time,
                'validation': validation_results,
                'performance': performance_results
            }
            
            print(f"  Generation: {generation_time:.4f}s ({batch_size/generation_time:.0f} vec/s)")
            print(f"  Exact reconstruction rate: {validation_results['exact_rate']:.2%}")
            print(f"  Mean error: {validation_results['mean_error']:.6f}")
        
        return results


def create_simulator(lattice_type: str = "E8", q: int = 3, M: int = 2, 
                    device: str = "cuda" if torch.cuda.is_available() else "cpu") -> LatticeVectorSimulator:
    """
    Convenience function to create a lattice vector simulator.
    
    Args:
        lattice_type: Type of lattice ("Z2", "D4", "E8")
        q: Quantization parameter
        M: Number of hierarchical levels
        device: Device to use
        
    Returns:
        LatticeVectorSimulator instance
    """
    return LatticeVectorSimulator(lattice_type, q, M, device)


def demo_simulation():
    """Demonstrate the lattice vector simulation system."""
    print("🚀 Lattice Vector Simulation Demo")
    print("=" * 50)
    
    # Create E8 simulator
    simulator = create_simulator("E8", q=3, M=2)
    
    # Generate some vectors
    print("\n📊 Generating test vectors...")
    vectors = simulator.generate_vectors(1000)
    print(f"Generated {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    print(f"Vector range: [{torch.min(vectors):.4f}, {torch.max(vectors):.4f}]")
    print(f"Vector mean: {torch.mean(vectors):.4f}")
    print(f"Vector std: {torch.std(vectors):.4f}")
    
    # Validate reconstruction
    print("\n🔍 Validating reconstruction...")
    validation_results = simulator.validate_reconstruction(vectors)
    print(f"Exact reconstruction rate: {validation_results['exact_rate']:.2%}")
    print(f"Mean error: {validation_results['mean_error']:.6f}")
    print(f"Max error: {validation_results['max_error']:.6f}")
    
    # Assess performance
    print("\n⚡ Assessing performance...")
    performance_results = simulator.assess_quantizer_performance(vectors)
    print(f"Encode throughput: {performance_results['encode_throughput']:.0f} vec/s")
    print(f"Decode throughput: {performance_results['decode_throughput']:.0f} vec/s")
    print(f"Total throughput: {performance_results['total_throughput']:.0f} vec/s")
    
    # Generate a matrix
    print("\n📐 Generating matrix...")
    matrix = simulator.generate_matrix(100, 256)
    print(f"Generated matrix: {matrix.shape}")
    print(f"Matrix range: [{torch.min(matrix):.4f}, {torch.max(matrix):.4f}]")
    
    # Benchmark across batch sizes
    print("\n📈 Benchmarking across batch sizes...")
    benchmark_results = simulator.benchmark_simulation([100, 1000, 5000])
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demo_simulation()
