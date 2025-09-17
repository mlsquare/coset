#!/usr/bin/env python3
"""
Combined Encoding+Decoding Performance Profiler

This script profiles the complete quantization process (encode + decode) for hierarchical 
nested-lattice quantization, comparing baseline PyTorch implementation against CUDA-optimized versions.

Usage:
    python benchmarks/profile_combined.py [--lattice D4] [--q 4] [--M 2] [--batch-sizes 100,1000,10000]
"""

import argparse
import time
import statistics
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coset.lattices import D4Lattice, Z2Lattice, E8Lattice
from coset.quant.params import QuantizationConfig
from coset.quant.functional import encode, decode, quantize
from coset.cuda.kernels import cuda_quantize, is_cuda_available


class CombinedProfiler:
    """Profiler for combined encoding+decoding operations with baseline and CUDA implementations."""
    
    def __init__(self, lattice_type: str = "D4", q: int = 4, M: int = 2):
        """
        Initialize the profiler.
        
        Args:
            lattice_type: Type of lattice ("Z2", "D4", "E8")
            q: Quantization parameter
            M: Number of hierarchical levels
        """
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        
        # Initialize lattice and config
        if lattice_type == "Z2":
            self.lattice = Z2Lattice()
        elif lattice_type == "D4":
            self.lattice = D4Lattice()
        elif lattice_type == "E8":
            self.lattice = E8Lattice()
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")
            
        self.config = QuantizationConfig(
            lattice_type=lattice_type,
            q=q,
            M=M,
            beta=1.0,
            disable_scaling=True,  # For consistent profiling
            disable_overload_protection=True
        )
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available() and is_cuda_available()
        if self.cuda_available:
            self.device = torch.device("cuda")
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print("CUDA kernels loaded successfully")
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                print("CUDA available but kernels not loaded, using CPU only")
            else:
                print("CUDA not available, using CPU only")
    
    def generate_test_data(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate test data for profiling."""
        # Generate random vectors in the lattice dimension
        d = self.lattice.d
        x = torch.randn(batch_size, d, device=device, dtype=torch.float32)
        
        # Scale to reasonable range to avoid overload issues
        x = x * 0.5  # Scale down to reduce overload probability
        
        return x
    
    def baseline_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Baseline quantization using current PyTorch implementation (encode + decode).
        
        Args:
            x: Input tensor of shape [batch_size, d]
            
        Returns:
            Quantized and reconstructed tensor of shape [batch_size, d]
        """
        batch_size = x.shape[0]
        quantized = []
        
        for i in range(batch_size):
            x_hat = quantize(x[i], self.lattice, self.config)
            quantized.append(x_hat)
        
        return torch.stack(quantized)
    
    def baseline_encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Baseline quantization using separate encode + decode operations.
        
        Args:
            x: Input tensor of shape [batch_size, d]
            
        Returns:
            Quantized and reconstructed tensor of shape [batch_size, d]
        """
        batch_size = x.shape[0]
        quantized = []
        
        for i in range(batch_size):
            # Encode
            b, t = encode(x[i], self.lattice, self.config)
            # Decode
            x_hat = decode(b, self.lattice, self.config, t)
            quantized.append(x_hat)
        
        return torch.stack(quantized)
    
    def cuda_optimized_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        CUDA-optimized quantization using actual CUDA kernels.
        
        Args:
            x: Input tensor of shape [batch_size, d]
            
        Returns:
            Quantized and reconstructed tensor of shape [batch_size, d]
        """
        if not self.cuda_available:
            # Fallback to baseline if CUDA not available
            return self.baseline_quantize(x)
        
        try:
            # Use real CUDA kernel
            return cuda_quantize(x, self.lattice, self.config)
        except Exception as e:
            print(f"CUDA quantization failed, falling back to baseline: {e}")
            return self.baseline_quantize(x)
    
    def profile_single_batch(self, batch_size: int, num_warmup: int = 5, num_iterations: int = 20) -> Dict[str, Any]:
        """
        Profile combined quantization for a single batch size.
        
        Args:
            batch_size: Size of the batch to test
            num_warmup: Number of warmup iterations
            num_iterations: Number of timing iterations
            
        Returns:
            Dictionary with timing results
        """
        print(f"Profiling batch size: {batch_size}")
        
        # Generate test data
        x = self.generate_test_data(batch_size, self.device)
        
        results = {
            'batch_size': batch_size,
            'lattice_type': self.lattice_type,
            'q': self.q,
            'M': self.M,
            'device': str(self.device)
        }
        
        # Profile baseline quantize() function
        print("  Profiling baseline quantize() function...")
        baseline_quantize_times = []
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.baseline_quantize(x)
            if self.cuda_available:
                torch.cuda.synchronize()
        
        # Timing
        for _ in range(num_iterations):
            if self.cuda_available:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            quantized = self.baseline_quantize(x)
            
            if self.cuda_available:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            baseline_quantize_times.append(end_time - start_time)
        
        results['baseline_quantize'] = {
            'mean_time': statistics.mean(baseline_quantize_times),
            'std_time': statistics.stdev(baseline_quantize_times) if len(baseline_quantize_times) > 1 else 0,
            'min_time': min(baseline_quantize_times),
            'max_time': max(baseline_quantize_times),
            'throughput': batch_size / statistics.mean(baseline_quantize_times),  # vectors/sec
            'all_times': baseline_quantize_times
        }
        
        # Profile baseline encode+decode operations
        print("  Profiling baseline encode+decode operations...")
        baseline_encode_decode_times = []
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.baseline_encode_decode(x)
            if self.cuda_available:
                torch.cuda.synchronize()
        
        # Timing
        for _ in range(num_iterations):
            if self.cuda_available:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            quantized = self.baseline_encode_decode(x)
            
            if self.cuda_available:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            baseline_encode_decode_times.append(end_time - start_time)
        
        results['baseline_encode_decode'] = {
            'mean_time': statistics.mean(baseline_encode_decode_times),
            'std_time': statistics.stdev(baseline_encode_decode_times) if len(baseline_encode_decode_times) > 1 else 0,
            'min_time': min(baseline_encode_decode_times),
            'max_time': max(baseline_encode_decode_times),
            'throughput': batch_size / statistics.mean(baseline_encode_decode_times),  # vectors/sec
            'all_times': baseline_encode_decode_times
        }
        
        # Profile CUDA-optimized implementation
        print("  Profiling CUDA-optimized implementation...")
        cuda_times = []
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.cuda_optimized_quantize(x)
            if self.cuda_available:
                torch.cuda.synchronize()
        
        # Timing
        for _ in range(num_iterations):
            if self.cuda_available:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            quantized = self.cuda_optimized_quantize(x)
            
            if self.cuda_available:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            cuda_times.append(end_time - start_time)
        
        results['cuda_optimized'] = {
            'mean_time': statistics.mean(cuda_times),
            'std_time': statistics.stdev(cuda_times) if len(cuda_times) > 1 else 0,
            'min_time': min(cuda_times),
            'max_time': max(cuda_times),
            'throughput': batch_size / statistics.mean(cuda_times),  # vectors/sec
            'all_times': cuda_times
        }
        
        # Calculate speedups
        speedup_vs_quantize = results['baseline_quantize']['mean_time'] / results['cuda_optimized']['mean_time']
        speedup_vs_encode_decode = results['baseline_encode_decode']['mean_time'] / results['cuda_optimized']['mean_time']
        
        results['speedup_vs_quantize'] = speedup_vs_quantize
        results['speedup_vs_encode_decode'] = speedup_vs_encode_decode
        
        print(f"    Baseline quantize():     {results['baseline_quantize']['mean_time']*1000:.2f}ms ± {results['baseline_quantize']['std_time']*1000:.2f}ms")
        print(f"    Baseline encode+decode:  {results['baseline_encode_decode']['mean_time']*1000:.2f}ms ± {results['baseline_encode_decode']['std_time']*1000:.2f}ms")
        print(f"    CUDA optimized:          {results['cuda_optimized']['mean_time']*1000:.2f}ms ± {results['cuda_optimized']['std_time']*1000:.2f}ms")
        print(f"    Speedup vs quantize():   {speedup_vs_quantize:.2f}x")
        print(f"    Speedup vs encode+decode: {speedup_vs_encode_decode:.2f}x")
        print(f"    Throughput: {results['cuda_optimized']['throughput']:.0f} vectors/sec")
        
        return results
    
    def profile_multiple_batches(self, batch_sizes: List[int]) -> List[Dict[str, Any]]:
        """
        Profile combined quantization for multiple batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of results for each batch size
        """
        all_results = []
        
        for batch_size in batch_sizes:
            try:
                result = self.profile_single_batch(batch_size)
                all_results.append(result)
            except Exception as e:
                print(f"Error profiling batch size {batch_size}: {e}")
                continue
        
        return all_results
    
    def plot_results(self, results: List[Dict[str, Any]], output_dir: Path):
        """Plot profiling results."""
        if not results:
            print("No results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Extract data for plotting
        batch_sizes = [r['batch_size'] for r in results]
        baseline_quantize_times = [r['baseline_quantize']['mean_time'] * 1000 for r in results]  # Convert to ms
        baseline_encode_decode_times = [r['baseline_encode_decode']['mean_time'] * 1000 for r in results]
        cuda_times = [r['cuda_optimized']['mean_time'] * 1000 for r in results]
        speedups_quantize = [r['speedup_vs_quantize'] for r in results]
        speedups_encode_decode = [r['speedup_vs_encode_decode'] for r in results]
        baseline_quantize_throughput = [r['baseline_quantize']['throughput'] for r in results]
        baseline_encode_decode_throughput = [r['baseline_encode_decode']['throughput'] for r in results]
        cuda_throughput = [r['cuda_optimized']['throughput'] for r in results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Execution time vs batch size
        ax1.loglog(batch_sizes, baseline_quantize_times, 'o-', label='Baseline quantize()', linewidth=2, markersize=8)
        ax1.loglog(batch_sizes, baseline_encode_decode_times, '^-', label='Baseline encode+decode', linewidth=2, markersize=8)
        ax1.loglog(batch_sizes, cuda_times, 's-', label='CUDA Optimized', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'Combined Quantization Time vs Batch Size\n({self.lattice_type}, q={self.q}, M={self.M})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup vs batch size
        ax2.semilogx(batch_sizes, speedups_quantize, 'o-', label='vs quantize()', linewidth=2, markersize=8)
        ax2.semilogx(batch_sizes, speedups_encode_decode, '^-', label='vs encode+decode', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('CUDA Speedup vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput vs batch size
        ax3.loglog(batch_sizes, baseline_quantize_throughput, 'o-', label='Baseline quantize()', linewidth=2, markersize=8)
        ax3.loglog(batch_sizes, baseline_encode_decode_throughput, '^-', label='Baseline encode+decode', linewidth=2, markersize=8)
        ax3.loglog(batch_sizes, cuda_throughput, 's-', label='CUDA Optimized', linewidth=2, markersize=8)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Throughput (vectors/sec)')
        ax3.set_title('Combined Quantization Throughput vs Batch Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory efficiency (compression ratio)
        # Calculate compression ratio: original_size / encoded_size
        original_size = [bs * self.lattice.d * 4 for bs in batch_sizes]  # 4 bytes per float32
        encoded_size = [bs * self.M * self.lattice.d * 1 for bs in batch_sizes]  # 1 byte per uint8
        compression_ratios = [orig / enc for orig, enc in zip(original_size, encoded_size)]
        
        ax4.semilogx(batch_sizes, compression_ratios, 'o-', color='green', linewidth=2, markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Compression Ratio')
        ax4.set_title('Memory Compression Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_dir / f'combined_profile_{self.lattice_type}_q{self.q}_M{self.M}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: Path):
        """Save profiling results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        import json
        results_path = output_dir / f'combined_results_{self.lattice_type}_q{self.q}_M{self.M}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            json_result['baseline_quantize']['all_times'] = result['baseline_quantize']['all_times']
            json_result['baseline_encode_decode']['all_times'] = result['baseline_encode_decode']['all_times']
            json_result['cuda_optimized']['all_times'] = result['cuda_optimized']['all_times']
            json_results.append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
        # Save summary as CSV
        import csv
        csv_path = output_dir / f'combined_summary_{self.lattice_type}_q{self.q}_M{self.M}.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'batch_size', 'lattice_type', 'q', 'M', 'device',
                'baseline_quantize_time_ms', 'baseline_quantize_std_ms', 'baseline_quantize_throughput',
                'baseline_encode_decode_time_ms', 'baseline_encode_decode_std_ms', 'baseline_encode_decode_throughput',
                'cuda_time_ms', 'cuda_std_ms', 'cuda_throughput',
                'speedup_vs_quantize', 'speedup_vs_encode_decode'
            ])
            
            for result in results:
                writer.writerow([
                    result['batch_size'],
                    result['lattice_type'],
                    result['q'],
                    result['M'],
                    result['device'],
                    result['baseline_quantize']['mean_time'] * 1000,
                    result['baseline_quantize']['std_time'] * 1000,
                    result['baseline_quantize']['throughput'],
                    result['baseline_encode_decode']['mean_time'] * 1000,
                    result['baseline_encode_decode']['std_time'] * 1000,
                    result['baseline_encode_decode']['throughput'],
                    result['cuda_optimized']['mean_time'] * 1000,
                    result['cuda_optimized']['std_time'] * 1000,
                    result['cuda_optimized']['throughput'],
                    result['speedup_vs_quantize'],
                    result['speedup_vs_encode_decode']
                ])
        
        print(f"Summary saved to: {csv_path}")


def main():
    """Main function for the combined profiler."""
    parser = argparse.ArgumentParser(description='Profile combined quantization performance')
    parser.add_argument('--lattice', type=str, default='D4', choices=['Z2', 'D4', 'E8'],
                        help='Lattice type to use')
    parser.add_argument('--q', type=int, default=4, help='Quantization parameter')
    parser.add_argument('--M', type=int, default=2, help='Number of hierarchical levels')
    parser.add_argument('--batch-sizes', type=str, default='1,10,100,1000,10000',
                        help='Comma-separated list of batch sizes to test')
    parser.add_argument('--output-dir', type=str, default='benchmarks/results',
                        help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Combined Quantization Performance Profiler")
    print(f"Lattice: {args.lattice}, q={args.q}, M={args.M}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Initialize profiler
    profiler = CombinedProfiler(lattice_type=args.lattice, q=args.q, M=args.M)
    
    # Run profiling
    results = profiler.profile_multiple_batches(batch_sizes)
    
    if not results:
        print("No results obtained. Exiting.")
        return
    
    # Save results
    profiler.save_results(results, output_dir)
    
    # Plot results
    if not args.no_plot:
        profiler.plot_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for result in results:
        bs = result['batch_size']
        baseline_quantize_tp = result['baseline_quantize']['throughput']
        baseline_encode_decode_tp = result['baseline_encode_decode']['throughput']
        cuda_tp = result['cuda_optimized']['throughput']
        speedup_quantize = result['speedup_vs_quantize']
        speedup_encode_decode = result['speedup_vs_encode_decode']
        
        print(f"Batch size {bs:>6}:")
        print(f"  Baseline quantize():     {baseline_quantize_tp:>8.0f} vectors/sec")
        print(f"  Baseline encode+decode:  {baseline_encode_decode_tp:>8.0f} vectors/sec")
        print(f"  CUDA optimized:          {cuda_tp:>8.0f} vectors/sec")
        print(f"  Speedup vs quantize():   {speedup_quantize:>5.2f}x")
        print(f"  Speedup vs encode+decode: {speedup_encode_decode:>5.2f}x")
        print()


if __name__ == "__main__":
    main()
