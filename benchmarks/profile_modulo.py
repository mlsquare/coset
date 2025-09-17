"""
Profiler for modulo arithmetic operations in HNLQ encoding space.

This profiler compares PyTorch CPU baseline vs CUDA-accelerated implementations
for MAC and A&A operations in the encoding domain.
"""

import torch
import time
import json
import os
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import coset modules
from coset.quant import QuantizationConfig, mac_encoding_space, accumulate_encoding_space
from coset.quant.lut import LUTManager, build_lut
from coset.quant.modulo import CarryAwareAccumulator, mac_with_dither, mac_with_scaling, adaptive_mac
from coset.lattices import D4Lattice, E8Lattice
from coset.cuda.kernels import is_cuda_available


class ModuloProfiler:
    """Profiler for modulo arithmetic operations."""
    
    def __init__(self, lattice_name: str = "D4", q: int = 4, M: int = 2):
        """
        Initialize the profiler.
        
        Args:
            lattice_name: Name of the lattice ("D4" or "E8")
            q: Quantization parameter
            M: Number of layers
        """
        self.lattice_name = lattice_name
        self.q = q
        self.M = M
        
        # Initialize lattice and config
        if lattice_name == "D4":
            self.lattice = D4Lattice()
        elif lattice_name == "E8":
            self.lattice = E8Lattice()
        else:
            raise ValueError(f"Unsupported lattice: {lattice_name}")
            
        self.config = QuantizationConfig(q=q, M=M)
        self.cuda_available = is_cuda_available()
        
        # Initialize LUT manager
        self.lut_manager = LUTManager(self.lattice, self.config)
        
        print(f"Modulo Profiler initialized:")
        print(f"  Lattice: {lattice_name}")
        print(f"  q: {q}, M: {M}")
        print(f"  CUDA available: {self.cuda_available}")
        print(f"  Lattice dimension: {self.lattice.d}")
        print(f"  LUT size: {self.lut_manager.get_lut_size()}")
    
    def generate_test_data(self, batch_size: int, device: torch.device = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate test encodings for profiling.
        
        Args:
            batch_size: Number of vectors to generate
            device: Device to store tensors on
            
        Returns:
            Tuple of (encodings_x, encodings_y) lists
        """
        if device is None:
            device = torch.device('cpu')
            
        # Generate random vectors
        x = torch.randn(batch_size, self.lattice.d, device=device)
        y = torch.randn(batch_size, self.lattice.d, device=device)
        
        # Encode vectors
        from coset.quant.functional import encode
        
        encodings_x = []
        encodings_y = []
        
        for i in range(self.M):
            # For now, create dummy encodings
            # In real implementation, these would come from encode()
            encoding_x = torch.randint(0, self.q, (batch_size, self.lattice.d), 
                                     dtype=torch.int8, device=device)
            encoding_y = torch.randint(0, self.q, (batch_size, self.lattice.d), 
                                     dtype=torch.int8, device=device)
            
            encodings_x.append(encoding_x)
            encodings_y.append(encoding_y)
        
        return encodings_x, encodings_y
    
    def pytorch_mac(self, encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor]) -> torch.Tensor:
        """PyTorch CPU baseline MAC operation."""
        return mac_encoding_space(encodings_x, encodings_y, self.lattice, self.config)
    
    def cuda_mac(self, encodings_x: List[torch.Tensor], encodings_y: List[torch.Tensor]) -> torch.Tensor:
        """CUDA-accelerated MAC operation."""
        if not self.cuda_available:
            return self.pytorch_mac(encodings_x, encodings_y)
        
        try:
            # Move to GPU if not already there
            encodings_x_gpu = [enc.to('cuda') for enc in encodings_x]
            encodings_y_gpu = [enc.to('cuda') for enc in encodings_y]
            
            return mac_encoding_space(encodings_x_gpu, encodings_y_gpu, self.lattice, self.config)
        except Exception as e:
            print(f"CUDA MAC failed, falling back to CPU: {e}")
            return self.pytorch_mac(encodings_x, encodings_y)
    
    def pytorch_accumulate(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """PyTorch CPU baseline A&A operation."""
        return accumulate_encoding_space(encodings, self.lattice, self.config)
    
    def cuda_accumulate(self, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        """CUDA-accelerated A&A operation."""
        if not self.cuda_available:
            return self.pytorch_accumulate(encodings)
        
        try:
            # Move to GPU if not already there
            encodings_gpu = [enc.to('cuda') for enc in encodings]
            
            return accumulate_encoding_space(encodings_gpu, self.lattice, self.config)
        except Exception as e:
            print(f"CUDA accumulate failed, falling back to CPU: {e}")
            return self.pytorch_accumulate(encodings)
    
    def profile_mac(self, batch_sizes: List[int], num_runs: int = 10) -> Dict[str, Any]:
        """
        Profile MAC operations across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with profiling results
        """
        print(f"\nProfiling MAC operations with {num_runs} runs each...")
        
        results = {
            'batch_sizes': batch_sizes,
            'pytorch_times': [],
            'cuda_times': [],
            'pytorch_throughput': [],
            'cuda_throughput': [],
            'speedup': []
        }
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Generate test data
            encodings_x, encodings_y = self.generate_test_data(batch_size)
            
            # Profile PyTorch CPU
            pytorch_times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = self.pytorch_mac(encodings_x, encodings_y)
                end_time = time.time()
                pytorch_times.append(end_time - start_time)
            
            # Profile CUDA
            cuda_times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = self.cuda_mac(encodings_x, encodings_y)
                end_time = time.time()
                cuda_times.append(end_time - start_time)
            
            # Calculate statistics
            pytorch_mean = np.mean(pytorch_times)
            cuda_mean = np.mean(cuda_times)
            
            pytorch_throughput = batch_size / pytorch_mean
            cuda_throughput = batch_size / cuda_mean
            speedup = pytorch_mean / cuda_mean if cuda_mean > 0 else 0
            
            results['pytorch_times'].append(pytorch_mean)
            results['cuda_times'].append(cuda_mean)
            results['pytorch_throughput'].append(pytorch_throughput)
            results['cuda_throughput'].append(cuda_throughput)
            results['speedup'].append(speedup)
            
            print(f"    PyTorch: {pytorch_mean:.6f}s ({pytorch_throughput:.0f} ops/s)")
            print(f"    CUDA:    {cuda_mean:.6f}s ({cuda_throughput:.0f} ops/s)")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def profile_accumulate(self, batch_sizes: List[int], num_runs: int = 10) -> Dict[str, Any]:
        """
        Profile A&A operations across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with profiling results
        """
        print(f"\nProfiling A&A operations with {num_runs} runs each...")
        
        results = {
            'batch_sizes': batch_sizes,
            'pytorch_times': [],
            'cuda_times': [],
            'pytorch_throughput': [],
            'cuda_throughput': [],
            'speedup': []
        }
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Generate test data
            encodings_x, _ = self.generate_test_data(batch_size)
            
            # Profile PyTorch CPU
            pytorch_times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = self.pytorch_accumulate(encodings_x)
                end_time = time.time()
                pytorch_times.append(end_time - start_time)
            
            # Profile CUDA
            cuda_times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = self.cuda_accumulate(encodings_x)
                end_time = time.time()
                cuda_times.append(end_time - start_time)
            
            # Calculate statistics
            pytorch_mean = np.mean(pytorch_times)
            cuda_mean = np.mean(cuda_times)
            
            pytorch_throughput = batch_size / pytorch_mean
            cuda_throughput = batch_size / cuda_mean
            speedup = pytorch_mean / cuda_mean if cuda_mean > 0 else 0
            
            results['pytorch_times'].append(pytorch_mean)
            results['cuda_times'].append(cuda_mean)
            results['pytorch_throughput'].append(pytorch_throughput)
            results['cuda_throughput'].append(cuda_throughput)
            results['speedup'].append(speedup)
            
            print(f"    PyTorch: {pytorch_mean:.6f}s ({pytorch_throughput:.0f} ops/s)")
            print(f"    CUDA:    {cuda_mean:.6f}s ({cuda_throughput:.0f} ops/s)")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def profile_lut_operations(self, batch_sizes: List[int], num_runs: int = 10) -> Dict[str, Any]:
        """
        Profile LUT operations across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with profiling results
        """
        print(f"\nProfiling LUT operations with {num_runs} runs each...")
        
        results = {
            'batch_sizes': batch_sizes,
            'build_times': [],
            'lookup_times': [],
            'build_throughput': [],
            'lookup_throughput': []
        }
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Profile LUT building
            build_times = []
            for _ in range(num_runs):
                start_time = time.time()
                lut = self.lut_manager.build_two_sided_lut()
                end_time = time.time()
                build_times.append(end_time - start_time)
            
            # Profile LUT lookup
            encodings_x, encodings_y = self.generate_test_data(batch_size)
            lookup_times = []
            for _ in range(num_runs):
                start_time = time.time()
                # Simulate LUT lookup
                result = self.pytorch_mac(encodings_x, encodings_y)
                end_time = time.time()
                lookup_times.append(end_time - start_time)
            
            # Calculate statistics
            build_mean = np.mean(build_times)
            lookup_mean = np.mean(lookup_times)
            
            build_throughput = 1 / build_mean  # LUTs per second
            lookup_throughput = batch_size / lookup_mean
            
            results['build_times'].append(build_mean)
            results['lookup_times'].append(lookup_mean)
            results['build_throughput'].append(build_throughput)
            results['lookup_throughput'].append(lookup_throughput)
            
            print(f"    Build:  {build_mean:.6f}s ({build_throughput:.2f} LUTs/s)")
            print(f"    Lookup: {lookup_mean:.6f}s ({lookup_throughput:.0f} ops/s)")
        
        return results
    
    def create_plots(self, mac_results: Dict[str, Any], accumulate_results: Dict[str, Any], 
                    lut_results: Dict[str, Any], output_dir: str = "benchmarks/results"):
        """Create performance plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Modulo Arithmetic Performance - {self.lattice_name} (q={self.q}, M={self.M})', 
                    fontsize=16, fontweight='bold')
        
        # MAC Performance
        ax1 = axes[0, 0]
        ax1.plot(mac_results['batch_sizes'], mac_results['pytorch_times'], 'o-', label='PyTorch CPU', linewidth=2)
        ax1.plot(mac_results['batch_sizes'], mac_results['cuda_times'], 's-', label='CUDA', linewidth=2)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('MAC Operation Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # MAC Throughput
        ax2 = axes[0, 1]
        ax2.plot(mac_results['batch_sizes'], mac_results['pytorch_throughput'], 'o-', label='PyTorch CPU', linewidth=2)
        ax2.plot(mac_results['batch_sizes'], mac_results['cuda_throughput'], 's-', label='CUDA', linewidth=2)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (ops/sec)')
        ax2.set_title('MAC Operation Throughput')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # A&A Performance
        ax3 = axes[1, 0]
        ax3.plot(accumulate_results['batch_sizes'], accumulate_results['pytorch_times'], 'o-', label='PyTorch CPU', linewidth=2)
        ax3.plot(accumulate_results['batch_sizes'], accumulate_results['cuda_times'], 's-', label='CUDA', linewidth=2)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('A&A Operation Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Speedup
        ax4 = axes[1, 1]
        ax4.plot(mac_results['batch_sizes'], mac_results['speedup'], 'o-', label='MAC Speedup', linewidth=2, color='red')
        ax4.plot(accumulate_results['batch_sizes'], accumulate_results['speedup'], 's-', label='A&A Speedup', linewidth=2, color='blue')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Speedup (x)')
        ax4.set_title('CUDA Speedup vs PyTorch CPU')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'modulo_performance_{self.lattice_name}_q{self.q}_M{self.M}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to: {plot_path}")
    
    def save_results(self, mac_results: Dict[str, Any], accumulate_results: Dict[str, Any], 
                    lut_results: Dict[str, Any], output_dir: str = "benchmarks/results"):
        """Save profiling results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        all_results = {
            'lattice': self.lattice_name,
            'q': self.q,
            'M': self.M,
            'cuda_available': self.cuda_available,
            'mac_results': mac_results,
            'accumulate_results': accumulate_results,
            'lut_results': lut_results
        }
        
        json_path = os.path.join(output_dir, f'modulo_results_{self.lattice_name}_q{self.q}_M{self.M}.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, f'modulo_summary_{self.lattice_name}_q{self.q}_M{self.M}.csv')
        with open(csv_path, 'w') as f:
            f.write("Operation,Batch Size,PyTorch Time (s),CUDA Time (s),PyTorch Throughput (ops/s),CUDA Throughput (ops/s),Speedup\n")
            
            for i, batch_size in enumerate(mac_results['batch_sizes']):
                f.write(f"MAC,{batch_size},{mac_results['pytorch_times'][i]:.6f},{mac_results['cuda_times'][i]:.6f},"
                       f"{mac_results['pytorch_throughput'][i]:.0f},{mac_results['cuda_throughput'][i]:.0f},"
                       f"{mac_results['speedup'][i]:.2f}\n")
            
            for i, batch_size in enumerate(accumulate_results['batch_sizes']):
                f.write(f"A&A,{batch_size},{accumulate_results['pytorch_times'][i]:.6f},{accumulate_results['cuda_times'][i]:.6f},"
                       f"{accumulate_results['pytorch_throughput'][i]:.0f},{accumulate_results['cuda_throughput'][i]:.0f},"
                       f"{accumulate_results['speedup'][i]:.2f}\n")
        
        print(f"Results saved to: {json_path}")
        print(f"Summary saved to: {csv_path}")


def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description='Profile modulo arithmetic operations')
    parser.add_argument('--lattice', type=str, default='D4', choices=['D4', 'E8'], help='Lattice type')
    parser.add_argument('--q', type=int, default=4, help='Quantization parameter')
    parser.add_argument('--M', type=int, default=2, help='Number of layers')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[10, 50, 100, 500, 1000], help='Batch sizes to test')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs for averaging')
    parser.add_argument('--output-dir', type=str, default='benchmarks/results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = ModuloProfiler(lattice_name=args.lattice, q=args.q, M=args.M)
    
    # Profile operations
    mac_results = profiler.profile_mac(args.batch_sizes, args.num_runs)
    accumulate_results = profiler.profile_accumulate(args.batch_sizes, args.num_runs)
    lut_results = profiler.profile_lut_operations(args.batch_sizes, args.num_runs)
    
    # Create plots and save results
    profiler.create_plots(mac_results, accumulate_results, lut_results, args.output_dir)
    profiler.save_results(mac_results, accumulate_results, lut_results, args.output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MODULO ARITHMETIC PROFILING SUMMARY")
    print(f"{'='*60}")
    print(f"Lattice: {args.lattice}")
    print(f"Parameters: q={args.q}, M={args.M}")
    print(f"CUDA Available: {profiler.cuda_available}")
    print(f"LUT Size: {profiler.lut_manager.get_lut_size()}")
    
    # Find best performance
    max_mac_speedup = max(mac_results['speedup']) if mac_results['speedup'] else 0
    max_accumulate_speedup = max(accumulate_results['speedup']) if accumulate_results['speedup'] else 0
    
    print(f"\nPeak Performance:")
    print(f"  MAC Speedup: {max_mac_speedup:.2f}x")
    print(f"  A&A Speedup: {max_accumulate_speedup:.2f}x")
    
    if profiler.cuda_available:
        max_mac_throughput = max(mac_results['cuda_throughput']) if mac_results['cuda_throughput'] else 0
        max_accumulate_throughput = max(accumulate_results['cuda_throughput']) if accumulate_results['cuda_throughput'] else 0
        
        print(f"  Peak MAC Throughput: {max_mac_throughput:.0f} ops/s")
        print(f"  Peak A&A Throughput: {max_accumulate_throughput:.0f} ops/s")


if __name__ == "__main__":
    main()
