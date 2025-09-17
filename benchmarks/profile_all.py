#!/usr/bin/env python3
"""
Master Performance Profiler

This script runs all three profiling processes (encoding, decoding, combined) and creates
a comprehensive performance comparison report between baseline PyTorch and CUDA-optimized versions.

Usage:
    python benchmarks/profile_all.py [--lattice D4] [--q 4] [--M 2] [--batch-sizes 100,1000,10000]
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import numpy as np


class MasterProfiler:
    """Master profiler that runs all three profiling processes and creates comprehensive reports."""
    
    def __init__(self, lattice_type: str = "D4", q: int = 4, M: int = 2, output_dir: str = "benchmarks/results"):
        """
        Initialize the master profiler.
        
        Args:
            lattice_type: Type of lattice ("Z2", "D4", "E8")
            q: Quantization parameter
            M: Number of hierarchical levels
            output_dir: Output directory for results
        """
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to the individual profilers
        self.profiler_scripts = {
            'encoding': Path(__file__).parent / 'profile_encoding.py',
            'decoding': Path(__file__).parent / 'profile_decoding.py',
            'combined': Path(__file__).parent / 'profile_combined.py'
        }
    
    def run_profiler(self, profiler_name: str, batch_sizes: List[int]) -> bool:
        """
        Run a specific profiler.
        
        Args:
            profiler_name: Name of the profiler ('encoding', 'decoding', 'combined')
            batch_sizes: List of batch sizes to test
            
        Returns:
            True if successful, False otherwise
        """
        if profiler_name not in self.profiler_scripts:
            print(f"Unknown profiler: {profiler_name}")
            return False
        
        script_path = self.profiler_scripts[profiler_name]
        batch_sizes_str = ','.join(map(str, batch_sizes))
        
        cmd = [
            sys.executable, str(script_path),
            '--lattice', self.lattice_type,
            '--q', str(self.q),
            '--M', str(self.M),
            '--batch-sizes', batch_sizes_str,
            '--output-dir', str(self.output_dir),
            '--no-plot'  # We'll create combined plots later
        ]
        
        print(f"Running {profiler_name} profiler...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ {profiler_name} profiler completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ {profiler_name} profiler failed:")
            print(f"  Return code: {e.returncode}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            return False
    
    def run_all_profilers(self, batch_sizes: List[int]) -> Dict[str, bool]:
        """
        Run all three profilers.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping profiler names to success status
        """
        results = {}
        
        for profiler_name in ['encoding', 'decoding', 'combined']:
            results[profiler_name] = self.run_profiler(profiler_name, batch_sizes)
        
        return results
    
    def load_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load results from all profilers.
        
        Returns:
            Dictionary mapping profiler names to their results
        """
        results = {}
        
        for profiler_name in ['encoding', 'decoding', 'combined']:
            json_file = self.output_dir / f'{profiler_name}_results_{self.lattice_type}_q{self.q}_M{self.M}.json'
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        results[profiler_name] = json.load(f)
                    print(f"✓ Loaded {profiler_name} results from {json_file}")
                except Exception as e:
                    print(f"✗ Failed to load {profiler_name} results: {e}")
                    results[profiler_name] = []
            else:
                print(f"✗ Results file not found: {json_file}")
                results[profiler_name] = []
        
        return results
    
    def create_comprehensive_plot(self, results: Dict[str, List[Dict[str, Any]]]):
        """Create a comprehensive plot comparing all three processes."""
        if not any(results.values()):
            print("No results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Colors for different profilers
        colors = {'encoding': 'blue', 'decoding': 'green', 'combined': 'red'}
        markers = {'encoding': 'o', 'decoding': 's', 'combined': '^'}
        
        # Plot 1: Execution time comparison
        for profiler_name, profiler_results in results.items():
            if not profiler_results:
                continue
            
            batch_sizes = [r['batch_size'] for r in profiler_results]
            
            # Handle different result structures
            if profiler_name == 'combined':
                # Combined profiler has baseline_quantize and baseline_encode_decode
                baseline_times = [r['baseline_quantize']['mean_time'] * 1000 for r in profiler_results]
            else:
                # Encoding and decoding profilers have baseline
                baseline_times = [r['baseline']['mean_time'] * 1000 for r in profiler_results]
            
            cuda_times = [r['cuda_optimized']['mean_time'] * 1000 for r in profiler_results]
            
            ax1.loglog(batch_sizes, baseline_times, 
                      marker=markers[profiler_name], linestyle='-', 
                      color=colors[profiler_name], alpha=0.7, linewidth=2, markersize=6,
                      label=f'{profiler_name.capitalize()} - Baseline')
            ax1.loglog(batch_sizes, cuda_times, 
                      marker=markers[profiler_name], linestyle='--', 
                      color=colors[profiler_name], linewidth=2, markersize=6,
                      label=f'{profiler_name.capitalize()} - CUDA')
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'Execution Time Comparison\n({self.lattice_type}, q={self.q}, M={self.M})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup comparison
        for profiler_name, profiler_results in results.items():
            if not profiler_results:
                continue
            
            batch_sizes = [r['batch_size'] for r in profiler_results]
            
            # Handle different result structures
            if profiler_name == 'combined':
                # Combined profiler has speedup_vs_quantize and speedup_vs_encode_decode
                speedups = [r['speedup_vs_quantize'] for r in profiler_results]
            else:
                # Encoding and decoding profilers have speedup
                speedups = [r['speedup'] for r in profiler_results]
            
            ax2.semilogx(batch_sizes, speedups, 
                        marker=markers[profiler_name], linestyle='-', 
                        color=colors[profiler_name], linewidth=2, markersize=8,
                        label=f'{profiler_name.capitalize()}')
        
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('CUDA Speedup Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput comparison
        for profiler_name, profiler_results in results.items():
            if not profiler_results:
                continue
            
            batch_sizes = [r['batch_size'] for r in profiler_results]
            
            # Handle different result structures
            if profiler_name == 'combined':
                # Combined profiler has baseline_quantize and baseline_encode_decode
                baseline_throughput = [r['baseline_quantize']['throughput'] for r in profiler_results]
            else:
                # Encoding and decoding profilers have baseline
                baseline_throughput = [r['baseline']['throughput'] for r in profiler_results]
            
            cuda_throughput = [r['cuda_optimized']['throughput'] for r in profiler_results]
            
            ax3.loglog(batch_sizes, baseline_throughput, 
                      marker=markers[profiler_name], linestyle='-', 
                      color=colors[profiler_name], alpha=0.7, linewidth=2, markersize=6,
                      label=f'{profiler_name.capitalize()} - Baseline')
            ax3.loglog(batch_sizes, cuda_throughput, 
                      marker=markers[profiler_name], linestyle='--', 
                      color=colors[profiler_name], linewidth=2, markersize=6,
                      label=f'{profiler_name.capitalize()} - CUDA')
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Throughput (vectors/sec)')
        ax3.set_title('Throughput Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance summary (bar chart)
        profiler_names = []
        avg_speedups = []
        
        for profiler_name, profiler_results in results.items():
            if not profiler_results:
                continue
            
            profiler_names.append(profiler_name.capitalize())
            
            # Handle different result structures
            if profiler_name == 'combined':
                # Combined profiler has speedup_vs_quantize and speedup_vs_encode_decode
                avg_speedup = np.mean([r['speedup_vs_quantize'] for r in profiler_results])
            else:
                # Encoding and decoding profilers have speedup
                avg_speedup = np.mean([r['speedup'] for r in profiler_results])
            
            avg_speedups.append(avg_speedup)
        
        if profiler_names:
            bars = ax4.bar(profiler_names, avg_speedups, 
                          color=[colors[name.lower()] for name in profiler_names], alpha=0.7)
            ax4.set_ylabel('Average Speedup (x)')
            ax4.set_title('Average CUDA Speedup by Process')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, speedup in zip(bars, avg_speedups):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = self.output_dir / f'comprehensive_profile_{self.lattice_type}_q{self.q}_M{self.M}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved to: {plot_path}")
        
        plt.show()
    
    def create_summary_report(self, results: Dict[str, List[Dict[str, Any]]]):
        """Create a comprehensive summary report."""
        report_path = self.output_dir / f'summary_report_{self.lattice_type}_q{self.q}_M{self.M}.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Performance Profiling Summary Report\n\n")
            f.write(f"**Configuration:** {self.lattice_type} lattice, q={self.q}, M={self.M}\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report compares the performance of baseline PyTorch implementations against CUDA-optimized versions for three key processes:\n\n")
            f.write("1. **Encoding**: Converting vectors to hierarchical lattice encodings\n")
            f.write("2. **Decoding**: Converting encodings back to vectors\n")
            f.write("3. **Combined**: Complete quantization process (encode + decode)\n\n")
            
            f.write("## Results Summary\n\n")
            
            for profiler_name, profiler_results in results.items():
                if not profiler_results:
                    f.write(f"### {profiler_name.capitalize()}\n\n")
                    f.write("No results available.\n\n")
                    continue
                
                f.write(f"### {profiler_name.capitalize()}\n\n")
                
                # Calculate summary statistics
                batch_sizes = [r['batch_size'] for r in profiler_results]
                
                # Handle different result structures
                if profiler_name == 'combined':
                    # Combined profiler has speedup_vs_quantize and speedup_vs_encode_decode
                    speedups = [r['speedup_vs_quantize'] for r in profiler_results]
                else:
                    # Encoding and decoding profilers have speedup
                    speedups = [r['speedup'] for r in profiler_results]
                
                max_throughput = max([r['cuda_optimized']['throughput'] for r in profiler_results])
                avg_speedup = np.mean(speedups)
                max_speedup = max(speedups)
                
                f.write(f"- **Batch sizes tested:** {batch_sizes}\n")
                f.write(f"- **Average speedup:** {avg_speedup:.2f}x\n")
                f.write(f"- **Maximum speedup:** {max_speedup:.2f}x\n")
                f.write(f"- **Peak throughput:** {max_throughput:.0f} vectors/sec\n\n")
                
                # Detailed results table
                f.write("| Batch Size | Baseline Time (ms) | CUDA Time (ms) | Speedup | Throughput (vec/s) |\n")
                f.write("|------------|-------------------|----------------|---------|-------------------|\n")
                
                for result in profiler_results:
                    bs = result['batch_size']
                    
                    # Handle different result structures
                    if profiler_name == 'combined':
                        baseline_time = result['baseline_quantize']['mean_time'] * 1000
                        speedup = result['speedup_vs_quantize']
                    else:
                        baseline_time = result['baseline']['mean_time'] * 1000
                        speedup = result['speedup']
                    
                    cuda_time = result['cuda_optimized']['mean_time'] * 1000
                    throughput = result['cuda_optimized']['throughput']
                    
                    f.write(f"| {bs:>10} | {baseline_time:>17.2f} | {cuda_time:>14.2f} | {speedup:>7.2f} | {throughput:>17.0f} |\n")
                
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            # Analyze results
            all_speedups = []
            all_throughputs = []
            
            for profiler_name, profiler_results in results.items():
                if profiler_results:
                    # Handle different result structures
                    if profiler_name == 'combined':
                        all_speedups.extend([r['speedup_vs_quantize'] for r in profiler_results])
                    else:
                        all_speedups.extend([r['speedup'] for r in profiler_results])
                    
                    all_throughputs.extend([r['cuda_optimized']['throughput'] for r in profiler_results])
            
            if all_speedups:
                f.write(f"- **Overall average speedup:** {np.mean(all_speedups):.2f}x\n")
                f.write(f"- **Best speedup achieved:** {max(all_speedups):.2f}x\n")
                f.write(f"- **Peak throughput achieved:** {max(all_throughputs):.0f} vectors/sec\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the profiling results:\n\n")
            
            if all_speedups:
                if np.mean(all_speedups) > 2.0:
                    f.write("- ✅ CUDA optimization shows significant performance gains\n")
                else:
                    f.write("- ⚠️ CUDA optimization shows modest performance gains\n")
                
                if max(all_throughputs) > 100000:
                    f.write("- ✅ Peak throughput meets target performance (>100K vectors/sec)\n")
                else:
                    f.write("- ⚠️ Peak throughput below target performance\n")
            
            f.write("- Consider further optimization for larger batch sizes\n")
            f.write("- Monitor memory usage and bandwidth utilization\n")
            f.write("- Validate numerical accuracy of CUDA implementations\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `comprehensive_profile_*.png`: Combined performance plots\n")
            f.write("- `*_results_*.json`: Detailed timing data\n")
            f.write("- `*_summary_*.csv`: Summary statistics\n")
            f.write("- `summary_report_*.md`: This report\n\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_complete_analysis(self, batch_sizes: List[int]):
        """Run complete profiling analysis."""
        print("="*60)
        print("MASTER PERFORMANCE PROFILER")
        print("="*60)
        print(f"Lattice: {self.lattice_type}, q={self.q}, M={self.M}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Run all profilers
        print("Step 1: Running all profilers...")
        profiler_results = self.run_all_profilers(batch_sizes)
        
        successful_profilers = [name for name, success in profiler_results.items() if success]
        failed_profilers = [name for name, success in profiler_results.items() if not success]
        
        if successful_profilers:
            print(f"✓ Successful profilers: {', '.join(successful_profilers)}")
        if failed_profilers:
            print(f"✗ Failed profilers: {', '.join(failed_profilers)}")
        
        print()
        
        # Load results
        print("Step 2: Loading results...")
        results = self.load_results()
        
        if not any(results.values()):
            print("No results available for analysis. Exiting.")
            return
        
        # Create comprehensive plot
        print("Step 3: Creating comprehensive plots...")
        self.create_comprehensive_plot(results)
        
        # Create summary report
        print("Step 4: Creating summary report...")
        self.create_summary_report(results)
        
        print()
        print("="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
        print("Check the summary report and plots for detailed analysis.")


def main():
    """Main function for the master profiler."""
    parser = argparse.ArgumentParser(description='Run comprehensive performance profiling')
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
    
    # Create master profiler and run analysis
    profiler = MasterProfiler(
        lattice_type=args.lattice,
        q=args.q,
        M=args.M,
        output_dir=args.output_dir
    )
    
    profiler.run_complete_analysis(batch_sizes)


if __name__ == "__main__":
    main()
