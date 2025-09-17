#!/usr/bin/env python3
"""
Quantization Error Distribution Profiler

This script profiles the quantization error distribution for row-normalized matrices
using E8 lattice quantization. It tiles the matrix to match the lattice dimension,
quantizes each row, and analyzes the distance between original and quantized vectors.

Usage:
    python benchmarks/profile_quantization_error.py [--batch-size 1000] [--matrix-size 512] [--lattice E8] [--q 4] [--M 2]
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


class QuantizationErrorProfiler:
    """Profiler for analyzing quantization error distributions."""
    
    def __init__(self, lattice_type: str = "E8", q: int = 4, M: int = 2):
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
    
    def generate_test_matrix(self, batch_size: int, matrix_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a row-normalized test matrix.
        
        Args:
            batch_size: Number of rows (B)
            matrix_size: Number of columns (N)
            device: Device to create tensor on
            
        Returns:
            Row-normalized matrix of shape [batch_size, matrix_size]
        """
        # Generate random matrix
        matrix = torch.randn(batch_size, matrix_size, device=device, dtype=torch.float32)
        
        # Row normalize (L2 norm)
        row_norms = torch.norm(matrix, dim=1, keepdim=True)
        matrix = matrix / (row_norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        return matrix
    
    def tile_matrix(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Tile the matrix to match the lattice dimension.
        
        Args:
            matrix: Input matrix of shape [B, N] (row-normalized)
            
        Returns:
            Tuple of (tiled_matrix, tiling_info) where:
            - tiled_matrix: Matrix tiled to shape [B * num_tiles, d]
            - tiling_info: Dictionary with tiling metadata
        """
        B, N = matrix.shape
        d = self.lattice.d
        
        # Calculate number of tiles needed
        num_tiles = (N + d - 1) // d  # Ceiling division
        
        # Scale the row-normalized matrix by number of tiles
        matrix_scaled = matrix * num_tiles
        
        # Pad the matrix if necessary
        if N % d != 0:
            pad_size = num_tiles * d - N
            padding = torch.zeros(B, pad_size, device=matrix.device, dtype=matrix.dtype)
            matrix_padded = torch.cat([matrix_scaled, padding], dim=1)
        else:
            matrix_padded = matrix_scaled
        
        # Reshape to [B, num_tiles, d]
        matrix_reshaped = matrix_padded.view(B, num_tiles, d)
        
        # Reshape to [B * num_tiles, d] for batch processing
        tiled_matrix = matrix_reshaped.view(B * num_tiles, d)
        
        tiling_info = {
            'original_shape': (B, N),
            'lattice_dim': d,
            'num_tiles': num_tiles,
            'tiled_shape': (B * num_tiles, d),
            'padded_size': matrix_padded.shape[1] if N % d != 0 else N,
            'scaling_factor': num_tiles
        }
        
        return tiled_matrix, tiling_info
    
    def quantize_tiled_matrix(self, tiled_matrix: torch.Tensor) -> torch.Tensor:
        """
        Quantize the tiled matrix using the specified lattice.
        
        Args:
            tiled_matrix: Tiled matrix of shape [B * num_tiles, d]
            
        Returns:
            Quantized matrix of shape [B * num_tiles, d]
        """
        if self.cuda_available:
            try:
                # Use CUDA kernel for quantization
                return cuda_quantize(tiled_matrix, self.lattice, self.config)
            except Exception as e:
                print(f"CUDA quantization failed, falling back to baseline: {e}")
                return self._baseline_quantize(tiled_matrix)
        else:
            return self._baseline_quantize(tiled_matrix)
    
    def _baseline_quantize(self, tiled_matrix: torch.Tensor) -> torch.Tensor:
        """Baseline quantization using PyTorch implementation."""
        batch_size = tiled_matrix.shape[0]
        quantized = []
        
        for i in range(batch_size):
            x_hat = quantize(tiled_matrix[i], self.lattice, self.config)
            quantized.append(x_hat)
        
        return torch.stack(quantized)
    
    def compute_distances(self, original: torch.Tensor, quantized: torch.Tensor, 
                         tiling_info: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L2 distances between original and quantized vectors.
        
        Args:
            original: Original tiled vectors of shape [B * num_tiles, d]
            quantized: Quantized tiled vectors of shape [B * num_tiles, d]
            tiling_info: Tiling metadata
            
        Returns:
            Tuple of (tile_distances, row_distances) where:
            - tile_distances: L2 distances per tile [B * num_tiles]
            - row_distances: L2 distances per original row [B]
        """
        # Compute distances per tile
        tile_distances = torch.norm(original - quantized, dim=1)
        
        # Reconstruct original matrix shapes for row-wise comparison
        B, N = tiling_info['original_shape']
        d = tiling_info['lattice_dim']
        num_tiles = tiling_info['num_tiles']
        scaling_factor = tiling_info['scaling_factor']
        
        # Reshape back to [B, num_tiles, d]
        original_reshaped = original.view(B, num_tiles, d)
        quantized_reshaped = quantized.view(B, num_tiles, d)
        
        # Reconstruct original matrix shapes [B, N]
        original_matrix = original_reshaped.view(B, num_tiles * d)[:, :N]
        quantized_matrix = quantized_reshaped.view(B, num_tiles * d)[:, :N]
        
        # Scale back to original row-normalized scale for comparison
        original_matrix_scaled_back = original_matrix / scaling_factor
        quantized_matrix_scaled_back = quantized_matrix / scaling_factor
        
        # Compute distances per original row
        row_distances = torch.norm(original_matrix_scaled_back - quantized_matrix_scaled_back, dim=1)
        
        return tile_distances, row_distances
    
    def analyze_error_distribution(self, tile_distances: torch.Tensor, row_distances: torch.Tensor, 
                                 tiling_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the error distribution and compute statistics.
        
        Args:
            tile_distances: L2 distances per tile [B * num_tiles]
            row_distances: L2 distances per original row [B]
            tiling_info: Tiling metadata
            
        Returns:
            Dictionary with error distribution statistics
        """
        tile_distances_cpu = tile_distances.cpu().numpy()
        row_distances_cpu = row_distances.cpu().numpy()
        
        def compute_stats(distances):
            return {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'median': float(np.median(distances)),
                'q25': float(np.percentile(distances, 25)),
                'q75': float(np.percentile(distances, 75)),
                'q90': float(np.percentile(distances, 90)),
                'q95': float(np.percentile(distances, 95)),
                'q99': float(np.percentile(distances, 99)),
                'num_samples': len(distances)
            }
        
        stats = {
            'tile_distances': compute_stats(tile_distances_cpu),
            'row_distances': compute_stats(row_distances_cpu),
            'tiling_info': tiling_info
        }
        
        return stats
    
    def create_histogram(self, tile_distances: torch.Tensor, row_distances: torch.Tensor, 
                        output_dir: Path, batch_size: int, matrix_size: int) -> None:
        """
        Create and save histogram of quantization errors.
        
        Args:
            tile_distances: L2 distances per tile
            row_distances: L2 distances per original row
            output_dir: Output directory for plots
            batch_size: Batch size used
            matrix_size: Matrix size used
        """
        tile_distances_cpu = tile_distances.cpu().numpy()
        row_distances_cpu = row_distances.cpu().numpy()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Tile distances histogram
        ax1.hist(tile_distances_cpu, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('L2 Distance (Original - Quantized)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Tile-wise Quantization Error Distribution\n({self.lattice_type}, q={self.q}, M={self.M})')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text for tiles
        stats_text = f'Mean: {np.mean(tile_distances_cpu):.4f}\n'
        stats_text += f'Std: {np.std(tile_distances_cpu):.4f}\n'
        stats_text += f'Min: {np.min(tile_distances_cpu):.4f}\n'
        stats_text += f'Max: {np.max(tile_distances_cpu):.4f}\n'
        stats_text += f'Median: {np.median(tile_distances_cpu):.4f}\n'
        stats_text += f'Samples: {len(tile_distances_cpu)}'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Row distances histogram
        ax2.hist(row_distances_cpu, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('L2 Distance (Original - Quantized)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Row-wise Quantization Error Distribution\n({self.lattice_type}, q={self.q}, M={self.M})')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text for rows
        stats_text = f'Mean: {np.mean(row_distances_cpu):.4f}\n'
        stats_text += f'Std: {np.std(row_distances_cpu):.4f}\n'
        stats_text += f'Min: {np.min(row_distances_cpu):.4f}\n'
        stats_text += f'Max: {np.max(row_distances_cpu):.4f}\n'
        stats_text += f'Median: {np.median(row_distances_cpu):.4f}\n'
        stats_text += f'Samples: {len(row_distances_cpu)}'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Tile distances cumulative distribution
        sorted_tile_distances = np.sort(tile_distances_cpu)
        tile_cumulative = np.arange(1, len(sorted_tile_distances) + 1) / len(sorted_tile_distances)
        
        ax3.plot(sorted_tile_distances, tile_cumulative, linewidth=2, color='blue')
        ax3.set_xlabel('L2 Distance (Original - Quantized)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution - Tile-wise Errors')
        ax3.grid(True, alpha=0.3)
        
        # Add percentile lines for tiles
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(tile_distances_cpu, p)
            ax3.axvline(value, color='gray', linestyle='--', alpha=0.7)
            ax3.text(value, p/100, f'{p}%', rotation=90, verticalalignment='bottom')
        
        # Plot 4: Row distances cumulative distribution
        sorted_row_distances = np.sort(row_distances_cpu)
        row_cumulative = np.arange(1, len(sorted_row_distances) + 1) / len(sorted_row_distances)
        
        ax4.plot(sorted_row_distances, row_cumulative, linewidth=2, color='red')
        ax4.set_xlabel('L2 Distance (Original - Quantized)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution - Row-wise Errors')
        ax4.grid(True, alpha=0.3)
        
        # Add percentile lines for rows
        for p in percentiles:
            value = np.percentile(row_distances_cpu, p)
            ax4.axvline(value, color='gray', linestyle='--', alpha=0.7)
            ax4.text(value, p/100, f'{p}%', rotation=90, verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_dir / f'quantization_error_histogram_{self.lattice_type}_q{self.q}_M{self.M}_B{batch_size}_N{matrix_size}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {plot_path}")
        
        plt.show()
    
    def profile_quantization_errors(self, batch_size: int, matrix_size: int, 
                                  output_dir: Path) -> Dict[str, Any]:
        """
        Main profiling function for quantization errors.
        
        Args:
            batch_size: Number of rows (B)
            matrix_size: Number of columns (N)
            output_dir: Output directory for results
            
        Returns:
            Dictionary with profiling results
        """
        print(f"Profiling quantization errors...")
        print(f"Matrix size: {batch_size} x {matrix_size}")
        print(f"Lattice: {self.lattice_type}, q={self.q}, M={self.M}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        # Generate test matrix
        print("Generating row-normalized test matrix...")
        matrix = self.generate_test_matrix(batch_size, matrix_size, self.device)
        print(f"Matrix shape: {matrix.shape}")
        print(f"Row norms (first 5): {torch.norm(matrix[:5], dim=1)}")
        
        # Tile matrix
        print("Tiling matrix to match lattice dimension...")
        tiled_matrix, tiling_info = self.tile_matrix(matrix)
        print(f"Tiled matrix shape: {tiled_matrix.shape}")
        print(f"Number of tiles per row: {tiling_info['num_tiles']}")
        print(f"Scaling factor applied: {tiling_info['scaling_factor']}")
        print(f"Row norms after scaling (first 5): {torch.norm(tiled_matrix.view(tiling_info['original_shape'][0], -1)[:5], dim=1)}")
        
        # Quantize
        print("Quantizing tiled matrix...")
        start_time = time.perf_counter()
        quantized_matrix = self.quantize_tiled_matrix(tiled_matrix)
        end_time = time.perf_counter()
        
        quantization_time = end_time - start_time
        print(f"Quantization time: {quantization_time*1000:.2f}ms")
        print(f"Throughput: {tiled_matrix.shape[0] / quantization_time:.0f} vectors/sec")
        
        # Compute distances
        print("Computing L2 distances...")
        tile_distances, row_distances = self.compute_distances(tiled_matrix, quantized_matrix, tiling_info)
        
        # Analyze distribution
        print("Analyzing error distribution...")
        stats = self.analyze_error_distribution(tile_distances, row_distances, tiling_info)
        
        # Print statistics
        print("\n" + "="*50)
        print("QUANTIZATION ERROR STATISTICS")
        print("="*50)
        
        print("TILE-WISE ERRORS:")
        tile_stats = stats['tile_distances']
        print(f"  Number of samples: {tile_stats['num_samples']}")
        print(f"  Mean distance: {tile_stats['mean']:.6f}")
        print(f"  Std distance: {tile_stats['std']:.6f}")
        print(f"  Min distance: {tile_stats['min']:.6f}")
        print(f"  Max distance: {tile_stats['max']:.6f}")
        print(f"  Median distance: {tile_stats['median']:.6f}")
        print(f"  Q25 distance: {tile_stats['q25']:.6f}")
        print(f"  Q75 distance: {tile_stats['q75']:.6f}")
        print(f"  Q90 distance: {tile_stats['q90']:.6f}")
        print(f"  Q95 distance: {tile_stats['q95']:.6f}")
        print(f"  Q99 distance: {tile_stats['q99']:.6f}")
        
        print("\nROW-WISE ERRORS:")
        row_stats = stats['row_distances']
        print(f"  Number of samples: {row_stats['num_samples']}")
        print(f"  Mean distance: {row_stats['mean']:.6f}")
        print(f"  Std distance: {row_stats['std']:.6f}")
        print(f"  Min distance: {row_stats['min']:.6f}")
        print(f"  Max distance: {row_stats['max']:.6f}")
        print(f"  Median distance: {row_stats['median']:.6f}")
        print(f"  Q25 distance: {row_stats['q25']:.6f}")
        print(f"  Q75 distance: {row_stats['q75']:.6f}")
        print(f"  Q90 distance: {row_stats['q90']:.6f}")
        print(f"  Q95 distance: {row_stats['q95']:.6f}")
        print(f"  Q99 distance: {row_stats['q99']:.6f}")
        
        # Create histogram
        print("\nCreating histogram...")
        self.create_histogram(tile_distances, row_distances, output_dir, batch_size, matrix_size)
        
        # Compile results
        results = {
            'lattice_type': self.lattice_type,
            'q': self.q,
            'M': self.M,
            'batch_size': batch_size,
            'matrix_size': matrix_size,
            'device': str(self.device),
            'cuda_available': self.cuda_available,
            'quantization_time': quantization_time,
            'throughput': tiled_matrix.shape[0] / quantization_time,
            'tiling_info': tiling_info,
            'error_stats': stats,
            'tile_distances': tile_distances.cpu().numpy().tolist(),
            'row_distances': row_distances.cpu().numpy().tolist()
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save profiling results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        import json
        results_path = output_dir / f'quantization_error_results_{self.lattice_type}_q{self.q}_M{self.M}_B{results["batch_size"]}_N{results["matrix_size"]}.json'
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
        # Save summary as CSV
        import csv
        csv_path = output_dir / f'quantization_error_summary_{self.lattice_type}_q{self.q}_M{self.M}_B{results["batch_size"]}_N{results["matrix_size"]}.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'lattice_type', 'q', 'M', 'batch_size', 'matrix_size', 'device', 'cuda_available',
                'quantization_time_ms', 'throughput_vectors_per_sec',
                'tile_mean_distance', 'tile_std_distance', 'tile_min_distance', 'tile_max_distance', 'tile_median_distance',
                'tile_q25_distance', 'tile_q75_distance', 'tile_q90_distance', 'tile_q95_distance', 'tile_q99_distance',
                'tile_num_samples', 'row_mean_distance', 'row_std_distance', 'row_min_distance', 'row_max_distance', 'row_median_distance',
                'row_q25_distance', 'row_q75_distance', 'row_q90_distance', 'row_q95_distance', 'row_q99_distance',
                'row_num_samples', 'num_tiles_per_row', 'lattice_dim'
            ])
            
            tile_stats = results['error_stats']['tile_distances']
            row_stats = results['error_stats']['row_distances']
            
            writer.writerow([
                results['lattice_type'],
                results['q'],
                results['M'],
                results['batch_size'],
                results['matrix_size'],
                results['device'],
                results['cuda_available'],
                results['quantization_time'] * 1000,
                results['throughput'],
                tile_stats['mean'],
                tile_stats['std'],
                tile_stats['min'],
                tile_stats['max'],
                tile_stats['median'],
                tile_stats['q25'],
                tile_stats['q75'],
                tile_stats['q90'],
                tile_stats['q95'],
                tile_stats['q99'],
                tile_stats['num_samples'],
                row_stats['mean'],
                row_stats['std'],
                row_stats['min'],
                row_stats['max'],
                row_stats['median'],
                row_stats['q25'],
                row_stats['q75'],
                row_stats['q90'],
                row_stats['q95'],
                row_stats['q99'],
                row_stats['num_samples'],
                results['tiling_info']['num_tiles'],
                results['tiling_info']['lattice_dim']
            ])
        
        print(f"Summary saved to: {csv_path}")


def main():
    """Main function for the quantization error profiler."""
    parser = argparse.ArgumentParser(description='Profile quantization error distributions')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of rows (B)')
    parser.add_argument('--matrix-size', type=int, default=512, help='Number of columns (N)')
    parser.add_argument('--lattice', type=str, default='E8', choices=['Z2', 'D4', 'E8'],
                        help='Lattice type to use')
    parser.add_argument('--q', type=int, default=4, help='Quantization parameter')
    parser.add_argument('--M', type=int, default=2, help='Number of hierarchical levels')
    parser.add_argument('--output-dir', type=str, default='benchmarks/results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Quantization Error Distribution Profiler")
    print(f"Lattice: {args.lattice}, q={args.q}, M={args.M}")
    print(f"Matrix size: {args.batch_size} x {args.matrix_size}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Initialize profiler
    profiler = QuantizationErrorProfiler(lattice_type=args.lattice, q=args.q, M=args.M)
    
    # Run profiling
    results = profiler.profile_quantization_errors(
        batch_size=args.batch_size,
        matrix_size=args.matrix_size,
        output_dir=output_dir
    )
    
    # Save results
    profiler.save_results(results, output_dir)
    
    print("\n" + "="*50)
    print("PROFILING COMPLETE")
    print("="*50)
    print(f"Results saved to: {output_dir}")
    print("Check the histogram and CSV files for detailed analysis.")


if __name__ == "__main__":
    main()
