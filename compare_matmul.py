#!/usr/bin/env python3
"""
Compare Quantized vs Standard Matrix Multiplication

This script compares the results of quantized matrix multiplication
with standard PyTorch matrix multiplication to analyze the accuracy
and behavior of the quantized operations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coset.quantizers.config import LatticeConfig, LatticeType
from coset.layers.linear import QuantizedLinear
from coset.layers.autograd import fused_quantized_linear, standard_quantized_linear
from coset.quantizers.hnlq import LatticeQuantizer


def create_test_matrices(
    batch_size: int = 4,
    in_features: int = 16,
    out_features: int = 8,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create test matrices A and B for matrix multiplication.
    
    Args:
        batch_size: Batch size for input matrix A
        in_features: Input features (columns of A, rows of B)
        out_features: Output features (columns of B)
        seed: Random seed for reproducibility
        
    Returns:
        A: Input matrix [batch_size, in_features]
        B: Weight matrix [out_features, in_features]
    """
    torch.manual_seed(seed)
    
    # Create input matrix A
    A = torch.randn(batch_size, in_features) * 0.5  # Small values for better quantization
    
    # Create weight matrix B
    B = torch.randn(out_features, in_features) * 0.3
    
    return A, B


def standard_matmul(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Perform standard PyTorch matrix multiplication.
    
    Args:
        A: Input matrix [batch_size, in_features]
        B: Weight matrix [out_features, in_features]
        bias: Optional bias [out_features]
        
    Returns:
        C: Result matrix [batch_size, out_features]
    """
    C = torch.matmul(A, B.t())
    if bias is not None:
        C = C + bias
    return C


def quantized_matmul_comparison(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    config: LatticeConfig,
    depth: int = 1
) -> Dict[str, Any]:
    """
    Compare quantized matrix multiplication with standard matrix multiplication.
    
    Args:
        A: Input matrix [batch_size, in_features]
        B: Weight matrix [out_features, in_features]
        bias: Bias vector [out_features]
        config: Lattice quantization configuration
        depth: Quantization depth
        
    Returns:
        Dictionary containing comparison results
    """
    # Standard matrix multiplication
    C_standard = standard_matmul(A, B, bias)
    
    # Create quantizer
    quantizer = LatticeQuantizer(config)
    
    # Quantized matrix multiplication with LUTs
    C_quantized_lut = fused_quantized_linear(A, B, bias, quantizer, depth, use_ste=False)
    
    # Quantized matrix multiplication without LUTs
    C_quantized_std = standard_quantized_linear(A, B, bias, quantizer, depth)
    
    # Compute differences
    diff_lut = C_standard - C_quantized_lut
    diff_std = C_standard - C_quantized_std
    
    # Compute norms
    norm_lut = torch.norm(diff_lut).item()
    norm_std = torch.norm(diff_std).item()
    
    # Compute relative errors
    rel_error_lut = norm_lut / torch.norm(C_standard).item()
    rel_error_std = norm_std / torch.norm(C_standard).item()
    
    # Compute element-wise statistics
    mae_lut = torch.mean(torch.abs(diff_lut)).item()
    mae_std = torch.mean(torch.abs(diff_std)).item()
    
    mse_lut = torch.mean(diff_lut ** 2).item()
    mse_std = torch.mean(diff_std ** 2).item()
    
    return {
        'C_standard': C_standard,
        'C_quantized_lut': C_quantized_lut,
        'C_quantized_std': C_quantized_std,
        'diff_lut': diff_lut,
        'diff_std': diff_std,
        'norm_lut': norm_lut,
        'norm_std': norm_std,
        'rel_error_lut': rel_error_lut,
        'rel_error_std': rel_error_std,
        'mae_lut': mae_lut,
        'mae_std': mae_std,
        'mse_lut': mse_lut,
        'mse_std': mse_std
    }


def plot_comparison(results: Dict[str, Any], save_path: str = None):
    """
    Create comparison plots for quantized vs standard matrix multiplication.
    
    Args:
        results: Results from quantized_matmul_comparison
        save_path: Optional path to save plots
    """
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantized vs Standard Matrix Multiplication Comparison', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. Scatter plot: LUT-based quantized vs standard
    ax1 = axes[0]
    C_std = results['C_standard'].detach().numpy().flatten()
    C_lut = results['C_quantized_lut'].detach().numpy().flatten()
    
    ax1.scatter(C_std, C_lut, alpha=0.6, s=20)
    
    # Add perfect correlation line (x=y)
    min_val = min(C_std.min(), C_lut.min())
    max_val = max(C_std.max(), C_lut.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect correlation (x=y)')
    
    ax1.set_xlabel('Standard Matrix Multiplication')
    ax1.set_ylabel('Quantized Matrix Multiplication (LUT)')
    ax1.set_title(f'LUT-based Quantized vs Standard\n(MAE: {results["mae_lut"]:.4f}, Rel Error: {results["rel_error_lut"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: Standard quantized vs standard
    ax2 = axes[1]
    C_std_q = results['C_quantized_std'].detach().numpy().flatten()
    
    ax2.scatter(C_std, C_std_q, alpha=0.6, s=20, color='orange')
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect correlation (x=y)')
    
    ax2.set_xlabel('Standard Matrix Multiplication')
    ax2.set_ylabel('Quantized Matrix Multiplication (Standard)')
    ax2.set_title(f'Standard Quantized vs Standard\n(MAE: {results["mae_std"]:.4f}, Rel Error: {results["rel_error_std"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference histogram: LUT-based
    ax3 = axes[2]
    diff_lut = results['diff_lut'].detach().numpy().flatten()
    ax3.hist(diff_lut, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Difference (Standard - Quantized LUT)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Distribution of Differences (LUT)\n(Mean: {np.mean(diff_lut):.4f}, Std: {np.std(diff_lut):.4f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Difference histogram: Standard quantized
    ax4 = axes[3]
    diff_std = results['diff_std'].detach().numpy().flatten()
    ax4.hist(diff_std, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Difference (Standard - Quantized Standard)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Distribution of Differences (Standard)\n(Mean: {np.mean(diff_std):.4f}, Std: {np.std(diff_std):.4f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
    
    plt.show()


def print_comparison_summary(results: Dict[str, Any]):
    """
    Print a summary of the comparison results.
    
    Args:
        results: Results from quantized_matmul_comparison
    """
    print("=" * 80)
    print("MATRIX MULTIPLICATION COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\nMatrix Dimensions:")
    print(f"  Input A: {results['C_standard'].shape}")
    print(f"  Weight B: {results['C_standard'].shape[1]} x {results['C_standard'].shape[0]}")
    print(f"  Output C: {results['C_standard'].shape}")
    
    print(f"\nLUT-based Quantized vs Standard:")
    print(f"  L2 Norm of Difference: {results['norm_lut']:.6f}")
    print(f"  Relative Error: {results['rel_error_lut']:.6f}")
    print(f"  Mean Absolute Error: {results['mae_lut']:.6f}")
    print(f"  Mean Squared Error: {results['mse_lut']:.6f}")
    
    print(f"\nStandard Quantized vs Standard:")
    print(f"  L2 Norm of Difference: {results['norm_std']:.6f}")
    print(f"  Relative Error: {results['rel_error_std']:.6f}")
    print(f"  Mean Absolute Error: {results['mae_std']:.6f}")
    print(f"  Mean Squared Error: {results['mse_std']:.6f}")
    
    print(f"\nValue Ranges:")
    print(f"  Standard: [{results['C_standard'].min():.4f}, {results['C_standard'].max():.4f}]")
    print(f"  Quantized LUT: [{results['C_quantized_lut'].min():.4f}, {results['C_quantized_lut'].max():.4f}]")
    print(f"  Quantized Standard: [{results['C_quantized_std'].min():.4f}, {results['C_quantized_std'].max():.4f}]")
    
    print(f"\nCorrelation Analysis:")
    C_std = results['C_standard'].detach().numpy().flatten()
    C_lut = results['C_quantized_lut'].detach().numpy().flatten()
    C_std_q = results['C_quantized_std'].detach().numpy().flatten()
    
    corr_lut = np.corrcoef(C_std, C_lut)[0, 1]
    corr_std = np.corrcoef(C_std, C_std_q)[0, 1]
    
    print(f"  LUT vs Standard Correlation: {corr_lut:.6f}")
    print(f"  Standard Quantized vs Standard Correlation: {corr_std:.6f}")


def main():
    """Main function to run the matrix multiplication comparison."""
    print("🔬 Quantized vs Standard Matrix Multiplication Comparison")
    print("=" * 60)
    
    # Create test matrices (compatible with E8 lattice dimension = 8)
    print("Creating test matrices...")
    A, B = create_test_matrices(batch_size=8, in_features=16, out_features=8)  # 16 = 2 * 8
    bias = torch.randn(8) * 0.1
    
    print(f"Input matrix A: {A.shape}")
    print(f"Weight matrix B: {B.shape}")
    print(f"Bias vector: {bias.shape}")
    
    # Create lattice configuration
    config = LatticeConfig(
        type=LatticeType.E8,
        radix=4,
        num_layers=3
    )
    
    print(f"\nLattice configuration: {config}")
    
    # Perform comparison
    print("\nPerforming matrix multiplication comparison...")
    results = quantized_matmul_comparison(A, B, bias, config, depth=1)
    
    # Print summary
    print_comparison_summary(results)
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, save_path="matmul_comparison.png")
    
    print("\n✅ Comparison complete!")
    print("\nInterpretation:")
    print("- Scatter plots should show points close to the x=y line for good accuracy")
    print("- Lower MAE and relative error indicate better quantization accuracy")
    print("- Correlation close to 1.0 indicates strong linear relationship")
    print("- Difference histograms should be centered around 0 with small spread")


if __name__ == "__main__":
    main()
