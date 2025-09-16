#!/usr/bin/env python3
"""
Large Matrix Multiplication Comparison with E8 Lattice
======================================================

This script compares quantized vs standard matrix multiplication using large matrices
and generates comprehensive scatter plots and analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from coset.quantizers.config import LatticeConfig, LatticeType
from coset.layers.autograd import fused_quantized_linear, standard_quantized_linear
from coset.quantizers.hnlq import LatticeQuantizer


def create_large_test_matrices(batch_size=32, in_features=64, out_features=48, seed=42):
    """Create large test matrices for comparison."""
    torch.manual_seed(seed)
    
    # Create test matrices
    A = torch.randn(batch_size, in_features) * 0.5
    B = torch.randn(out_features, in_features) * 0.3
    bias = torch.randn(out_features) * 0.1
    
    return A, B, bias


def large_matmul_comparison(A, B, bias, config, depth=1):
    """Perform large matrix multiplication comparison."""
    print(f"Input matrix A: {A.shape}")
    print(f"Weight matrix B: {B.shape}")
    print(f"Bias vector: {bias.shape}")
    
    # Standard matrix multiplication
    C_standard = torch.matmul(A, B.t()) + bias
    
    # Create quantizer
    quantizer = LatticeQuantizer(config)
    
    # Quantized matrix multiplication with LUTs
    C_quantized_lut = fused_quantized_linear(A, B, bias, quantizer, depth=depth, use_ste=False)
    
    # Quantized matrix multiplication without LUTs
    C_quantized_standard = standard_quantized_linear(A, B, bias, quantizer, depth=depth)
    
    # Calculate differences
    diff_lut = C_standard - C_quantized_lut
    diff_standard = C_standard - C_quantized_standard
    
    # Calculate metrics
    l2_norm_lut = torch.norm(diff_lut).item()
    l2_norm_standard = torch.norm(diff_standard).item()
    
    rel_error_lut = l2_norm_lut / torch.norm(C_standard).item()
    rel_error_standard = l2_norm_standard / torch.norm(C_standard).item()
    
    mae_lut = torch.mean(torch.abs(diff_lut)).item()
    mae_standard = torch.mean(torch.abs(diff_standard)).item()
    
    mse_lut = torch.mean(diff_lut ** 2).item()
    mse_standard = torch.mean(diff_standard ** 2).item()
    
    # Calculate correlation
    corr_lut = torch.corrcoef(torch.stack([C_standard.flatten(), C_quantized_lut.flatten()]))[0, 1].item()
    corr_standard = torch.corrcoef(torch.stack([C_standard.flatten(), C_quantized_standard.flatten()]))[0, 1].item()
    
    return {
        'standard': C_standard,
        'quantized_lut': C_quantized_lut,
        'quantized_standard': C_quantized_standard,
        'diff_lut': diff_lut,
        'diff_standard': diff_standard,
        'l2_norm_lut': l2_norm_lut,
        'l2_norm_standard': l2_norm_standard,
        'rel_error_lut': rel_error_lut,
        'rel_error_standard': rel_error_standard,
        'mae_lut': mae_lut,
        'mae_standard': mae_standard,
        'mse_lut': mse_lut,
        'mse_standard': mse_standard,
        'corr_lut': corr_lut,
        'corr_standard': corr_standard
    }


def plot_comparison_results(results, save_path="large_matmul_comparison.png"):
    """Generate comprehensive comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Large Matrix Multiplication Comparison: E8 Lattice (32×64 → 32×48)', fontsize=16, fontweight='bold')
    
    # Flatten tensors for plotting
    standard_flat = results['standard'].flatten().detach().cpu().numpy()
    quantized_lut_flat = results['quantized_lut'].flatten().detach().cpu().numpy()
    quantized_standard_flat = results['quantized_standard'].flatten().detach().cpu().numpy()
    diff_lut_flat = results['diff_lut'].flatten().detach().cpu().numpy()
    diff_standard_flat = results['diff_standard'].flatten().detach().cpu().numpy()
    
    # Plot 1: LUT-based vs Standard scatter plot
    axes[0, 0].scatter(standard_flat, quantized_lut_flat, alpha=0.6, s=20, color='blue')
    axes[0, 0].plot([standard_flat.min(), standard_flat.max()], [standard_flat.min(), standard_flat.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Standard Matrix Multiplication')
    axes[0, 0].set_ylabel('Quantized LUT-based')
    axes[0, 0].set_title(f'LUT-based vs Standard\nCorrelation: {results["corr_lut"]:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Standard Quantized vs Standard scatter plot
    axes[0, 1].scatter(standard_flat, quantized_standard_flat, alpha=0.6, s=20, color='green')
    axes[0, 1].plot([standard_flat.min(), standard_flat.max()], [standard_flat.min(), standard_flat.max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Standard Matrix Multiplication')
    axes[0, 1].set_ylabel('Standard Quantized')
    axes[0, 1].set_title(f'Standard Quantized vs Standard\nCorrelation: {results["corr_standard"]:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: LUT vs Standard Quantized scatter plot
    axes[0, 2].scatter(quantized_standard_flat, quantized_lut_flat, alpha=0.6, s=20, color='purple')
    axes[0, 2].plot([quantized_standard_flat.min(), quantized_standard_flat.max()], 
                    [quantized_standard_flat.min(), quantized_standard_flat.max()], 'r--', alpha=0.8)
    axes[0, 2].set_xlabel('Standard Quantized')
    axes[0, 2].set_ylabel('Quantized LUT-based')
    axes[0, 2].set_title('LUT-based vs Standard Quantized')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Difference histogram (LUT)
    axes[1, 0].hist(diff_lut_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('Difference (Standard - LUT-based)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'LUT Difference Distribution\nMAE: {results["mae_lut"]:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Difference histogram (Standard Quantized)
    axes[1, 1].hist(diff_standard_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('Difference (Standard - Standard Quantized)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Standard Quantized Difference Distribution\nMAE: {results["mae_standard"]:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Value range comparison
    methods = ['Standard', 'LUT-based', 'Std Quantized']
    min_vals = [standard_flat.min(), quantized_lut_flat.min(), quantized_standard_flat.min()]
    max_vals = [standard_flat.max(), quantized_lut_flat.max(), quantized_standard_flat.max()]
    
    x_pos = np.arange(len(methods))
    axes[1, 2].bar(x_pos - 0.2, min_vals, 0.4, label='Min', alpha=0.7, color='red')
    axes[1, 2].bar(x_pos + 0.2, max_vals, 0.4, label='Max', alpha=0.7, color='blue')
    axes[1, 2].set_xlabel('Method')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_title('Value Range Comparison')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(methods, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {save_path}")
    
    return fig


def print_comparison_summary(results):
    """Print detailed comparison summary."""
    print("\n" + "=" * 80)
    print("LARGE MATRIX MULTIPLICATION COMPARISON SUMMARY")
    print("=" * 80 + "\n")
    
    print("📊 METRICS COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'LUT-based':<15} {'Standard Quantized':<20}")
    print("-" * 50)
    print(f"{'L2 Norm of Difference':<25} {results['l2_norm_lut']:<15.6f} {results['l2_norm_standard']:<20.6f}")
    print(f"{'Relative Error':<25} {results['rel_error_lut']:<15.6f} {results['rel_error_standard']:<20.6f}")
    print(f"{'Mean Absolute Error':<25} {results['mae_lut']:<15.6f} {results['mae_standard']:<20.6f}")
    print(f"{'Mean Squared Error':<25} {results['mse_lut']:<15.6f} {results['mse_standard']:<20.6f}")
    print(f"{'Correlation':<25} {results['corr_lut']:<15.6f} {results['corr_standard']:<20.6f}")
    
    print(f"\n📈 VALUE RANGES")
    print("-" * 50)
    standard_flat = results['standard'].flatten().detach().cpu().numpy()
    quantized_lut_flat = results['quantized_lut'].flatten().detach().cpu().numpy()
    quantized_standard_flat = results['quantized_standard'].flatten().detach().cpu().numpy()
    
    print(f"Standard:           [{standard_flat.min():.4f}, {standard_flat.max():.4f}]")
    print(f"Quantized LUT:      [{quantized_lut_flat.min():.4f}, {quantized_lut_flat.max():.4f}]")
    print(f"Quantized Standard: [{quantized_standard_flat.min():.4f}, {quantized_standard_flat.max():.4f}]")
    
    print(f"\n🔍 ELEMENT-WISE COMPARISON")
    print("-" * 50)
    total_elements = len(standard_flat)
    
    # Calculate thresholds
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
    
    print(f"{'Threshold':<12} {'LUT-based':<15} {'Standard Quantized':<20}")
    print("-" * 50)
    
    for threshold in thresholds:
        lut_count = np.sum(np.abs(results['diff_lut'].flatten().detach().cpu().numpy()) <= threshold)
        std_count = np.sum(np.abs(results['diff_standard'].flatten().detach().cpu().numpy()) <= threshold)
        
        lut_pct = (lut_count / total_elements) * 100
        std_pct = (std_count / total_elements) * 100
        
        print(f"{threshold:<12.2f} {lut_count:<8} ({lut_pct:>5.1f}%) {std_count:<8} ({std_pct:>5.1f}%)")
    
    print(f"\n💡 INTERPRETATION:")
    print("-" * 50)
    print(f"- Large matrix test: {results['standard'].shape[0]}×{results['standard'].shape[1]} elements")
    print(f"- E8 lattice (8D) with radix=4, M=3 layers")
    print(f"- Lower relative error and MAE indicate better accuracy")
    print(f"- Correlation close to 1.0 indicates strong linear relationship")
    print(f"- Higher percentage of elements within small thresholds is better")
    print(f"- LUT-based quantization shows {'better' if results['rel_error_lut'] < results['rel_error_standard'] else 'worse'} accuracy than standard quantized")


def main():
    """Main function to run large matrix multiplication comparison."""
    print("🔬 Large Matrix Multiplication Comparison with E8 Lattice")
    print("=" * 70)
    
    # Create large test matrices
    print("Creating large test matrices...")
    A, B, bias = create_large_test_matrices(batch_size=32, in_features=64, out_features=48)
    
    # Create E8 lattice configuration
    config = LatticeConfig(type=LatticeType.E8, radix=4, num_layers=3)
    print(f"\nLattice configuration: {config}")
    
    # Perform comparison
    print("\nPerforming large matrix multiplication comparison...")
    results = large_matmul_comparison(A, B, bias, config, depth=1)
    
    # Print summary
    print_comparison_summary(results)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison_results(results, save_path="large_matmul_comparison.png")
    
    print("\n✅ Large matrix comparison complete!")
    print("\nInterpretation:")
    print("- Scatter plots should show points close to the x=y line for good accuracy")
    print("- Lower MAE and relative error indicate better quantization accuracy")
    print("- Correlation close to 1.0 indicates strong linear relationship")
    print("- Difference histograms should be centered around 0 with small spread")


if __name__ == "__main__":
    main()
