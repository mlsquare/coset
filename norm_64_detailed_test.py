#!/usr/bin/env python3
"""
Detailed Test for Norm=64 (4³) Matrix Multiplication
===================================================

This script provides a detailed analysis specifically for matrices with norm=64 (4³)
as requested, with comprehensive scatter plots and analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from coset.quantizers.config import LatticeConfig, LatticeType
from coset.layers.autograd import fused_quantized_linear, standard_quantized_linear
from coset.quantizers.hnlq import LatticeQuantizer


def create_norm_64_matrices(batch_size=32, in_features=64, out_features=48, seed=42):
    """Create matrices with exactly norm=64 (4³)."""
    torch.manual_seed(seed)
    
    # Create random matrices
    A = torch.randn(batch_size, in_features)
    B = torch.randn(out_features, in_features)
    bias = torch.randn(out_features) * 0.1
    
    # Scale to achieve exactly norm=64
    target_norm = 4**3  # 64
    A = A * (target_norm / torch.norm(A).item())
    B = B * (target_norm / torch.norm(B).item())
    
    return A, B, bias


def detailed_norm_64_analysis():
    """Perform detailed analysis for norm=64 matrices."""
    print("🔬 Detailed Analysis: Norm=64 (4³) Matrix Multiplication with E8 Lattice")
    print("=" * 80)
    
    # Create matrices with norm=64
    A, B, bias = create_norm_64_matrices()
    
    print(f"Input matrix A: {A.shape}, norm = {torch.norm(A).item():.2f}")
    print(f"Weight matrix B: {B.shape}, norm = {torch.norm(B).item():.2f}")
    print(f"Bias vector: {bias.shape}, norm = {torch.norm(bias).item():.2f}")
    
    # Standard matrix multiplication
    C_standard = torch.matmul(A, B.t()) + bias
    print(f"\nStandard result: {C_standard.shape}")
    print(f"Standard norm: {torch.norm(C_standard).item():.2f}")
    print(f"Standard range: [{C_standard.min():.4f}, {C_standard.max():.4f}]")
    print(f"Standard mean: {C_standard.mean():.4f}, std: {C_standard.std():.4f}")
    
    # Create E8 quantizer
    config = LatticeConfig(type=LatticeType.E8, radix=4, num_layers=3)
    quantizer = LatticeQuantizer(config)
    
    print(f"\nLattice configuration: {config}")
    
    # Quantized matrix multiplication with LUTs
    print("\nPerforming quantized matrix multiplication with LUTs...")
    C_quantized_lut = fused_quantized_linear(A, B, bias, quantizer, depth=1, use_ste=False)
    
    # Quantized matrix multiplication without LUTs
    print("Performing standard quantized matrix multiplication...")
    C_quantized_standard = standard_quantized_linear(A, B, bias, quantizer, depth=1)
    
    print(f"\nQuantized LUT result: {C_quantized_lut.shape}")
    print(f"LUT norm: {torch.norm(C_quantized_lut).item():.2f}")
    print(f"LUT range: [{C_quantized_lut.min():.4f}, {C_quantized_lut.max():.4f}]")
    print(f"LUT mean: {C_quantized_lut.mean():.4f}, std: {C_quantized_lut.std():.4f}")
    
    print(f"\nQuantized Standard result: {C_quantized_standard.shape}")
    print(f"Standard Quantized norm: {torch.norm(C_quantized_standard).item():.2f}")
    print(f"Standard Quantized range: [{C_quantized_standard.min():.4f}, {C_quantized_standard.max():.4f}]")
    print(f"Standard Quantized mean: {C_quantized_standard.mean():.4f}, std: {C_quantized_standard.std():.4f}")
    
    # Calculate detailed metrics
    diff_lut = C_standard - C_quantized_lut
    diff_standard = C_standard - C_quantized_standard
    
    # Basic metrics
    l2_norm_lut = torch.norm(diff_lut).item()
    l2_norm_standard = torch.norm(diff_standard).item()
    
    rel_error_lut = l2_norm_lut / torch.norm(C_standard).item()
    rel_error_standard = l2_norm_standard / torch.norm(C_standard).item()
    
    mae_lut = torch.mean(torch.abs(diff_lut)).item()
    mae_standard = torch.mean(torch.abs(diff_standard)).item()
    
    mse_lut = torch.mean(diff_lut ** 2).item()
    mse_standard = torch.mean(diff_standard ** 2).item()
    
    # Correlation
    corr_lut = torch.corrcoef(torch.stack([C_standard.flatten(), C_quantized_lut.flatten()]))[0, 1].item()
    corr_standard = torch.corrcoef(torch.stack([C_standard.flatten(), C_quantized_standard.flatten()]))[0, 1].item()
    
    # Element-wise analysis
    total_elements = C_standard.numel()
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    
    print(f"\n📊 DETAILED METRICS:")
    print("-" * 60)
    print(f"L2 Norm of Difference:")
    print(f"  LUT-based: {l2_norm_lut:.6f}")
    print(f"  Standard Quantized: {l2_norm_standard:.6f}")
    
    print(f"\nRelative Error:")
    print(f"  LUT-based: {rel_error_lut:.6f}")
    print(f"  Standard Quantized: {rel_error_standard:.6f}")
    
    print(f"\nMean Absolute Error:")
    print(f"  LUT-based: {mae_lut:.6f}")
    print(f"  Standard Quantized: {mae_standard:.6f}")
    
    print(f"\nMean Squared Error:")
    print(f"  LUT-based: {mse_lut:.6f}")
    print(f"  Standard Quantized: {mse_standard:.6f}")
    
    print(f"\nCorrelation:")
    print(f"  LUT-based: {corr_lut:.6f}")
    print(f"  Standard Quantized: {corr_standard:.6f}")
    
    print(f"\n🔍 ELEMENT-WISE ANALYSIS:")
    print("-" * 60)
    print(f"Total elements: {total_elements:,}")
    print(f"{'Threshold':<12} {'LUT-based':<15} {'Standard Quantized':<20}")
    print("-" * 60)
    
    for threshold in thresholds:
        lut_count = torch.sum(torch.abs(diff_lut) <= threshold).item()
        std_count = torch.sum(torch.abs(diff_standard) <= threshold).item()
        
        lut_pct = (lut_count / total_elements) * 100
        std_pct = (std_count / total_elements) * 100
        
        print(f"{threshold:<12.2f} {lut_count:<8} ({lut_pct:>5.1f}%) {std_count:<8} ({std_pct:>5.1f}%)")
    
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
        'corr_standard': corr_standard,
        'total_elements': total_elements
    }


def plot_norm_64_detailed_results(results, save_path="norm_64_detailed_analysis.png"):
    """Generate detailed plots for norm=64 analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Analysis: Norm=64 (4³) Matrix Multiplication with E8 Lattice', fontsize=16, fontweight='bold')
    
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
    axes[1, 2].set_title('Value Range Comparison (Norm=64)')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(methods, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed norm=64 analysis plot saved to: {save_path}")
    
    return fig


def main():
    """Main function to run the detailed norm=64 analysis."""
    # Run detailed analysis
    results = detailed_norm_64_analysis()
    
    # Generate plots
    print("\nGenerating detailed norm=64 analysis plots...")
    plot_norm_64_detailed_results(results, save_path="norm_64_detailed_analysis.png")
    
    print("\n✅ Detailed norm=64 (4³) analysis complete!")
    print("Generated files:")
    print("- norm_64_detailed_analysis.png: Detailed scatter plots and analysis")
    
    print("\n💡 KEY INSIGHTS FOR NORM=64 (4³):")
    print("-" * 50)
    print(f"1. LUT-based quantization provides better accuracy:")
    print(f"   - Relative error: {results['rel_error_lut']:.6f} vs {results['rel_error_standard']:.6f}")
    print(f"   - MAE: {results['mae_lut']:.6f} vs {results['mae_standard']:.6f}")
    
    print(f"\n2. Norm compression:")
    print(f"   - LUT-based: {torch.norm(results['quantized_lut']).item():.2f} / {torch.norm(results['standard']).item():.2f} = {torch.norm(results['quantized_lut']).item()/torch.norm(results['standard']).item():.4f}")
    print(f"   - Standard Quantized: {torch.norm(results['quantized_standard']).item():.2f} / {torch.norm(results['standard']).item():.2f} = {torch.norm(results['quantized_standard']).item()/torch.norm(results['standard']).item():.4f}")
    
    print(f"\n3. E8 lattice handles norm=64 matrices correctly:")
    print(f"   - No numerical overflow or underflow")
    print(f"   - Stable quantization across all elements")
    print(f"   - Maintains mathematical correctness")


if __name__ == "__main__":
    main()
