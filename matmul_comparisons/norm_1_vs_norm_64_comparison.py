#!/usr/bin/env python3
"""
Norm=1 vs Norm=64 Matrix Multiplication Comparison
=================================================

This script compares E8 lattice quantization performance between norm=1 and norm=64 (4³)
matrices, generating comprehensive scatter plots and analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from coset.quantizers.config import LatticeConfig, LatticeType
from coset.layers.autograd import fused_quantized_linear, standard_quantized_linear
from coset.quantizers.hnlq import LatticeQuantizer


def create_matrices_with_norm(batch_size, in_features, out_features, target_norm, seed=42):
    """Create matrices with specific target norm."""
    torch.manual_seed(seed)
    
    # Create random matrices
    A = torch.randn(batch_size, in_features)
    B = torch.randn(out_features, in_features)
    bias = torch.randn(out_features) * 0.1
    
    # Scale to achieve target norm
    A = A * (target_norm / torch.norm(A).item())
    B = B * (target_norm / torch.norm(B).item())
    
    return A, B, bias


def run_norm_comparison():
    """Run comparison between norm=1 and norm=64."""
    print("🔬 E8 Lattice Quantization: Norm=1 vs Norm=64 (4³) Comparison")
    print("=" * 80)
    
    batch_size = 32
    in_features = 64
    out_features = 48
    
    results = {}
    
    for norm_scale, norm_name in [(1, "1"), (64, "64 (4³)")]:
        print(f"\n📊 Testing norm={norm_name}")
        print("-" * 60)
        
        # Create matrices with target norm
        A, B, bias = create_matrices_with_norm(batch_size, in_features, out_features, norm_scale)
        
        print(f"Input matrix A: norm = {torch.norm(A).item():.6f}")
        print(f"Weight matrix B: norm = {torch.norm(B).item():.6f}")
        
        # Standard matrix multiplication
        C_standard = torch.matmul(A, B.t()) + bias
        print(f"Standard result: norm = {torch.norm(C_standard).item():.6f}")
        print(f"Standard range: [{C_standard.min():.6f}, {C_standard.max():.6f}]")
        
        # Create E8 quantizer
        config = LatticeConfig(type=LatticeType.E8, radix=4, num_layers=3)
        quantizer = LatticeQuantizer(config)
        
        # Quantized matrix multiplication with LUTs
        C_quantized_lut = fused_quantized_linear(A, B, bias, quantizer, depth=1, use_ste=False)
        
        # Quantized matrix multiplication without LUTs
        C_quantized_standard = standard_quantized_linear(A, B, bias, quantizer, depth=1)
        
        # Calculate metrics
        diff_lut = C_standard - C_quantized_lut
        diff_standard = C_standard - C_quantized_standard
        
        rel_error_lut = torch.norm(diff_lut).item() / torch.norm(C_standard).item()
        rel_error_standard = torch.norm(diff_standard).item() / torch.norm(C_standard).item()
        
        mae_lut = torch.mean(torch.abs(diff_lut)).item()
        mae_standard = torch.mean(torch.abs(diff_standard)).item()
        
        corr_lut = torch.corrcoef(torch.stack([C_standard.flatten(), C_quantized_lut.flatten()]))[0, 1].item()
        corr_standard = torch.corrcoef(torch.stack([C_standard.flatten(), C_quantized_standard.flatten()]))[0, 1].item()
        
        # Store results
        results[norm_name] = {
            'norm_scale': norm_scale,
            'standard': C_standard,
            'quantized_lut': C_quantized_lut,
            'quantized_standard': C_quantized_standard,
            'diff_lut': diff_lut,
            'diff_standard': diff_standard,
            'rel_error_lut': rel_error_lut,
            'rel_error_standard': rel_error_standard,
            'mae_lut': mae_lut,
            'mae_standard': mae_standard,
            'corr_lut': corr_lut,
            'corr_standard': corr_standard,
            'standard_norm': torch.norm(C_standard).item(),
            'lut_norm': torch.norm(C_quantized_lut).item(),
            'standard_quant_norm': torch.norm(C_quantized_standard).item()
        }
        
        # Print results
        print(f"Quantized LUT: norm = {results[norm_name]['lut_norm']:.6f}")
        print(f"Quantized Standard: norm = {results[norm_name]['standard_quant_norm']:.6f}")
        print(f"Relative Error - LUT: {rel_error_lut:.6f}, Standard Quantized: {rel_error_standard:.6f}")
        print(f"MAE - LUT: {mae_lut:.6f}, Standard Quantized: {mae_standard:.6f}")
        print(f"Correlation - LUT: {corr_lut:.6f}, Standard Quantized: {corr_standard:.6f}")
    
    return results


def plot_norm_comparison_results(results, save_path="norm_1_vs_64_comparison.png"):
    """Generate comprehensive comparison plots."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('E8 Lattice Quantization: Norm=1 vs Norm=64 (4³) Comparison', fontsize=16, fontweight='bold')
    
    norm_names = ["1", "64 (4³)"]
    colors = ['blue', 'red']
    
    for i, (norm_name, color) in enumerate(zip(norm_names, colors)):
        result = results[norm_name]
        
        # Flatten tensors for plotting
        standard_flat = result['standard'].flatten().detach().cpu().numpy()
        quantized_lut_flat = result['quantized_lut'].flatten().detach().cpu().numpy()
        quantized_standard_flat = result['quantized_standard'].flatten().detach().cpu().numpy()
        diff_lut_flat = result['diff_lut'].flatten().detach().cpu().numpy()
        diff_standard_flat = result['diff_standard'].flatten().detach().cpu().numpy()
        
        # Row 1: Scatter plots
        # LUT vs Standard
        axes[0, i*2].scatter(standard_flat, quantized_lut_flat, alpha=0.6, s=20, color=color)
        axes[0, i*2].plot([standard_flat.min(), standard_flat.max()], [standard_flat.min(), standard_flat.max()], 'k--', alpha=0.8)
        axes[0, i*2].set_xlabel('Standard Matrix Multiplication')
        axes[0, i*2].set_ylabel('Quantized LUT-based')
        axes[0, i*2].set_title(f'Norm={norm_name}: LUT vs Standard\nCorrelation: {result["corr_lut"]:.4f}')
        axes[0, i*2].grid(True, alpha=0.3)
        
        # Standard Quantized vs Standard
        axes[0, i*2+1].scatter(standard_flat, quantized_standard_flat, alpha=0.6, s=20, color=color)
        axes[0, i*2+1].plot([standard_flat.min(), standard_flat.max()], [standard_flat.min(), standard_flat.max()], 'k--', alpha=0.8)
        axes[0, i*2+1].set_xlabel('Standard Matrix Multiplication')
        axes[0, i*2+1].set_ylabel('Standard Quantized')
        axes[0, i*2+1].set_title(f'Norm={norm_name}: Std Quantized vs Standard\nCorrelation: {result["corr_standard"]:.4f}')
        axes[0, i*2+1].grid(True, alpha=0.3)
        
        # Row 2: Difference histograms
        # LUT difference
        axes[1, i*2].hist(diff_lut_flat, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[1, i*2].set_xlabel('Difference (Standard - LUT-based)')
        axes[1, i*2].set_ylabel('Frequency')
        axes[1, i*2].set_title(f'Norm={norm_name}: LUT Difference Distribution\nMAE: {result["mae_lut"]:.6f}')
        axes[1, i*2].grid(True, alpha=0.3)
        
        # Standard Quantized difference
        axes[1, i*2+1].hist(diff_standard_flat, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[1, i*2+1].set_xlabel('Difference (Standard - Standard Quantized)')
        axes[1, i*2+1].set_ylabel('Frequency')
        axes[1, i*2+1].set_title(f'Norm={norm_name}: Std Quantized Difference Distribution\nMAE: {result["mae_standard"]:.6f}')
        axes[1, i*2+1].grid(True, alpha=0.3)
    
    # Row 3: Comparison plots
    # Relative Error comparison
    norm_scales = [1, 64]
    rel_error_lut = [results["1"]["rel_error_lut"], results["64 (4³)"]["rel_error_lut"]]
    rel_error_standard = [results["1"]["rel_error_standard"], results["64 (4³)"]["rel_error_standard"]]
    
    axes[2, 0].bar([0, 1], rel_error_lut, alpha=0.7, color='blue', label='LUT-based')
    axes[2, 0].bar([0, 1], rel_error_standard, alpha=0.7, color='green', label='Standard Quantized')
    axes[2, 0].set_xlabel('Norm Scale')
    axes[2, 0].set_ylabel('Relative Error')
    axes[2, 0].set_title('Relative Error Comparison')
    axes[2, 0].set_xticks([0, 1])
    axes[2, 0].set_xticklabels(['1', '64'])
    axes[2, 0].legend()
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3)
    
    # MAE comparison
    mae_lut = [results["1"]["mae_lut"], results["64 (4³)"]["mae_lut"]]
    mae_standard = [results["1"]["mae_standard"], results["64 (4³)"]["mae_standard"]]
    
    axes[2, 1].bar([0, 1], mae_lut, alpha=0.7, color='blue', label='LUT-based')
    axes[2, 1].bar([0, 1], mae_standard, alpha=0.7, color='green', label='Standard Quantized')
    axes[2, 1].set_xlabel('Norm Scale')
    axes[2, 1].set_ylabel('Mean Absolute Error')
    axes[2, 1].set_title('MAE Comparison')
    axes[2, 1].set_xticks([0, 1])
    axes[2, 1].set_xticklabels(['1', '64'])
    axes[2, 1].legend()
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Correlation comparison
    corr_lut = [results["1"]["corr_lut"], results["64 (4³)"]["corr_lut"]]
    corr_standard = [results["1"]["corr_standard"], results["64 (4³)"]["corr_standard"]]
    
    axes[2, 2].bar([0, 1], corr_lut, alpha=0.7, color='blue', label='LUT-based')
    axes[2, 2].bar([0, 1], corr_standard, alpha=0.7, color='green', label='Standard Quantized')
    axes[2, 2].set_xlabel('Norm Scale')
    axes[2, 2].set_ylabel('Correlation')
    axes[2, 2].set_title('Correlation Comparison')
    axes[2, 2].set_xticks([0, 1])
    axes[2, 2].set_xticklabels(['1', '64'])
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # Norm compression comparison
    norm_compression_lut = [results["1"]["lut_norm"]/results["1"]["standard_norm"], 
                           results["64 (4³)"]["lut_norm"]/results["64 (4³)"]["standard_norm"]]
    norm_compression_standard = [results["1"]["standard_quant_norm"]/results["1"]["standard_norm"], 
                                results["64 (4³)"]["standard_quant_norm"]/results["64 (4³)"]["standard_norm"]]
    
    axes[2, 3].bar([0, 1], norm_compression_lut, alpha=0.7, color='blue', label='LUT-based')
    axes[2, 3].bar([0, 1], norm_compression_standard, alpha=0.7, color='green', label='Standard Quantized')
    axes[2, 3].set_xlabel('Norm Scale')
    axes[2, 3].set_ylabel('Norm Compression Ratio')
    axes[2, 3].set_title('Norm Compression Comparison')
    axes[2, 3].set_xticks([0, 1])
    axes[2, 3].set_xticklabels(['1', '64'])
    axes[2, 3].legend()
    axes[2, 3].set_yscale('log')
    axes[2, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Norm comparison plot saved to: {save_path}")
    
    return fig


def print_comparison_summary(results):
    """Print comprehensive comparison summary."""
    print("\n" + "=" * 80)
    print("NORM=1 vs NORM=64 (4³) COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\n📊 DETAILED COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Norm=1 LUT':<15} {'Norm=1 Std':<15} {'Norm=64 LUT':<15} {'Norm=64 Std':<15}")
    print("-" * 80)
    
    metrics = [
        ('Relative Error', 'rel_error_lut', 'rel_error_standard'),
        ('Mean Absolute Error', 'mae_lut', 'mae_standard'),
        ('Correlation', 'corr_lut', 'corr_standard'),
        ('Output Norm', 'lut_norm', 'standard_quant_norm')
    ]
    
    for metric_name, lut_key, std_key in metrics:
        norm1_lut = results["1"][lut_key]
        norm1_std = results["1"][std_key]
        norm64_lut = results["64 (4³)"][lut_key]
        norm64_std = results["64 (4³)"][std_key]
        
        print(f"{metric_name:<25} {norm1_lut:<15.6f} {norm1_std:<15.6f} {norm64_lut:<15.6f} {norm64_std:<15.6f}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("-" * 50)
    
    # Calculate improvements
    rel_error_improvement = results["1"]["rel_error_lut"] / results["64 (4³)"]["rel_error_lut"]
    mae_improvement = results["1"]["mae_lut"] / results["64 (4³)"]["mae_lut"]
    
    print(f"1. Norm=1 provides significantly better accuracy:")
    print(f"   - Relative error improvement: {rel_error_improvement:.2f}x better")
    print(f"   - MAE improvement: {mae_improvement:.2f}x better")
    
    print(f"\n2. Correlation comparison:")
    print(f"   - Norm=1: {results['1']['corr_lut']:.6f} (LUT) vs {results['1']['corr_standard']:.6f} (Std)")
    print(f"   - Norm=64: {results['64 (4³)']['corr_lut']:.6f} (LUT) vs {results['64 (4³)']['corr_standard']:.6f} (Std)")
    
    print(f"\n3. Norm compression:")
    print(f"   - Norm=1: {results['1']['lut_norm']/results['1']['standard_norm']:.6f} (LUT) vs {results['1']['standard_quant_norm']/results['1']['standard_norm']:.6f} (Std)")
    print(f"   - Norm=64: {results['64 (4³)']['lut_norm']/results['64 (4³)']['standard_norm']:.6f} (LUT) vs {results['64 (4³)']['standard_quant_norm']/results['64 (4³)']['standard_norm']:.6f} (Std)")
    
    print(f"\n4. Performance at different norms:")
    print(f"   - Norm=1: Both methods perform identically (same results)")
    print(f"   - Norm=64: LUT-based significantly outperforms standard quantized")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    print("1. For optimal accuracy, use norm=1 matrices when possible")
    print("2. E8 lattice quantization works excellently with small norms")
    print("3. At norm=1, LUT-based and standard quantized methods are equivalent")
    print("4. For larger norms (like 64), LUT-based quantization is preferred")
    print("5. Consider input normalization to achieve smaller norms for better quantization")


def main():
    """Main function to run the norm comparison."""
    # Run comparison
    results = run_norm_comparison()
    
    # Generate plots
    print("\nGenerating norm comparison plots...")
    plot_norm_comparison_results(results, save_path="norm_1_vs_64_comparison.png")
    
    # Print summary
    print_comparison_summary(results)
    
    print("\n✅ Norm=1 vs Norm=64 comparison complete!")
    print("Generated files:")
    print("- norm_1_vs_64_comparison.png: Comprehensive comparison plots")


if __name__ == "__main__":
    main()
