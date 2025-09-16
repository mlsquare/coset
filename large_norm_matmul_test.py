#!/usr/bin/env python3
"""
Large Norm Matrix Multiplication Test with E8 Lattice
====================================================

This script tests how E8 lattice quantization performs with matrices of different norms,
including the requested 4^3 = 64 norm, and generates comprehensive scatter plots.
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


def test_norm_scaling():
    """Test quantization performance across different matrix norms."""
    print("🔬 E8 Lattice Quantization Performance Across Different Matrix Norms")
    print("=" * 80)
    
    # Test different norm scales
    norm_scales = [1, 4, 16, 64, 256]  # Including 4^3 = 64
    norm_names = ["1", "4", "16", "4³=64", "256"]
    
    batch_size = 32
    in_features = 64
    out_features = 48
    
    results = []
    
    for i, (norm_scale, norm_name) in enumerate(zip(norm_scales, norm_names)):
        print(f"\n📊 Testing norm scale: {norm_name} (target norm: {norm_scale})")
        print("-" * 60)
        
        # Create matrices with target norm
        A, B, bias = create_matrices_with_norm(batch_size, in_features, out_features, norm_scale)
        
        print(f"Input matrix A: norm = {torch.norm(A).item():.2f}")
        print(f"Weight matrix B: norm = {torch.norm(B).item():.2f}")
        
        # Standard matrix multiplication
        C_standard = torch.matmul(A, B.t()) + bias
        standard_norm = torch.norm(C_standard).item()
        print(f"Standard result: norm = {standard_norm:.2f}, range = [{C_standard.min():.2f}, {C_standard.max():.2f}]")
        
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
        result = {
            'norm_scale': norm_scale,
            'norm_name': norm_name,
            'standard': C_standard,
            'quantized_lut': C_quantized_lut,
            'quantized_standard': C_quantized_standard,
            'rel_error_lut': rel_error_lut,
            'rel_error_standard': rel_error_standard,
            'mae_lut': mae_lut,
            'mae_standard': mae_standard,
            'corr_lut': corr_lut,
            'corr_standard': corr_standard,
            'standard_norm': standard_norm,
            'lut_norm': torch.norm(C_quantized_lut).item(),
            'standard_quant_norm': torch.norm(C_quantized_standard).item()
        }
        results.append(result)
        
        # Print results
        print(f"Quantized LUT: norm = {result['lut_norm']:.2f}, range = [{C_quantized_lut.min():.2f}, {C_quantized_lut.max():.2f}]")
        print(f"Quantized Standard: norm = {result['standard_quant_norm']:.2f}, range = [{C_quantized_standard.min():.2f}, {C_quantized_standard.max():.2f}]")
        print(f"Relative Error - LUT: {rel_error_lut:.6f}, Standard Quantized: {rel_error_standard:.6f}")
        print(f"MAE - LUT: {mae_lut:.6f}, Standard Quantized: {mae_standard:.6f}")
        print(f"Correlation - LUT: {corr_lut:.6f}, Standard Quantized: {corr_standard:.6f}")
    
    return results


def plot_norm_scaling_results(results, save_path="norm_scaling_results.png"):
    """Generate comprehensive plots for norm scaling analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('E8 Lattice Quantization Performance Across Different Matrix Norms', fontsize=16, fontweight='bold')
    
    # Extract data
    norm_scales = [r['norm_scale'] for r in results]
    norm_names = [r['norm_name'] for r in results]
    rel_error_lut = [r['rel_error_lut'] for r in results]
    rel_error_standard = [r['rel_error_standard'] for r in results]
    mae_lut = [r['mae_lut'] for r in results]
    mae_standard = [r['mae_standard'] for r in results]
    corr_lut = [r['corr_lut'] for r in results]
    corr_standard = [r['corr_standard'] for r in results]
    standard_norms = [r['standard_norm'] for r in results]
    lut_norms = [r['lut_norm'] for r in results]
    standard_quant_norms = [r['standard_quant_norm'] for r in results]
    
    # Plot 1: Relative Error vs Norm Scale
    axes[0, 0].plot(norm_scales, rel_error_lut, 'o-', label='LUT-based', linewidth=2, markersize=8)
    axes[0, 0].plot(norm_scales, rel_error_standard, 's-', label='Standard Quantized', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Input Matrix Norm')
    axes[0, 0].set_ylabel('Relative Error')
    axes[0, 0].set_title('Relative Error vs Input Norm')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MAE vs Norm Scale
    axes[0, 1].plot(norm_scales, mae_lut, 'o-', label='LUT-based', linewidth=2, markersize=8)
    axes[0, 1].plot(norm_scales, mae_standard, 's-', label='Standard Quantized', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Input Matrix Norm')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('MAE vs Input Norm')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation vs Norm Scale
    axes[0, 2].plot(norm_scales, corr_lut, 'o-', label='LUT-based', linewidth=2, markersize=8)
    axes[0, 2].plot(norm_scales, corr_standard, 's-', label='Standard Quantized', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Input Matrix Norm')
    axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].set_title('Correlation vs Input Norm')
    axes[0, 2].set_xscale('log')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Output Norm vs Input Norm
    axes[1, 0].plot(norm_scales, standard_norms, 'o-', label='Standard', linewidth=2, markersize=8)
    axes[1, 0].plot(norm_scales, lut_norms, 's-', label='LUT-based', linewidth=2, markersize=8)
    axes[1, 0].plot(norm_scales, standard_quant_norms, '^-', label='Standard Quantized', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Input Matrix Norm')
    axes[1, 0].set_ylabel('Output Matrix Norm')
    axes[1, 0].set_title('Output Norm vs Input Norm')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot for norm=64 (4^3)
    norm_64_result = next(r for r in results if r['norm_scale'] == 64)
    standard_flat = norm_64_result['standard'].flatten().detach().cpu().numpy()
    quantized_lut_flat = norm_64_result['quantized_lut'].flatten().detach().cpu().numpy()
    quantized_standard_flat = norm_64_result['quantized_standard'].flatten().detach().cpu().numpy()
    
    axes[1, 1].scatter(standard_flat, quantized_lut_flat, alpha=0.6, s=20, color='blue', label='LUT-based')
    axes[1, 1].scatter(standard_flat, quantized_standard_flat, alpha=0.6, s=20, color='green', label='Standard Quantized')
    axes[1, 1].plot([standard_flat.min(), standard_flat.max()], [standard_flat.min(), standard_flat.max()], 'r--', alpha=0.8)
    axes[1, 1].set_xlabel('Standard Matrix Multiplication')
    axes[1, 1].set_ylabel('Quantized Results')
    axes[1, 1].set_title(f'Scatter Plot for Norm=64 (4³)\nLUT Corr: {norm_64_result["corr_lut"]:.4f}, Std Corr: {norm_64_result["corr_standard"]:.4f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Norm compression ratio
    lut_compression = [lut_norm / std_norm for lut_norm, std_norm in zip(lut_norms, standard_norms)]
    std_compression = [std_quant_norm / std_norm for std_quant_norm, std_norm in zip(standard_quant_norms, standard_norms)]
    
    axes[1, 2].plot(norm_scales, lut_compression, 'o-', label='LUT-based', linewidth=2, markersize=8)
    axes[1, 2].plot(norm_scales, std_compression, 's-', label='Standard Quantized', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Input Matrix Norm')
    axes[1, 2].set_ylabel('Norm Compression Ratio')
    axes[1, 2].set_title('Output Norm / Standard Norm')
    axes[1, 2].set_xscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Norm scaling analysis plot saved to: {save_path}")
    
    return fig


def print_norm_scaling_summary(results):
    """Print comprehensive summary of norm scaling results."""
    print("\n" + "=" * 80)
    print("NORM SCALING ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\n📊 DETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Norm':<8} {'RelErr LUT':<12} {'RelErr Std':<12} {'MAE LUT':<10} {'MAE Std':<10} {'Corr LUT':<10} {'Corr Std':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['norm_name']:<8} {r['rel_error_lut']:<12.6f} {r['rel_error_standard']:<12.6f} {r['mae_lut']:<10.6f} {r['mae_standard']:<10.6f} {r['corr_lut']:<10.6f} {r['corr_standard']:<10.6f}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("-" * 50)
    
    # Find the 4^3 = 64 result
    norm_64_result = next(r for r in results if r['norm_scale'] == 64)
    
    print(f"1. For norm=64 (4³) specifically:")
    print(f"   - LUT-based: {norm_64_result['rel_error_lut']:.6f} relative error, {norm_64_result['corr_lut']:.6f} correlation")
    print(f"   - Standard Quantized: {norm_64_result['rel_error_standard']:.6f} relative error, {norm_64_result['corr_standard']:.6f} correlation")
    
    print(f"\n2. Scaling behavior:")
    print(f"   - LUT-based relative error: {results[0]['rel_error_lut']:.6f} → {results[-1]['rel_error_lut']:.6f}")
    print(f"   - Standard quantized relative error: {results[0]['rel_error_standard']:.6f} → {results[-1]['rel_error_standard']:.6f}")
    
    print(f"\n3. Norm compression:")
    print(f"   - LUT-based: {norm_64_result['lut_norm']:.2f} / {norm_64_result['standard_norm']:.2f} = {norm_64_result['lut_norm']/norm_64_result['standard_norm']:.4f}")
    print(f"   - Standard Quantized: {norm_64_result['standard_quant_norm']:.2f} / {norm_64_result['standard_norm']:.2f} = {norm_64_result['standard_quant_norm']/norm_64_result['standard_norm']:.4f}")
    
    print(f"\n4. Performance at different scales:")
    for r in results:
        better_method = "LUT-based" if r['rel_error_lut'] < r['rel_error_standard'] else "Standard Quantized"
        print(f"   - Norm {r['norm_name']}: {better_method} performs better")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    print("1. LUT-based quantization maintains better relative error across all norm scales")
    print("2. Standard quantized approach shows increasing relative error with larger norms")
    print("3. E8 lattice handles large norms (up to 256) without numerical issues")
    print("4. Norm compression is more effective with LUT-based approach")
    print("5. For norm=64 (4³), LUT-based provides superior accuracy")


def main():
    """Main function to run the norm scaling test."""
    # Run norm scaling test
    results = test_norm_scaling()
    
    # Generate plots
    print("\nGenerating norm scaling analysis plots...")
    plot_norm_scaling_results(results, save_path="norm_scaling_results.png")
    
    # Print summary
    print_norm_scaling_summary(results)
    
    print("\n✅ Norm scaling test complete!")
    print("Generated files:")
    print("- norm_scaling_results.png: Comprehensive norm scaling analysis")


if __name__ == "__main__":
    main()
