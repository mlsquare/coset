#!/usr/bin/env python3
"""
E8 Lattice Matrix Multiplication Comparison

This script compares quantized matrix multiplication using E8 lattice
with standard PyTorch matrix multiplication.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coset.quantizers.config import LatticeConfig, LatticeType
from coset.layers.autograd import fused_quantized_linear, standard_quantized_linear
from coset.quantizers.hnlq import LatticeQuantizer


def e8_matmul_comparison():
    """Comparison using E8 lattice with proper dimensions."""
    print("🔬 E8 Lattice Quantized vs Standard Matrix Multiplication Comparison")
    print("=" * 75)
    
    # Create test matrices (E8 lattice dimension = 8)
    torch.manual_seed(42)
    A = torch.randn(4, 8) * 0.5   # Input matrix (8 = 1 * 8, exact lattice dimension)
    B = torch.randn(6, 8) * 0.3   # Weight matrix (8 = 1 * 8, exact lattice dimension)
    bias = torch.randn(6) * 0.1   # Bias vector
    
    print(f"Input matrix A: {A.shape}")
    print(f"Weight matrix B: {B.shape}")
    print(f"Bias vector: {bias.shape}")
    print(f"E8 lattice dimension: 8")
    
    # Standard matrix multiplication
    C_standard = torch.matmul(A, B.t()) + bias
    
    # Create E8 quantizer
    config = LatticeConfig(type=LatticeType.E8, radix=4, num_layers=3)
    quantizer = LatticeQuantizer(config)
    
    print(f"\nLattice configuration: {config}")
    
    try:
        # Quantized matrix multiplication with LUTs
        C_quantized_lut = fused_quantized_linear(A, B, bias, quantizer, depth=1, use_ste=False)
        
        # Quantized matrix multiplication without LUTs
        C_quantized_std = standard_quantized_linear(A, B, bias, quantizer, depth=1)
        
        # Compute differences
        diff_lut = C_standard - C_quantized_lut
        diff_std = C_standard - C_quantized_std
        
        # Compute metrics
        norm_lut = torch.norm(diff_lut).item()
        norm_std = torch.norm(diff_std).item()
        
        rel_error_lut = norm_lut / torch.norm(C_standard).item()
        rel_error_std = norm_std / torch.norm(C_standard).item()
        
        mae_lut = torch.mean(torch.abs(diff_lut)).item()
        mae_std = torch.mean(torch.abs(diff_std)).item()
        
        # Print results
        print(f"\n📊 COMPARISON RESULTS")
        print(f"{'Metric':<25} {'LUT-based':<15} {'Standard Quantized':<20}")
        print("-" * 60)
        print(f"{'L2 Norm of Difference':<25} {norm_lut:<15.6f} {norm_std:<20.6f}")
        print(f"{'Relative Error':<25} {rel_error_lut:<15.6f} {rel_error_std:<20.6f}")
        print(f"{'Mean Absolute Error':<25} {mae_lut:<15.6f} {mae_std:<20.6f}")
        
        print(f"\n📈 VALUE RANGES")
        print(f"Standard:           [{C_standard.min():.4f}, {C_standard.max():.4f}]")
        print(f"Quantized LUT:      [{C_quantized_lut.min():.4f}, {C_quantized_lut.max():.4f}]")
        print(f"Quantized Standard: [{C_quantized_std.min():.4f}, {C_quantized_std.max():.4f}]")
        
        # Correlation analysis
        C_std_flat = C_standard.detach().numpy().flatten()
        C_lut_flat = C_quantized_lut.detach().numpy().flatten()
        C_std_q_flat = C_quantized_std.detach().numpy().flatten()
        
        corr_lut = np.corrcoef(C_std_flat, C_lut_flat)[0, 1]
        corr_std = np.corrcoef(C_std_flat, C_std_q_flat)[0, 1]
        
        print(f"\n🔗 CORRELATION ANALYSIS")
        print(f"LUT vs Standard:           {corr_lut:.6f}")
        print(f"Standard Quantized vs Standard: {corr_std:.6f}")
        
        # Element-wise comparison
        print(f"\n🔍 ELEMENT-WISE COMPARISON")
        print(f"Total elements: {C_standard.numel()}")
        
        # Count elements within different error thresholds
        abs_diff_lut = torch.abs(diff_lut)
        abs_diff_std = torch.abs(diff_std)
        
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        print(f"\n{'Threshold':<12} {'LUT-based':<15} {'Standard Quantized':<20}")
        print("-" * 50)
        for threshold in thresholds:
            count_lut = torch.sum(abs_diff_lut < threshold).item()
            count_std = torch.sum(abs_diff_std < threshold).item()
            pct_lut = (count_lut / C_standard.numel()) * 100
            pct_std = (count_std / C_standard.numel()) * 100
            print(f"{threshold:<12.2f} {count_lut:<15} ({pct_lut:>5.1f}%) {count_std:<20} ({pct_std:>5.1f}%)")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"- E8 lattice (8D) with radix=4, M=3 layers")
        print(f"- Lower relative error and MAE indicate better accuracy")
        print(f"- Correlation close to 1.0 indicates strong linear relationship")
        print(f"- Higher percentage of elements within small thresholds is better")
        print(f"- LUT-based quantization shows {'better' if rel_error_lut < rel_error_std else 'worse'} accuracy than standard quantized")
        
        return {
            'standard': C_standard,
            'quantized_lut': C_quantized_lut,
            'quantized_std': C_quantized_std,
            'rel_error_lut': rel_error_lut,
            'rel_error_std': rel_error_std,
            'corr_lut': corr_lut,
            'corr_std': corr_std
        }
        
    except Exception as e:
        print(f"\n❌ Error with E8 lattice: {e}")
        print(f"This suggests there may be an issue with the E8 lattice implementation.")
        print(f"Let's try with a different lattice type...")
        
        # Fallback to Z2 lattice
        print(f"\n🔄 Falling back to Z2 lattice...")
        config_z2 = LatticeConfig(type=LatticeType.Z2, radix=4, num_layers=3)
        quantizer_z2 = LatticeQuantizer(config_z2)
        
        # Adjust dimensions for Z2 (dimension = 2)
        A_z2 = torch.randn(4, 8) * 0.5   # 8 = 4 * 2
        B_z2 = torch.randn(6, 8) * 0.3   # 8 = 4 * 2
        bias_z2 = torch.randn(6) * 0.1
        
        C_standard_z2 = torch.matmul(A_z2, B_z2.t()) + bias_z2
        C_quantized_lut_z2 = fused_quantized_linear(A_z2, B_z2, bias_z2, quantizer_z2, depth=1, use_ste=False)
        C_quantized_std_z2 = standard_quantized_linear(A_z2, B_z2, bias_z2, quantizer_z2, depth=1)
        
        diff_lut_z2 = C_standard_z2 - C_quantized_lut_z2
        diff_std_z2 = C_standard_z2 - C_quantized_std_z2
        
        norm_lut_z2 = torch.norm(diff_lut_z2).item()
        norm_std_z2 = torch.norm(diff_std_z2).item()
        rel_error_lut_z2 = norm_lut_z2 / torch.norm(C_standard_z2).item()
        rel_error_std_z2 = norm_std_z2 / torch.norm(C_standard_z2).item()
        
        print(f"\n📊 Z2 LATTICE COMPARISON RESULTS")
        print(f"L2 Norm of Difference (LUT): {norm_lut_z2:.6f}")
        print(f"L2 Norm of Difference (Std): {norm_std_z2:.6f}")
        print(f"Relative Error (LUT): {rel_error_lut_z2:.6f}")
        print(f"Relative Error (Std): {rel_error_std_z2:.6f}")
        
        return None


if __name__ == "__main__":
    results = e8_matmul_comparison()
    if results:
        print(f"\n✅ E8 lattice comparison complete!")
    else:
        print(f"\n⚠️  E8 lattice had issues, but Z2 fallback worked!")
