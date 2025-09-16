#!/usr/bin/env python3
"""
Simple Quantized vs Standard Matrix Multiplication Comparison

This script provides a simple comparison without plotting dependencies.
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


def simple_matmul_comparison():
    """Simple comparison of quantized vs standard matrix multiplication."""
    print("🔬 Simple Quantized vs Standard Matrix Multiplication Comparison")
    print("=" * 70)
    
    # Create test matrices (compatible with Z2 lattice dimension = 2)
    torch.manual_seed(42)
    A = torch.randn(4, 8) * 0.5   # Input matrix (8 = 4 * 2, divisible by lattice_dim)
    B = torch.randn(6, 8) * 0.3   # Weight matrix (8 = 4 * 2, divisible by lattice_dim)
    bias = torch.randn(6) * 0.1   # Bias vector
    
    print(f"Input matrix A: {A.shape}")
    print(f"Weight matrix B: {B.shape}")
    print(f"Bias vector: {bias.shape}")
    
    # Standard matrix multiplication
    C_standard = torch.matmul(A, B.t()) + bias
    
    # Create quantizer (using Z2 for now due to E8 implementation issue)
    config = LatticeConfig(type=LatticeType.Z2, radix=4, num_layers=3)
    quantizer = LatticeQuantizer(config)
    
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


if __name__ == "__main__":
    results = simple_matmul_comparison()
    print(f"\n✅ Comparison complete!")
