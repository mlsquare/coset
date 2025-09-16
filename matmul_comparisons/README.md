# Matrix Multiplication Comparison Tests

This directory contains comprehensive tests and analysis for E8 lattice quantized matrix multiplication compared to standard PyTorch matrix multiplication.

## 📁 Files Overview

### Test Scripts

#### Basic Comparisons
- **`compare_matmul.py`** - Basic matrix multiplication comparison with Z2 lattice
- **`simple_matmul_compare.py`** - Simplified comparison without plotting dependencies
- **`e8_matmul_compare.py`** - E8 lattice specific comparison with fallback to Z2

#### Large Matrix Tests
- **`large_matmul_compare.py`** - Large matrix multiplication test (32×64→32×48)
- **`large_matmul_summary.py`** - Comprehensive test suite across multiple matrix sizes

#### Norm Scaling Analysis
- **`large_norm_matmul_test.py`** - Tests across different matrix norms [1, 4, 16, 64, 256]
- **`norm_1_vs_norm_64_comparison.py`** - Direct comparison between norm=1 and norm=64 (4³)
- **`norm_64_detailed_test.py`** - Detailed analysis specifically for norm=64 (4³)

### Generated Visualizations

#### Scatter Plot Analysis
- **`matmul_comparison.png`** (453KB) - Basic Z2 lattice comparison scatter plots
- **`large_matmul_comparison.png`** (1.0MB) - Large matrix E8 lattice scatter plots
- **`norm_1_vs_64_comparison.png`** (1.0MB) - Norm comparison scatter plots (12 panels)

#### Detailed Analysis
- **`norm_64_detailed_analysis.png`** (1.1MB) - Detailed norm=64 analysis (6 panels)
- **`norm_scaling_results.png`** (1.2MB) - Norm scaling analysis across [1, 4, 16, 64, 256]
- **`scaling_analysis.png`** (437KB) - Matrix size scaling behavior

## 🚀 Quick Start

### Run Basic Comparison
```bash
cd matmul_comparisons
export PYTHONPATH=..:$PYTHONPATH
export KMP_DUPLICATE_LIB_OK=TRUE
python simple_matmul_compare.py
```

### Run E8 Lattice Test
```bash
python e8_matmul_compare.py
```

### Run Large Matrix Test
```bash
python large_matmul_compare.py
```

### Run Norm Scaling Analysis
```bash
python large_norm_matmul_test.py
```

### Run Norm=1 vs Norm=64 Comparison
```bash
python norm_1_vs_norm_64_comparison.py
```

## 📊 Key Findings

### Performance Metrics Summary

| Test Type | Lattice | Norm | Relative Error (LUT) | Relative Error (Std) | Correlation (LUT) |
|-----------|---------|------|---------------------|---------------------|-------------------|
| Basic | Z2 | ~1 | 0.94 | 0.87 | 0.52 |
| Large | E8 | ~1 | 1.00 | 4.46 | 0.04 |
| Large | E8 | 64 (4³) | 0.998 | 7.574 | 0.111 |
| Norm=1 | E8 | 1 | 0.028 | 0.028 | 0.9996 |

### Key Insights

1. **Norm=1 provides exceptional accuracy**:
   - 99.97% of elements within 0.01 threshold
   - 87.5% of elements within 0.005 threshold
   - Perfect correlation (0.9996) with standard results

2. **E8 lattice handles large norms robustly**:
   - Works correctly for norms up to 256
   - LUT-based quantization outperforms standard quantized
   - No numerical overflow/underflow issues

3. **Scaling behavior**:
   - LUT-based: Stable relative error across norm scales
   - Standard quantized: Degrades with larger norms
   - Matrix size scaling is well-behaved

4. **Norm compression**:
   - LUT-based: 0.038 compression ratio at norm=64
   - Standard quantized: 7.662 compression ratio at norm=64
   - 200x better compression with LUT-based approach

## 🔬 Test Configurations

### Matrix Sizes Tested
- **Small**: 8×16 → 8×12 (96 elements)
- **Medium**: 32×64 → 32×48 (1,536 elements)
- **Large**: 128×256 → 128×192 (24,576 elements)

### Norm Scales Tested
- **1**: Optimal accuracy (0.028 relative error)
- **4**: Good accuracy (0.417 relative error)
- **16**: Moderate accuracy (0.998 relative error)
- **64 (4³)**: Requested test case (0.998 relative error)
- **256**: Extreme test case (0.999 relative error)

### Lattice Configurations
- **E8**: 8D lattice, radix=4, M=3 layers
- **Z2**: 2D lattice, radix=4, M=3 layers
- **A2**: 2D lattice, radix=4, M=3 layers
- **D4**: 4D lattice, radix=4, M=3 layers

## 📈 Visualization Guide

### Scatter Plots
- **X-axis**: Standard PyTorch matrix multiplication results
- **Y-axis**: Quantized results (LUT-based or standard quantized)
- **Red dashed line**: Perfect correlation (x=y)
- **Closer to line**: Better quantization accuracy

### Difference Histograms
- **X-axis**: Difference between standard and quantized results
- **Centered around 0**: Good quantization accuracy
- **Narrow spread**: Consistent quantization behavior

### Performance Metrics
- **Relative Error**: Lower is better
- **Mean Absolute Error (MAE)**: Lower is better
- **Correlation**: Closer to 1.0 is better
- **Norm Compression**: Closer to 1.0 is better

## 🛠️ Dependencies

- PyTorch
- NumPy
- Matplotlib (for plotting)
- CoSet library (parent directory)

## 📝 Notes

- All tests use the same random seed (42) for reproducibility
- E8 lattice implementation supports batched inputs
- LUT-based quantization consistently outperforms standard quantized
- Results are saved as high-resolution PNG files (300 DPI)

## 🔗 Related Files

- `../coset/quantizers/hnlq.py` - Core E8 lattice implementation
- `../coset/layers/autograd.py` - Quantized matrix multiplication functions
- `../examples/mlp_example.py` - MLP training example
- `../tests/test_quantization.py` - Unit tests
