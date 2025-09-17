# Analysis Directory

This directory contains comprehensive analysis and benchmarking tools for the CoSet library.

## 📁 Subdirectories

### `matmul_comparisons/`
Matrix multiplication comparison tests and analysis for E8 lattice quantized operations compared to standard PyTorch matrix multiplication.

**Contents:**
- **Test Scripts**: Norm scaling analysis and basic comparisons
- **Visualizations**: High-resolution plots showing quantization accuracy
- **Documentation**: Comprehensive usage guide and findings

**Key Features:**
- Norm scaling analysis across [1, 4, 16, 64, 256]
- E8 lattice performance evaluation
- LUT-based vs standard quantized comparison
- Detailed scatter plots and performance metrics

See `matmul_comparisons/README.md` for detailed usage instructions.

## 🚀 Quick Start

```bash
# Run matrix multiplication analysis
cd analysis/matmul_comparisons
export PYTHONPATH=../..:$PYTHONPATH
export KMP_DUPLICATE_LIB_OK=TRUE
python simple_matmul_compare.py
```

## 📊 Key Findings

- **Norm=1**: Exceptional accuracy (0.028 relative error, 0.9996 correlation)
- **Norm=64 (4³)**: LUT-based outperforms standard quantized (0.998 vs 7.574 relative error)
- **E8 Lattice**: Handles large norms robustly without numerical issues

## 🔗 Related Files

- `../coset/` - Core CoSet library implementation
- `../examples/` - Usage examples and tutorials
- `../tests/` - Unit tests and validation
