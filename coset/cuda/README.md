# CUDA vLUT Operations

This directory contains CUDA-accelerated implementations of Value Lookup Table (vLUT) operations for hierarchical nested-lattice quantization (HNLQ).

## Current Status

**All CUDA implementations have been removed for a fresh start.** The directory now contains only the testing framework and documentation.

## Remaining Files

- **test_foundation.py**: Simulation system validation and foundation testing
- **test_vlut.py**: Framework for testing vLUT implementations (currently only PyTorch and original vLUT)
- **performance_analysis.md**: Performance analysis documentation
- **README.md**: This file

## Available Implementations

Currently only these implementations are available:
1. **PyTorch CPU/GPU**: Reference implementations for accuracy verification
2. **Original vLUT Manager**: Basic vLUT manager from `coset.quant.vlut`

## Testing

Run the current tests:
```bash
python coset/cuda/test_foundation.py    # Foundation validation
python coset/cuda/test_vlut.py          # vLUT implementation testing
```

## Next Steps

Ready to implement new CUDA vLUT operations from scratch with:
- Clean architecture
- Proper error handling
- Comprehensive testing
- Performance optimization

## Previous Work (Removed)

The following implementations were previously available but have been removed for a fresh start:
- One-sided vLUT operations (optimized and ultra-optimized)
- Two-sided vLUT operations (optimized and ultra-optimized)
- Various CUDA kernel implementations
- Neural network layers with vLUT operations
- Comprehensive benchmarking tools

All performance analysis and optimization insights are preserved in the documentation files.