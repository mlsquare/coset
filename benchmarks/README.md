# Performance Profiling Scripts

This directory contains comprehensive performance profiling scripts for the Coset library, comparing baseline PyTorch implementations against CUDA-optimized versions.

## Overview

The profiling suite consists of three main profilers:

1. **Encoding Profiler** (`profile_encoding.py`) - Profiles the encoding process
2. **Decoding Profiler** (`profile_decoding.py`) - Profiles the decoding process  
3. **Combined Profiler** (`profile_combined.py`) - Profiles the complete quantization process (encode + decode)

## Quick Start

### Test the Profilers

First, run the test script to ensure everything works:

```bash
cd benchmarks
python test_profilers.py
```

### Run Individual Profilers

```bash
# Profile encoding only
python profile_encoding.py --lattice D4 --q 4 --M 2 --batch-sizes 1,10,100,1000

# Profile decoding only  
python profile_decoding.py --lattice D4 --q 4 --M 2 --batch-sizes 1,10,100,1000

# Profile combined quantization
python profile_combined.py --lattice D4 --q 4 --M 2 --batch-sizes 1,10,100,1000
```

### Run Complete Analysis

```bash
# Run all profilers and generate comprehensive report
python profile_all.py --lattice D4 --q 4 --M 2 --batch-sizes 1,10,100,1000,10000
```

## Command Line Options

All profilers support the following options:

- `--lattice`: Lattice type (`Z2`, `D4`, `E8`) - default: `D4`
- `--q`: Quantization parameter - default: `4`
- `--M`: Number of hierarchical levels - default: `2`
- `--batch-sizes`: Comma-separated list of batch sizes - default: `1,10,100,1000,10000`
- `--output-dir`: Output directory for results - default: `benchmarks/results`
- `--no-plot`: Skip generating plots

## Output Files

Each profiler generates:

- **JSON results**: Detailed timing data (`*_results_*.json`)
- **CSV summary**: Summary statistics (`*_summary_*.csv`)
- **Plots**: Performance visualizations (`*_profile_*.png`)

The master profiler (`profile_all.py`) additionally generates:

- **Comprehensive plot**: Combined comparison of all processes
- **Summary report**: Markdown report with analysis and recommendations

## Understanding the Results

### Baseline vs CUDA-Optimized

- **Baseline**: Current PyTorch implementation (CPU/GPU with PyTorch operations)
- **CUDA-Optimized**: Placeholder for future CUDA kernel implementations

Currently, the "CUDA-optimized" versions use vectorized PyTorch operations as a proxy for what actual CUDA kernels should achieve.

### Key Metrics

- **Execution Time**: Time to process a batch (milliseconds)
- **Throughput**: Vectors processed per second
- **Speedup**: Ratio of baseline time to CUDA time
- **Compression Ratio**: Memory savings from quantization

### Performance Targets

Based on the blueprint specifications:

- **Encoding**: >100K vectors/sec (D₄, q=4, M=2)
- **Decoding**: >200K vectors/sec
- **CUDA Speedup**: 10-30x over CPU reference
- **Memory Compression**: 4-8x reduction

## Example Usage

### Basic Profiling

```bash
# Quick test with small batches
python profile_encoding.py --batch-sizes 1,10,100

# Test different lattice types
python profile_encoding.py --lattice Z2 --batch-sizes 1,10,100,1000
python profile_encoding.py --lattice E8 --batch-sizes 1,10,100,1000
```

### Comprehensive Analysis

```bash
# Full analysis with all profilers
python profile_all.py --lattice D4 --q 4 --M 2 --batch-sizes 1,10,100,1000,10000

# Test different configurations
python profile_all.py --lattice D4 --q 8 --M 3 --batch-sizes 1,100,10000
```

### Custom Output Directory

```bash
# Save results to custom directory
python profile_all.py --output-dir my_results --batch-sizes 1,10,100,1000
```

## Interpreting Results

### Good Performance Indicators

- ✅ CUDA speedup > 2x over baseline
- ✅ Peak throughput > 100K vectors/sec
- ✅ Consistent performance across batch sizes
- ✅ Low variance in timing measurements

### Areas for Improvement

- ⚠️ CUDA speedup < 2x (may need better optimization)
- ⚠️ Throughput below targets (may need larger batches or better kernels)
- ⚠️ High variance in timing (may need more warmup iterations)

## Future CUDA Implementation

The current "CUDA-optimized" implementations are placeholders. Future work should:

1. **Implement actual CUDA kernels** for encoding, decoding, and combined operations
2. **Optimize memory access patterns** with coalesced reads/writes
3. **Use shared memory** for small lattice dimensions
4. **Fuse operations** to reduce kernel launch overhead
5. **Validate numerical accuracy** against baseline implementations

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root or have the correct Python path
2. **CUDA errors**: Check that PyTorch was installed with CUDA support
3. **Memory errors**: Reduce batch sizes if running out of GPU memory
4. **Plot errors**: Install matplotlib and seaborn: `pip install matplotlib seaborn`

### Getting Help

- Check the test script output for basic functionality
- Verify CUDA availability with `torch.cuda.is_available()`
- Run with smaller batch sizes to isolate issues
- Check the generated log files for detailed error messages

## Contributing

When adding new profilers or improving existing ones:

1. Follow the existing code structure and naming conventions
2. Add comprehensive error handling and validation
3. Include both JSON and CSV output formats
4. Generate informative plots with proper labels and legends
5. Update this README with new features or options
