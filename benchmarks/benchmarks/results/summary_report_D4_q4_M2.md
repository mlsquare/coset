# Performance Profiling Summary Report

**Configuration:** D4 lattice, q=4, M=2

**Generated:** 2025-09-17 13:19:32

## Overview

This report compares the performance of baseline PyTorch implementations against CUDA-optimized versions for three key processes:

1. **Encoding**: Converting vectors to hierarchical lattice encodings
2. **Decoding**: Converting encodings back to vectors
3. **Combined**: Complete quantization process (encode + decode)

## Results Summary

### Encoding

- **Batch sizes tested:** [1, 10, 100, 1000]
- **Average speedup:** 2274.85x
- **Maximum speedup:** 8164.30x
- **Peak throughput:** 7162843 vectors/sec

| Batch Size | Baseline Time (ms) | CUDA Time (ms) | Speedup | Throughput (vec/s) |
|------------|-------------------|----------------|---------|-------------------|
|          1 |              1.26 |           0.19 |    6.80 |              5397 |
|         10 |             11.51 |           0.14 |   82.24 |             71438 |
|        100 |            115.04 |           0.14 |  846.05 |            735438 |
|       1000 |           1139.81 |           0.14 | 8164.30 |           7162843 |

### Decoding

- **Batch sizes tested:** [1, 10, 100, 1000]
- **Average speedup:** 0.97x
- **Maximum speedup:** 1.04x
- **Peak throughput:** 1261 vectors/sec

| Batch Size | Baseline Time (ms) | CUDA Time (ms) | Speedup | Throughput (vec/s) |
|------------|-------------------|----------------|---------|-------------------|
|          1 |              0.84 |           0.95 |    0.89 |              1053 |
|         10 |              8.28 |           7.93 |    1.04 |              1261 |
|        100 |             76.92 |          81.57 |    0.94 |              1226 |
|       1000 |            788.91 |         796.92 |    0.99 |              1255 |

### Combined

- **Batch sizes tested:** [1, 10, 100, 1000]
- **Average speedup:** 4130.33x
- **Maximum speedup:** 14768.70x
- **Peak throughput:** 7352311 vectors/sec

| Batch Size | Baseline Time (ms) | CUDA Time (ms) | Speedup | Throughput (vec/s) |
|------------|-------------------|----------------|---------|-------------------|
|          1 |              2.61 |           0.15 |   16.99 |              6512 |
|         10 |             21.99 |           0.13 |  173.02 |             78679 |
|        100 |            198.98 |           0.13 | 1562.62 |            785302 |
|       1000 |           2008.72 |           0.14 | 14768.70 |           7352311 |

## Key Findings

- **Overall average speedup:** 2135.38x
- **Best speedup achieved:** 14768.70x
- **Peak throughput achieved:** 7352311 vectors/sec

## Recommendations

Based on the profiling results:

- ✅ CUDA optimization shows significant performance gains
- ✅ Peak throughput meets target performance (>100K vectors/sec)
- Consider further optimization for larger batch sizes
- Monitor memory usage and bandwidth utilization
- Validate numerical accuracy of CUDA implementations

## Files Generated

- `comprehensive_profile_*.png`: Combined performance plots
- `*_results_*.json`: Detailed timing data
- `*_summary_*.csv`: Summary statistics
- `summary_report_*.md`: This report

