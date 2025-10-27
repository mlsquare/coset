# Optimized Quantization Examples

This directory contains examples showcasing the optimized quantization implementations from `coset.optim`, including E8 lattice quantization and scalar quantization.

## Examples

### 1. MNIST CPU vs GPU Comparison (`mnist_cpu_gpu_comparison.py`)

Comprehensive MNIST Quantization-Aware Training (QAT) example comparing:
- **CPU**: PyTorch CPU implementation
- **GPU (PyTorch)**: PyTorch GPU implementation with vectorized operations  
- **CUDA-Accelerated**: Custom CUDA kernels (when available)

**Features:**
- E8 lattice quantization with batch size 128
- Performance benchmarking (forward pass speed)
- Training accuracy comparison
- Automatic scaling for optimal quantization

**Running:**
```bash
cd /workspace/coset
python3 examples/optim/mnist_cpu_gpu_comparison.py
```

### 2. E8 GPU Benchmark (`e8_gpu_benchmark.py`)

Performance benchmarking of E8 lattice quantization comparing CPU vs GPU execution.

**Features:**
- Batch quantization performance
- Memory usage analysis
- Speedup measurements

**Running:**
```bash
cd /workspace/coset
python3 examples/optim/e8_gpu_benchmark.py
```

### 3. E8 GPU Usage (`e8_gpu_usage.py`)

Basic usage examples of E8 GPU quantization functions.

**Features:**
- Simple quantization examples
- Batch processing demonstrations
- Error analysis

**Running:**
```bash
cd /workspace/coset
python3 examples/optim/e8_gpu_usage.py
```

### 4. BERT Binary Classification (`bert_binary_classification.py`)

BERT-based text binary classification using E8 quantization for MLP layers while keeping the final output layer unquantized.

**Features:**
- Pre-trained BERT embeddings (frozen)
- E8-quantized MLP layers for feature processing
- Unquantized final linear layer (single output)
- Text binary classification task
- Performance comparison: Standard vs Quantized vs CUDA

**Running:**
```bash
cd /workspace/coset
python3 examples/optim/bert_binary_classification.py
```

### 5. Scalar Quantization Comparison (`scalar_comparison.py`)

Comprehensive comparison of different scalar quantization methods on MNIST, including symmetric/asymmetric modes and different bit-widths.

**Features:**
- Standard MLP (no quantization baseline)
- Scalar 4-bit symmetric quantization
- Scalar 4-bit asymmetric quantization  
- Scalar 8-bit symmetric quantization
- Scalar 8-bit asymmetric quantization
- E8 lattice quantization (reference)
- Performance and accuracy trade-off analysis
- Block-based and row-wise quantization support

**Running:**
```bash
cd /workspace/coset
python3 examples/optim/scalar_comparison.py
```

## Optimized Implementations

These examples use the optimized implementations from `coset.optim` which include:

### E8 Lattice Quantization (`coset.optim.e8`)
- **E8Config**: E8-specific configuration with optimal beta values
- **E8QLinear**: E8-optimized linear layer for QAT
- **batch_e8_quantize**: Vectorized E8 quantization
- **CUDA Kernels**: JIT-compiled CUDA acceleration (when available)

### Scalar Quantization (`coset.optim.scalar`)
- **ScalarConfig**: Configurable scalar quantization with q^M bit-widths
- **ScalarQLinear**: Scalar quantized linear layer for QAT
- **Symmetric/Asymmetric**: Both quantization modes supported
- **Block-based**: Configurable block sizes (4, 8, or row-wise)
- **Per-row scaling**: L2 norm scaling for stable quantization

## Performance Expectations

- **CPU**: ~2-3ms per epoch for MNIST
- **GPU (PyTorch)**: ~1-2ms per epoch (1.5-2x speedup)
- **CUDA-Accelerated**: Additional 1.5-2x speedup over GPU PyTorch
- **Accuracy**: 93-94% on MNIST (minimal quantization loss)

## Requirements

- PyTorch with CUDA support (for GPU examples)
- CUDA toolkit (for CUDA-accelerated examples)
- COSET library with E8 optimization modules
