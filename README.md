# Coset: Hierarchical Nested-Lattice Quantization for PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/coset/coset/workflows/Tests/badge.svg)](https://github.com/coset/coset/actions)

A high-performance PyTorch library implementing **Hierarchical Nested-Lattice Quantization (HNLQ)** for quantization-aware training (QAT) and distributed training optimization.

## Features

- **Multi-lattice Support**: Z², D₄, E₈ lattices with optimized nearest-neighbor algorithms
- **Hierarchical Encoding/Decoding**: M-level quantization with successive refinement
- **CUDA Acceleration**: Custom kernels for encode/decode operations with up to 332+ billion ops/sec
- **QAT Integration**: PyTorch modules for quantization-aware training
- **Distributed Training**: Gradient compression hooks for DDP (coming soon)
- **Value Lookup Tables (vLUT)**: Ultra-optimized inner product computation with 28.34x speedup over PyTorch
- **Overload Handling**: Automatic scaling with geometric progression

## Installation

```bash
# Install from source
git clone https://github.com/coset/coset.git
cd coset
pip install -e .

# Install with CUDA support
pip install -e ".[cuda]"

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Quantization

```python
import torch
from coset import D4Lattice, QuantizationConfig, encode, decode

# Create lattice and configuration
lattice = D4Lattice()
config = QuantizationConfig(lattice_type="D4", q=4, M=2)

# Quantize a vector
x = torch.randn(4)
b, T = encode(x, lattice, config)
x_reconstructed = decode(b, lattice, config, T)

print(f"Original: {x}")
print(f"Reconstructed: {x_reconstructed}")
print(f"Error: {torch.norm(x - x_reconstructed)}")
```

### Quantization-Aware Training

```python
import torch
import torch.nn as nn
from coset import QLinear, QuantizationConfig

# Create a quantized linear layer
config = QuantizationConfig(lattice_type="D4", q=4, M=2)
layer = QLinear(128, 256, config, quantize_weights=True)

# Use in your model
x = torch.randn(32, 128)
output = layer(x)
print(f"Output shape: {output.shape}")
```

### Complete Example: MNIST QAT

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from coset import QLinear, QuantizationConfig

# Define quantized model
class QuantizedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        config = QuantizationConfig(lattice_type="D4", q=4, M=2)
        self.fc1 = QLinear(784, 128, config, quantize_weights=True)
        self.fc2 = QLinear(128, 64, config, quantize_weights=True)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training loop
model = QuantizedMNIST()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Your training code here...
```

### Ultra-Optimized vLUT Operations

```python
import torch
from coset.cuda.test_optimized_kernels_v2 import create_vectorized_vlut_operations
from coset.lattices.e8 import E8Lattice

# Create E8 lattice and configuration
lattice = E8Lattice()
config = E8Config(q=3, M=2)

# Create ultra-optimized vLUT operations
operations = create_vectorized_vlut_operations(lattice, config)

# Generate test data
batch_size = 1000
input_encodings = torch.randint(0, 3, (batch_size, 8), device='cuda', dtype=torch.float32)
query_vector = torch.randn(8, device='cuda', dtype=torch.float32)

# Build vLUT and perform dot product
vlut = operations.build_vlut(query_vector)
results = operations.dot_product(input_encodings, vlut)

print(f"vLUT dot product results: {results.shape}")
print(f"Performance: Up to 332+ billion operations/second")
```

## API Reference

### Lattices

- `Z2Lattice`: 2D integer lattice (baseline)
- `D4Lattice`: 4D checkerboard lattice (recommended)
- `E8Lattice`: 8D optimal lattice (high precision)

### Quantization

- `QuantizationConfig`: Configuration for quantization parameters
- `encode()`: Hierarchical encoding (Algorithm 1)
- `decode()`: Hierarchical decoding (Algorithm 2)
- `quantize()`: Complete quantization (encode + decode)

### Neural Networks

- `QLinear`: Quantized linear layer with HNLQ

### Value Lookup Tables (vLUT)

- **Ultra-Optimized vLUT Operations**: High-performance quantized inner product computation
- **One-Sided vLUT**: Single quantization with unquantized queries
- **Two-Sided vLUT**: Dual quantization for both inputs and queries
- **Performance**: Up to 332+ billion operations/second, 28.34x speedup over PyTorch

## Configuration

```python
config = QuantizationConfig(
    lattice_type="D4",           # Lattice type: "Z2", "D4", "E8"
    q=4,                         # Quantization parameter (alphabet size)
    M=2,                         # Number of hierarchical levels
    beta=1.0,                    # Scaling parameter
    alpha=1.0,                   # Overload scaling parameter
    max_scaling_iterations=10,   # Maximum scaling iterations
    with_tie_dither=True,        # Use tie-breaking dither
    with_dither=False,           # Use randomized dither
    disable_scaling=False,       # Disable beta scaling (performance)
    disable_overload_protection=False,  # Disable overload protection (performance)
)
```

## Performance

### Quantization Performance
- **Encoding**: >100K vectors/sec (D₄, q=4, M=2)
- **Decoding**: >200K vectors/sec
- **Memory**: 4-8x compression ratio
- **QAT Overhead**: <5x slower than FP32

### vLUT Performance
- **Maximum Throughput**: 332+ billion operations/second
- **Average Speedup**: 28.34x over PyTorch native operations
- **Best Performance**: Original Optimized v2 kernels
- **Scalability**: Tested up to 50K batch size with 200 queries

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=coset

# Format code
black coset tests
ruff check coset tests

# Type checking
mypy coset
```

## Roadmap

- [x] Core lattice implementations (Z², D₄, E₈)
- [x] Basic encoding/decoding algorithms
- [x] Quantization-aware training modules
- [ ] CUDA acceleration kernels
- [ ] Lookup table optimization
- [ ] Distributed training hooks
- [ ] Advanced quantization strategies

## Citation

If you use this library in your research, please cite:

```bibtex
@article{kaplan2025high,
  title={High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables},
  author={Kaplan, Haim and Ordentlich, Or},
  journal={arXiv preprint arXiv:2505.13164},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- Based on the work of Kaplan & Ordentlich (2025)
- Inspired by the PyTorch ecosystem
- Built with the scientific Python community
