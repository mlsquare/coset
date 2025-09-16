# CoSet Installation Guide

This guide provides step-by-step instructions for installing and building CoSet with CUDA support.

## Prerequisites

### System Requirements
- **CUDA**: Version 11.0 or higher
- **Python**: 3.8 or higher
- **PyTorch**: 1.9.0 or higher
- **CMake**: 3.15 or higher
- **GCC**: 7.0 or higher (for CUDA compilation)

### Hardware Requirements
- **GPU**: NVIDIA GPU with Compute Capability 7.0 or higher
- **Memory**: At least 8GB GPU memory recommended
- **Storage**: At least 2GB free space

## Installation Steps

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Build CUDA Extensions

```bash
# Clone the repository
git clone https://github.com/mlsquare/coset.git
cd coset

# Build CUDA extensions
python setup.py build_ext --inplace
```

### 3. Install CoSet

```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```

## Verification

### Test Installation

```bash
# Run tests
python -m pytest tests/ -v

# Run example
python examples/mlp_example.py
```

### Check CUDA Support

```python
import torch
import coset

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Test basic functionality
from coset import LatticeConfig, LatticeType, LatticeQuantizer

config = LatticeConfig(type=LatticeType.HNLQ, lattice_dim=8)
quantizer = LatticeQuantizer(config)
print("CoSet installation successful!")
```

## Troubleshooting

### Common Issues

#### 1. CUDA Compilation Errors
```
Error: nvcc not found
```
**Solution**: Ensure CUDA toolkit is installed and `nvcc` is in PATH.

#### 2. PyTorch Version Mismatch
```
Error: PyTorch version incompatible
```
**Solution**: Install compatible PyTorch version:
```bash
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. Memory Issues
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size or use smaller models.

#### 4. Import Errors
```
Error: ModuleNotFoundError: No module named 'coset_cuda'
```
**Solution**: Rebuild CUDA extensions:
```bash
python setup.py clean --all
python setup.py build_ext --inplace
```

### Build Options

#### Custom CUDA Architecture
```bash
# For specific GPU architecture
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
python setup.py build_ext --inplace
```

#### Debug Build
```bash
# Build with debug information
python setup.py build_ext --inplace --debug
```

#### CPU-Only Build
```bash
# Build without CUDA support
export CUDA_VISIBLE_DEVICES=""
python setup.py build_ext --inplace
```

## Performance Optimization

### CUDA Optimization
- Use latest CUDA toolkit for best performance
- Enable mixed precision training with `torch.cuda.amp`
- Use appropriate batch sizes for your GPU memory

### Memory Optimization
- Use gradient checkpointing for large models
- Enable gradient compression for distributed training
- Use appropriate quantization depths

## Development Setup

### For Contributors

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run linting
black coset/
flake8 coset/
mypy coset/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html
```

## Docker Installation

### Using Docker

```bash
# Build Docker image
docker build -t coset .

# Run container
docker run --gpus all -it coset
```

### Dockerfile Example

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip cmake build-essential

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CoSet
COPY . /coset
WORKDIR /coset
RUN pip3 install -e .

# Test installation
RUN python3 -c "import coset; print('CoSet installed successfully!')"
```

## Support

For installation issues:
1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed error information

For questions:
- GitHub Discussions: [Link to discussions]
- Email: [Contact email]
