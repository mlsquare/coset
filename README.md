# Coset: Hierarchical Nested-Lattice Quantization for PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance PyTorch library implementing **Hierarchical Nested-Lattice Quantization (HNLQ)** for quantization-aware training (QAT) with transformer models.

## Features

- **E8 and D4 Lattice Support**: High-dimensional E8 and D4 lattice quantization with optimized algorithms
- **Transformer Integration**: Pre-trained BERT with quantized classification heads
- **QAT with Cold Start**: Gradual quantization activation for stable training
- **Constructor-Based API**: Easy-to-use layer constructors for different lattices
- **Flexible Scale Parameters**: Learnable or fixed scale parameters for quantization
- **Comprehensive Examples**: Binary and multi-class classification examples
- **Future Support**: other lattice types will be added

## Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/coset/coset.git
cd coset

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install transformers torchvision scikit-learn matplotlib
```

### Manual Installation

```bash
# Install core package
pip install -e .

# Install example dependencies
pip install transformers torchvision scikit-learn matplotlib

# For CUDA support (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Usage Experiments
All experiments using this repo are present in https://github.com/ResearchCyAI/LatticeBasedQuantization.git
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
- Special thanks to the Hugging Face transformers library