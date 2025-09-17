"""
Coset: Hierarchical Nested-Lattice Quantization for PyTorch

A high-performance PyTorch library implementing Hierarchical Nested-Lattice Quantization (HNLQ)
for quantization-aware training (QAT) and distributed training optimization.

Key Features:
- Multi-lattice support (Z², D₄, E₈)
- Hierarchical encoding/decoding with M levels
- CUDA acceleration
- QAT integration
- Distributed training optimization
- Lookup table-based inner products
"""

__version__ = "0.1.0"
__author__ = "Coset Development Team"

# Core imports
from .lattices import Lattice, Z2Lattice, D4Lattice, E8Lattice
from .quant import QuantizationConfig, encode, decode, quantize
from .nn import QLinear

__all__ = [
    "Lattice",
    "Z2Lattice", 
    "D4Lattice",
    "E8Lattice",
    "QuantizationConfig",
    "encode",
    "decode", 
    "quantize",
    "QLinear",
]
