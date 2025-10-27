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

# Optimized lattice modules
from .optim import (
    # E8 optimized
    E8Config,
    get_e8_config,
    E8_FAST,
    E8_ACCURATE,
    E8_HIGH_COMPRESSION,
    E8_LOW_DISTORTION,
    e8_quantize,
    batch_e8_quantize,
    e8_encode,
    e8_decode,
    batch_e8_encode,
    batch_e8_decode,
    E8QLinear,
    E8StraightThroughQuantize,
    E8OneSidedLUT,
    E8TwoSidedLUT,
    E8LUTManager,
    e8_quantize_cuda_jit,
    e8_cuda_available,
    e8_quantize_cuda_wrapper,
)

__all__ = [
    # Core lattice classes
    "Lattice",
    "Z2Lattice", 
    "D4Lattice",
    "E8Lattice",
    
    # Generic quantization
    "QuantizationConfig",
    "encode",
    "decode", 
    "quantize",
    "QLinear",
    
    # E8 optimized
    "E8Config",
    "get_e8_config",
    "E8_FAST",
    "E8_ACCURATE",
    "E8_HIGH_COMPRESSION",
    "E8_LOW_DISTORTION",
    "e8_quantize",
    "batch_e8_quantize",
    "e8_encode",
    "e8_decode",
    "batch_e8_encode",
    "batch_e8_decode",
    "E8QLinear",
    "E8StraightThroughQuantize",
    "E8OneSidedLUT",
    "E8TwoSidedLUT",
    "E8LUTManager",
    "e8_quantize_cuda_jit",
    "e8_cuda_available",
    "e8_quantize_cuda_wrapper",
]
