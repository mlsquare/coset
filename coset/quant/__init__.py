"""
Quantization module for hierarchical nested-lattice quantization.

This module provides the core quantization functionality including:
- Configuration management
- Encoding and decoding functions
- Lookup table operations
- Overload handling
"""

from .params import QuantizationConfig
from .functional import encode, decode, quantize, mac_modq, accumulate_modq

__all__ = [
    "QuantizationConfig",
    "encode", 
    "decode",
    "quantize",
    "mac_modq",
    "accumulate_modq",
]
