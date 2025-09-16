"""
CoSet: Hierarchical Nested Lattice Quantization for Matrix Operations
"""

__version__ = "0.1.0"

from .quantizers import LatticeQuantizer, LatticeConfig, LatticeType
from .layers import QuantizedLinear, QuantizedMLP
from .distributed import QuantizedGradientHook

__all__ = [
    "LatticeQuantizer",
    "LatticeConfig", 
    "LatticeType",
    "QuantizedLinear",
    "QuantizedMLP", 
    "QuantizedGradientHook",
]
