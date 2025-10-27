"""
Quantization module for hierarchical nested-lattice quantization.

This module provides the core quantization functionality including:
- Configuration management
- Encoding and decoding functions
- Lookup table operations
- Overload handling
"""

from .params import QuantizationConfig
from .functional import encode, decode, quantize, batch_quantize, mac_modq, accumulate_modq
from .vlut import vLUTManager, build_vlut, vlut_mac_operation, vlut_accumulate_operation
from .elut import eLUTManager, build_elut, elut_mac_operation, elut_accumulate_operation
from .xac import mac, aac, mac_with_dither, mac_with_scaling, adaptive_mac, batch_mac, validate_operations
from .e8_gpu import batch_e8_quantize, batch_encode_e8, batch_decode_e8, batch_quantize_e8
from .e8_cuda_ops import (
    is_cuda_available, 
    e8_quantize_cuda, 
    e8_encode_cuda, 
    e8_decode_cuda,
    e8_quantize_cuda_wrapper
)

__all__ = [
    "QuantizationConfig",
    "encode", 
    "decode",
    "quantize",
    "batch_quantize",
    "mac_modq",
    "accumulate_modq",
    "vLUTManager",
    "build_vlut",
    "vlut_mac_operation",
    "vlut_accumulate_operation",
    "eLUTManager",
    "build_elut",
    "elut_mac_operation",
    "elut_accumulate_operation",
    "mac",
    "aac",
    "mac_with_dither",
    "mac_with_scaling",
    "adaptive_mac",
    "batch_mac",
    "validate_operations",
    "batch_e8_quantize",
    "batch_encode_e8",
    "batch_decode_e8",
    "batch_quantize_e8",
    "is_cuda_available",
    "e8_quantize_cuda",
    "e8_encode_cuda",
    "e8_decode_cuda",
    "e8_quantize_cuda_wrapper",
]
