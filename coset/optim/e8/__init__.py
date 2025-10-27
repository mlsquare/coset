"""
E8 Lattice Optimized Module

This module provides optimized implementations specifically for E8 lattice quantization,
including configuration, codecs, layers, lookup tables, and CUDA acceleration.
"""

from .config import E8Config, get_e8_config, E8_FAST, E8_ACCURATE, E8_HIGH_COMPRESSION, E8_LOW_DISTORTION
from .codecs import (
    e8_quantize, 
    batch_e8_quantize, 
    e8_encode, 
    e8_decode, 
    batch_e8_encode, 
    batch_e8_decode
)
from .layers import E8QLinear, E8StraightThroughQuantize
from .lut import E8OneSidedLUT, E8TwoSidedLUT, E8LUTManager
from .cuda import e8_quantize_cuda_jit, e8_cuda_available, e8_quantize_cuda_wrapper

__all__ = [
    # Configuration
    'E8Config',
    'get_e8_config',
    'E8_FAST',
    'E8_ACCURATE', 
    'E8_HIGH_COMPRESSION',
    'E8_LOW_DISTORTION',
    
    # Codecs
    'e8_quantize',
    'batch_e8_quantize',
    'e8_encode',
    'e8_decode',
    'batch_e8_encode',
    'batch_e8_decode',
    
    # Layers
    'E8QLinear',
    'E8StraightThroughQuantize',
    
    # Lookup Tables
    'E8OneSidedLUT',
    'E8TwoSidedLUT',
    'E8LUTManager',
    
    # CUDA
    'e8_quantize_cuda_jit',
    'e8_cuda_available',
    'e8_quantize_cuda_wrapper',
]
