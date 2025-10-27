"""
Optimized Lattice Modules

This package provides lattice-specific optimized implementations for different
lattice types (E8, D4, etc.) with specialized codecs, layers, and CUDA acceleration.
"""

# E8 optimized module
from .e8 import (
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

# TODO: Add D4 optimized module when implemented
# from .d4 import (
#     D4Config,
#     get_d4_config,
#     d4_quantize,
#     batch_d4_quantize,
#     d4_encode,
#     d4_decode,
#     batch_d4_encode,
#     batch_d4_decode,
#     D4QLinear,
#     D4StraightThroughQuantize,
#     D4OneSidedLUT,
#     D4TwoSidedLUT,
#     D4LUTManager,
# )

__all__ = [
    # E8 exports
    'E8Config',
    'get_e8_config',
    'E8_FAST',
    'E8_ACCURATE',
    'E8_HIGH_COMPRESSION',
    'E8_LOW_DISTORTION',
    'e8_quantize',
    'batch_e8_quantize',
    'e8_encode',
    'e8_decode',
    'batch_e8_encode',
    'batch_e8_decode',
    'E8QLinear',
    'E8StraightThroughQuantize',
    'E8OneSidedLUT',
    'E8TwoSidedLUT',
    'E8LUTManager',
    'e8_quantize_cuda_jit',
    'e8_cuda_available',
    'e8_quantize_cuda_wrapper',
    
    # TODO: Add D4 exports when implemented
]
