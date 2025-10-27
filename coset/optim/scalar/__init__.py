"""
Scalar Quantization Optimized Module

This module provides optimized implementations for scalar quantization,
including configuration, quantizers, layers, and CUDA acceleration.
"""

from .config import (
    ScalarConfig, 
    get_scalar_config,
    SCALAR_INT4_SYM, 
    SCALAR_INT8_SYM,
    SCALAR_INT4_ASYM, 
    SCALAR_INT8_ASYM,
    SCALAR_INT4_SYM_BLOCK4,
    SCALAR_INT8_SYM_BLOCK4,
    SCALAR_INT4_ASYM_BLOCK4,
    SCALAR_INT8_ASYM_BLOCK4,
    SCALAR_INT4_SYM_ROW,
    SCALAR_INT8_SYM_ROW,
    SCALAR_INT4_ASYM_ROW,
    SCALAR_INT8_ASYM_ROW,
    SCALAR_INT4_SYM_MATRIX,
    SCALAR_INT8_SYM_MATRIX,
    SCALAR_INT4_ASYM_MATRIX,
    SCALAR_INT8_ASYM_MATRIX,
)
from .quantizers import (
    scalar_quantize,
    batch_scalar_quantize,
    scalar_dequantize,
    symmetric_quantize,
    asymmetric_quantize,
    matrix_level_quantize,
)
from .layers import ScalarQLinear, ScalarStraightThroughQuantize
from .cuda import (
    scalar_quantize_cuda_jit,
    scalar_cuda_available,
    scalar_quantize_cuda_wrapper,
)

__all__ = [
    # Configuration
    'ScalarConfig',
    'get_scalar_config',
    'SCALAR_INT4_SYM',
    'SCALAR_INT8_SYM',
    'SCALAR_INT4_ASYM',
    'SCALAR_INT8_ASYM',
    'SCALAR_INT4_SYM_BLOCK4',
    'SCALAR_INT8_SYM_BLOCK4',
    'SCALAR_INT4_ASYM_BLOCK4',
    'SCALAR_INT8_ASYM_BLOCK4',
    'SCALAR_INT4_SYM_ROW',
    'SCALAR_INT8_SYM_ROW',
    'SCALAR_INT4_ASYM_ROW',
    'SCALAR_INT8_ASYM_ROW',
    'SCALAR_INT4_SYM_MATRIX',
    'SCALAR_INT8_SYM_MATRIX',
    'SCALAR_INT4_ASYM_MATRIX',
    'SCALAR_INT8_ASYM_MATRIX',
    
    # Quantizers
    'scalar_quantize',
    'batch_scalar_quantize',
    'scalar_dequantize',
    'symmetric_quantize',
    'asymmetric_quantize',
    'matrix_level_quantize',
    
    # Layers
    'ScalarQLinear',
    'ScalarStraightThroughQuantize',
    
    # CUDA
    'scalar_quantize_cuda_jit',
    'scalar_cuda_available',
    'scalar_quantize_cuda_wrapper',
]
