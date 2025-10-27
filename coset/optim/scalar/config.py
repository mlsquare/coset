"""
Scalar Quantization Configuration Module

This module provides configuration for scalar quantization with support for
symmetric and asymmetric quantization modes, configurable bit-widths via q^M
parameters, and block-based grouping.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math


@dataclass
class ScalarConfig:
    """
    Configuration class for scalar quantization.
    
    Supports both symmetric and asymmetric quantization with configurable
    bit-widths through q^M parameters and block-based grouping.
    
    Attributes:
        q: Base quantization levels (alphabet size)
        M: Hierarchical levels (effective bits = log2(q^M))
        mode: Quantization mode ("symmetric" or "asymmetric")
        block_size: Elements per block (4, 8, or None for whole row)
        per_row_scaling: Whether to apply per-row L2 norm scaling
        scale_factor: Optional manual scale factor override
        asymmetric_method: Method for asymmetric quantization ("zero_point" or "affine")
        with_dither: Whether to add dither for randomized quantization
        with_tie_dither: Whether to add dither to break ties
    """
    
    q: int = 4
    M: int = 2
    mode: str = "symmetric"  # "symmetric" or "asymmetric"
    block_size: int = 8  # Elements per block (4, 8, or None for whole row)
    per_row_scaling: bool = True  # Always enabled per requirement
    scale_factor: Optional[float] = None  # Manual scaling override
    asymmetric_method: str = "zero_point"  # "zero_point" or "affine"
    with_dither: bool = False
    with_tie_dither: bool = True
    matrix_level_quantization: bool = False  # Quantize whole matrix as one unit
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.mode not in ["symmetric", "asymmetric"]:
            raise ValueError(f"mode must be 'symmetric' or 'asymmetric', got {self.mode}")
        
        if self.asymmetric_method not in ["zero_point", "affine"]:
            raise ValueError(f"asymmetric_method must be 'zero_point' or 'affine', got {self.asymmetric_method}")
        
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
    
    def effective_bits(self, matrix_shape: Optional[Tuple[int, int]] = None, dtype_bits: int = 32) -> int:
        """
        Get effective number of bits for quantization.
        
        Args:
            matrix_shape: Optional tuple of (rows, cols) for matrix-level quantization bit adjustment
            dtype_bits: Number of bits in the data type (32 for fp32, 16 for fp16, etc.)
            
        Returns:
            Effective number of bits for quantization
        """
        base_bits = int(math.log2(self.q ** self.M))
        
        if self.matrix_level_quantization and matrix_shape is not None:
            m, n = matrix_shape
            # Add m * dtype_bits bits for matrix-level quantization
            # This matches the scalar quantization bit allocation: M*log2(q) + m*dtype_bits
            additional_bits = m * dtype_bits
            # Cap at 64 bits to avoid overflow issues while allowing reasonable bit allocation
            total_bits = base_bits + additional_bits
            return min(total_bits, 64)
        
        return base_bits
    
    def max_value(self, matrix_shape: Optional[Tuple[int, int]] = None, dtype_bits: int = 32) -> int:
        """Get maximum quantized value."""
        bits = self.effective_bits(matrix_shape, dtype_bits)
        if self.mode == "symmetric":
            return (2 ** (bits - 1)) - 1
        else:  # asymmetric
            return (2 ** bits) - 1
    
    def min_value(self, matrix_shape: Optional[Tuple[int, int]] = None, dtype_bits: int = 32) -> int:
        """Get minimum quantized value."""
        bits = self.effective_bits(matrix_shape, dtype_bits)
        if self.mode == "symmetric":
            return -(2 ** (bits - 1))
        else:  # asymmetric
            return 0
    
    def quantization_levels(self, matrix_shape: Optional[Tuple[int, int]] = None, dtype_bits: int = 32) -> int:
        """Get total number of quantization levels."""
        return 2 ** self.effective_bits(matrix_shape, dtype_bits)
    
    def to_dict(self, matrix_shape: Optional[Tuple[int, int]] = None, dtype_bits: int = 32) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "quantization_type": "scalar",
            "q": self.q,
            "M": self.M,
            "effective_bits": self.effective_bits(matrix_shape, dtype_bits),
            "mode": self.mode,
            "block_size": self.block_size,
            "per_row_scaling": self.per_row_scaling,
            "scale_factor": self.scale_factor,
            "asymmetric_method": self.asymmetric_method,
            "with_dither": self.with_dither,
            "with_tie_dither": self.with_tie_dither,
            "matrix_level_quantization": self.matrix_level_quantization,
            "max_value": self.max_value(matrix_shape, dtype_bits),
            "min_value": self.min_value(matrix_shape, dtype_bits),
            "quantization_levels": self.quantization_levels(matrix_shape, dtype_bits),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ScalarConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k != "quantization_type"})


# Preset configurations
SCALAR_INT4_SYM = ScalarConfig(q=4, M=2, mode="symmetric", block_size=8)  # 4-bit symmetric
SCALAR_INT8_SYM = ScalarConfig(q=2, M=8, mode="symmetric", block_size=8)  # 8-bit symmetric
SCALAR_INT4_ASYM = ScalarConfig(q=4, M=2, mode="asymmetric", block_size=8)  # 4-bit asymmetric
SCALAR_INT8_ASYM = ScalarConfig(q=2, M=8, mode="asymmetric", block_size=8)  # 8-bit asymmetric

# Additional presets for different block sizes
SCALAR_INT4_SYM_BLOCK4 = ScalarConfig(q=4, M=2, mode="symmetric", block_size=4)
SCALAR_INT8_SYM_BLOCK4 = ScalarConfig(q=2, M=8, mode="symmetric", block_size=4)
SCALAR_INT4_ASYM_BLOCK4 = ScalarConfig(q=4, M=2, mode="asymmetric", block_size=4)
SCALAR_INT8_ASYM_BLOCK4 = ScalarConfig(q=2, M=8, mode="asymmetric", block_size=4)

# Row-wise quantization (no blocking)
SCALAR_INT4_SYM_ROW = ScalarConfig(q=4, M=2, mode="symmetric", block_size=None)
SCALAR_INT8_SYM_ROW = ScalarConfig(q=2, M=8, mode="symmetric", block_size=None)
SCALAR_INT4_ASYM_ROW = ScalarConfig(q=4, M=2, mode="asymmetric", block_size=None)
SCALAR_INT8_ASYM_ROW = ScalarConfig(q=2, M=8, mode="asymmetric", block_size=None)

# Matrix-level quantization (whole matrix as one unit)
SCALAR_INT4_SYM_MATRIX = ScalarConfig(q=4, M=2, mode="symmetric", block_size=None, matrix_level_quantization=True)
SCALAR_INT8_SYM_MATRIX = ScalarConfig(q=2, M=8, mode="symmetric", block_size=None, matrix_level_quantization=True)
SCALAR_INT4_ASYM_MATRIX = ScalarConfig(q=4, M=2, mode="asymmetric", block_size=None, matrix_level_quantization=True)
SCALAR_INT8_ASYM_MATRIX = ScalarConfig(q=2, M=8, mode="asymmetric", block_size=None, matrix_level_quantization=True)


def get_scalar_config(preset: str = "int4_sym") -> ScalarConfig:
    """
    Get a preset scalar configuration.
    
    Args:
        preset: Configuration preset name
        
    Returns:
        ScalarConfig instance
        
    Available presets:
        - "int4_sym": 4-bit symmetric, block_size=8
        - "int8_sym": 8-bit symmetric, block_size=8
        - "int4_asym": 4-bit asymmetric, block_size=8
        - "int8_asym": 8-bit asymmetric, block_size=8
        - "int4_sym_block4": 4-bit symmetric, block_size=4
        - "int8_sym_block4": 8-bit symmetric, block_size=4
        - "int4_asym_block4": 4-bit asymmetric, block_size=4
        - "int8_asym_block4": 8-bit asymmetric, block_size=4
        - "int4_sym_row": 4-bit symmetric, row-wise
        - "int8_sym_row": 8-bit symmetric, row-wise
        - "int4_asym_row": 4-bit asymmetric, row-wise
        - "int8_asym_row": 8-bit asymmetric, row-wise
        - "int4_sym_matrix": 4-bit symmetric, matrix-level
        - "int8_sym_matrix": 8-bit symmetric, matrix-level
        - "int4_asym_matrix": 4-bit asymmetric, matrix-level
        - "int8_asym_matrix": 8-bit asymmetric, matrix-level
    """
    presets = {
        "int4_sym": SCALAR_INT4_SYM,
        "int8_sym": SCALAR_INT8_SYM,
        "int4_asym": SCALAR_INT4_ASYM,
        "int8_asym": SCALAR_INT8_ASYM,
        "int4_sym_block4": SCALAR_INT4_SYM_BLOCK4,
        "int8_sym_block4": SCALAR_INT8_SYM_BLOCK4,
        "int4_asym_block4": SCALAR_INT4_ASYM_BLOCK4,
        "int8_asym_block4": SCALAR_INT8_ASYM_BLOCK4,
        "int4_sym_row": SCALAR_INT4_SYM_ROW,
        "int8_sym_row": SCALAR_INT8_SYM_ROW,
        "int4_asym_row": SCALAR_INT4_ASYM_ROW,
        "int8_asym_row": SCALAR_INT8_ASYM_ROW,
        "int4_sym_matrix": SCALAR_INT4_SYM_MATRIX,
        "int8_sym_matrix": SCALAR_INT8_SYM_MATRIX,
        "int4_asym_matrix": SCALAR_INT4_ASYM_MATRIX,
        "int8_asym_matrix": SCALAR_INT8_ASYM_MATRIX,
    }
    
    if preset not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")
    
    return presets[preset]
