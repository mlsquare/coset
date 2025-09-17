"""
Parameter management for hierarchical nested-lattice quantization.

This module provides configuration classes and parameter validation
for the quantization system.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QuantizationConfig:
    """
    Configuration class for hierarchical nested lattice quantizer parameters.
    
    This class provides a structured way to manage quantization parameters
    with built-in validation and default values.
    
    Attributes:
        lattice_type: Type of lattice ('D4', 'E8', 'Z2')
        q: Quantization parameter (alphabet size)
        M: Number of hierarchical levels
        beta: Scaling parameter for quantization
        alpha: Scaling parameter for overload handling
        max_scaling_iterations: Maximum number of scaling iterations
        with_tie_dither: Whether to add dither to break ties
        with_dither: Whether to add dither for randomized quantization
        disable_scaling: Whether to disable beta scaling (performance optimization)
        disable_overload_protection: Whether to disable overload protection (performance optimization)
    """
    
    lattice_type: str = "D4"
    q: int = 4
    M: int = 2
    beta: float = 1.0
    alpha: float = 1.0
    max_scaling_iterations: int = 10
    with_tie_dither: bool = True
    with_dither: bool = False
    disable_scaling: bool = True
    disable_overload_protection: bool = True
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.lattice_type not in ["D4", "E8", "Z2"]:
            raise ValueError(f"Unsupported lattice type: {self.lattice_type}")
        if self.q <= 0:
            raise ValueError("Quantization parameter q must be positive")
        if self.beta <= 0:
            raise ValueError("Scaling parameter beta must be positive")
        if self.alpha <= 0:
            raise ValueError("Scaling parameter alpha must be positive")
        if self.M <= 0:
            raise ValueError("Number of hierarchical levels M must be positive")
        if self.max_scaling_iterations <= 0:
            raise ValueError("max_scaling_iterations must be positive")
        if not isinstance(self.with_tie_dither, bool):
            raise ValueError("with_tie_dither must be a Boolean")
        if not isinstance(self.with_dither, bool):
            raise ValueError("with_dither must be a Boolean")
        if not isinstance(self.disable_scaling, bool):
            raise ValueError("disable_scaling must be a Boolean")
        if not isinstance(self.disable_overload_protection, bool):
            raise ValueError("disable_overload_protection must be a Boolean")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "lattice_type": self.lattice_type,
            "q": self.q,
            "M": self.M,
            "beta": self.beta,
            "alpha": self.alpha,
            "max_scaling_iterations": self.max_scaling_iterations,
            "with_tie_dither": self.with_tie_dither,
            "with_dither": self.with_dither,
            "disable_scaling": self.disable_scaling,
            "disable_overload_protection": self.disable_overload_protection,
        }
    
    def get_rate(self) -> float:
        """
        Calculate the rate in bits per dimension.
        
        Returns:
            Rate in bits per dimension
        """
        return self.M * self.q.bit_length() - 1  # log2(q) bits per level
    
    def get_compression_ratio(self) -> float:
        """
        Calculate the theoretical compression ratio.
        
        Returns:
            Compression ratio (original bits / compressed bits)
        """
        original_bits = 32  # float32
        compressed_bits = self.get_rate()
        return original_bits / compressed_bits
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (
            f"QuantizationConfig(lattice={self.lattice_type}, q={self.q}, "
            f"M={self.M}, beta={self.beta:.3f}, alpha={self.alpha:.3f}, "
            f"disable_scaling={self.disable_scaling}, "
            f"disable_overload_protection={self.disable_overload_protection})"
        )
