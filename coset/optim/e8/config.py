"""
E8 Lattice Configuration Module

This module provides E8-specific configuration and optimal parameters
for hierarchical nested-lattice quantization.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class E8Config:
    """
    Configuration class for E8 lattice quantization.
    
    Optimized parameters specifically for E8 lattice with pre-computed
    optimal beta values for different quantization settings.
    
    Attributes:
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
    
    q: int = 4
    M: int = 2
    beta: float = 1.0
    alpha: float = 1.0
    max_scaling_iterations: int = 10
    with_tie_dither: bool = True
    with_dither: bool = False
    disable_scaling: bool = True
    disable_overload_protection: bool = True
    
    @staticmethod
    def get_optimal_beta(q: int, M: int = 2) -> float:
        """
        Get optimal beta scaling factor for E8 lattice.
        
        These values are pre-computed to ensure HNLQ is not overloaded and doesn't spend
        wasteful bits in the last level. The optimal beta depends on q and M.
        
        Args:
            q: Quantization parameter (alphabet size)
            M: Number of hierarchical levels
            
        Returns:
            Optimal beta value for E8 with given configuration
        """
        # Optimal beta values for E8 (pre-computed based on E8 geometry and q)
        optimal_betas: Dict[Tuple[int, int], float] = {
            (2, 2): 0.4,   # E8, q=2, M=2
            (3, 2): 0.35,  # E8, q=3, M=2
            (4, 2): 0.3,   # E8, q=4, M=2
            (5, 2): 0.25,  # E8, q=5, M=2
            (8, 2): 0.2,   # E8, q=8, M=2
            (16, 2): 0.15, # E8, q=16, M=2
        }
        
        key = (q, M)
        if key in optimal_betas:
            return optimal_betas[key]
        
        # Default fallback: use a heuristic based on q
        # As q increases, beta should decrease to avoid overload
        default_beta = 0.4 / (q ** 0.5)
        return default_beta
    
    def __post_init__(self):
        """Validate parameters and auto-set optimal beta if needed."""
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
        
        # Auto-set optimal beta if using default value
        if self.beta == 1.0:
            self.beta = self.get_optimal_beta(self.q, self.M)
    
    def get_compression_ratio(self) -> float:
        """
        Calculate the compression ratio achieved by this configuration.
        
        For E8 lattice (d=8), compression ratio is: (8 * 32) / (M * 8 * log2(q))
        where 32 is bits per float32, M is number of levels, q is alphabet size.
        
        Returns:
            Compression ratio (original_bits / compressed_bits)
        """
        import math
        d = 8  # E8 dimension
        bits_per_float = 32
        bits_per_level = d * math.log2(self.q)
        total_bits = self.M * bits_per_level
        original_bits = d * bits_per_float
        return original_bits / total_bits
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'lattice_type': 'E8',
            'q': self.q,
            'M': self.M,
            'beta': self.beta,
            'alpha': self.alpha,
            'max_scaling_iterations': self.max_scaling_iterations,
            'with_tie_dither': self.with_tie_dither,
            'with_dither': self.with_dither,
            'disable_scaling': self.disable_scaling,
            'disable_overload_protection': self.disable_overload_protection,
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"E8Config(q={self.q}, M={self.M}, beta={self.beta:.3f}, "
            f"alpha={self.alpha:.3f}, disable_scaling={self.disable_scaling}, "
            f"disable_overload_protection={self.disable_overload_protection})"
        )


def get_e8_config(q: int = 4, M: int = 2, **kwargs) -> E8Config:
    """
    Factory function to create E8Config with optimal defaults.
    
    Args:
        q: Quantization parameter (default: 4)
        M: Number of hierarchical levels (default: 2)
        **kwargs: Additional configuration parameters
        
    Returns:
        E8Config instance with optimal beta pre-set
        
    Example:
        >>> config = get_e8_config(q=4, M=2)
        >>> print(config.beta)  # 0.3 (optimal for E8, q=4, M=2)
    """
    return E8Config(q=q, M=M, **kwargs)


# Pre-defined common configurations
E8_FAST = E8Config(
    q=4, M=2, beta=0.3,
    disable_scaling=True,
    disable_overload_protection=True,
    with_tie_dither=False,
    with_dither=False,
)

E8_ACCURATE = E8Config(
    q=4, M=2, beta=0.3,
    disable_scaling=False,
    disable_overload_protection=False,
    with_tie_dither=True,
    with_dither=False,
)

E8_HIGH_COMPRESSION = E8Config(
    q=8, M=2, beta=0.2,
    disable_scaling=True,
    disable_overload_protection=True,
)

E8_LOW_DISTORTION = E8Config(
    q=2, M=3, beta=0.4,
    disable_scaling=False,
    disable_overload_protection=False,
)
