"""
Configuration classes for lattice quantization
"""

from enum import Enum
from typing import List, Optional
import torch


class LatticeType(Enum):
    """Supported lattice types for quantization."""
    E8 = "e8"               # E8 lattice (8D)
    A2 = "a2"               # A2 lattice (2D hexagonal)
    Z2 = "z2"               # Z2 lattice (2D square)
    D4 = "d4"               # D4 lattice (4D)
    CUSTOM = "custom"       # User-defined lattice
    
    def get_dimension(self) -> int:
        """Get the dimension of the lattice type."""
        dimension_map = {
            LatticeType.E8: 8,
            LatticeType.A2: 2,
            LatticeType.Z2: 2,
            LatticeType.D4: 4,
            LatticeType.CUSTOM: None  # User must specify
        }
        return dimension_map[self]


class LatticeConfig:
    """
    Configuration for lattice quantization using HNLQ method.
    
    This class defines the parameters for hierarchical nested lattice quantization (HNLQ),
    which is the process of encoding continuous values to discrete lattice points and
    decoding them back. HNLQ can be applied to different lattice types (E8, A2, Z2, D4, etc.)
    with configurable dimensions, radix values, and encoding/decoding depths.
    """
    
    def __init__(
        self,
        type: LatticeType = LatticeType.Z2,
        radix: int = 4,
        num_layers: int = 3,
        lattice_dim: Optional[int] = None,
        beta: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-8,
        overload: bool = True,
        max_scaling_iterations: int = 10,
        with_tie_dither: bool = True,
        with_dither: bool = False,
    ):
        """
        Initialize lattice configuration.
        
        Args:
            type: Type of lattice to use (E8=8D, A2=2D, Z2=2D, D4=4D, CUSTOM)
            radix: Base for HNLQ quantization (quantization parameter q)
            num_layers: Number of hierarchy levels for HNLQ encoding/decoding
            lattice_dim: Dimension of the lattice (auto-set based on type, required for CUSTOM)
            beta: Scaling parameter for quantization
            alpha: Scaling parameter for overload handling
            eps: Small perturbation parameter
            overload: Whether to handle overload by scaling
            max_scaling_iterations: Maximum number of scaling iterations
            with_tie_dither: Whether to add tie dither
            with_dither: Whether to add dither for randomized quantization
        """
        self.type = type
        self.radix = radix
        self.num_layers = num_layers
        
        # Set lattice dimension based on lattice type
        if lattice_dim is None:
            if type == LatticeType.CUSTOM:
                raise ValueError("lattice_dim must be specified for CUSTOM lattice type")
            self.lattice_dim = type.get_dimension()
        else:
            if type != LatticeType.CUSTOM and lattice_dim != type.get_dimension():
                raise ValueError(f"lattice_dim {lattice_dim} does not match {type.name} lattice dimension {type.get_dimension()}")
            self.lattice_dim = lattice_dim
            
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.overload = overload
        self.max_scaling_iterations = max_scaling_iterations
        self.with_tie_dither = with_tie_dither
        self.with_dither = with_dither
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.radix < 2:
            raise ValueError(f"Radix must be >= 2, got {self.radix}")
        
        if self.num_layers < 1:
            raise ValueError(f"Number of layers must be >= 1, got {self.num_layers}")
        
        if self.lattice_dim < 1:
            raise ValueError(f"Lattice dimension must be >= 1, got {self.lattice_dim}")
        
        if self.beta <= 0:
            raise ValueError(f"Beta must be > 0, got {self.beta}")
        
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be > 0, got {self.alpha}")
        
        if self.max_scaling_iterations <= 0:
            raise ValueError(f"Max scaling iterations must be > 0, got {self.max_scaling_iterations}")
    
    def get_num_codewords(self) -> int:
        """Get number of codewords in the lattice."""
        return 2 ** self.lattice_dim
    
    def get_max_radix_value(self, depth: int) -> int:
        """Get maximum value for radix-q encoding at given depth."""
        return self.radix ** depth
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'type': self.type.value,
            'radix': self.radix,
            'num_layers': self.num_layers,
            'lattice_dim': self.lattice_dim,
            'beta': self.beta,
            'alpha': self.alpha,
            'eps': self.eps,
            'overload': self.overload,
            'max_scaling_iterations': self.max_scaling_iterations,
            'with_tie_dither': self.with_tie_dither,
            'with_dither': self.with_dither,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'LatticeConfig':
        """Create configuration from dictionary."""
        config_dict = config_dict.copy()
        config_dict['type'] = LatticeType(config_dict['type'])
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return (f"LatticeConfig(type={self.type.value}, radix={self.radix}, "
                f"num_layers={self.num_layers}, lattice_dim={self.lattice_dim})")
