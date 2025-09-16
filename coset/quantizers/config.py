"""
Configuration classes for lattice quantization
"""

from enum import Enum
from typing import List, Optional
import torch


class LatticeType(Enum):
    """Supported lattice types for quantization."""
    HNLQ = "hnlq"           # Hierarchical Nested Lattice Quantization
    E8 = "e8"               # E8 lattice (8D)
    A2 = "a2"               # A2 lattice (2D hexagonal)
    Z2 = "z2"               # Z2 lattice (2D square)
    D4 = "d4"               # D4 lattice (4D)
    CUSTOM = "custom"       # User-defined lattice


class LatticeConfig:
    """
    Configuration for lattice quantization.
    
    This class defines the parameters for hierarchical nested lattice quantization,
    including lattice type, dimensions, radix values, and quantization depths.
    """
    
    def __init__(
        self,
        type: LatticeType = LatticeType.HNLQ,
        radix: int = 4,
        num_layers: int = 3,
        lattice_dim: int = 8,
        scales: Optional[List[float]] = None,
        zero_points: Optional[List[int]] = None,
        learnable_scales: bool = True,
        learnable_zero_points: bool = False,
    ):
        """
        Initialize lattice configuration.
        
        Args:
            type: Type of lattice to use
            radix: Base for radix-q encoding
            num_layers: Number of hierarchy levels
            lattice_dim: Dimension of the lattice
            scales: Scale factors for each layer (if None, auto-generated)
            zero_points: Zero points for each layer (if None, auto-generated)
            learnable_scales: Whether scales are learnable parameters
            learnable_zero_points: Whether zero points are learnable parameters
        """
        self.type = type
        self.radix = radix
        self.num_layers = num_layers
        self.lattice_dim = lattice_dim
        self.learnable_scales = learnable_scales
        self.learnable_zero_points = learnable_zero_points
        
        # Initialize scales and zero points
        if scales is None:
            self.scales = [2.0 ** i for i in range(num_layers)]
        else:
            self.scales = scales
            
        if zero_points is None:
            self.zero_points = [0] * num_layers
        else:
            self.zero_points = zero_points
            
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
        
        if len(self.scales) != self.num_layers:
            raise ValueError(f"Number of scales ({len(self.scales)}) must match number of layers ({self.num_layers})")
        
        if len(self.zero_points) != self.num_layers:
            raise ValueError(f"Number of zero points ({len(self.zero_points)}) must match number of layers ({self.num_layers})")
    
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
            'scales': self.scales,
            'zero_points': self.zero_points,
            'learnable_scales': self.learnable_scales,
            'learnable_zero_points': self.learnable_zero_points,
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
