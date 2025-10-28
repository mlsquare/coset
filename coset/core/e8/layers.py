"""
E8-specific layer utilities and convenience functions.

This module provides E8-specific helper functions and convenience wrappers
for the generic vector quantization layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from .lattice import E8Lattice
from .codecs import e8_encode, e8_decode


def create_e8_hnlq_linear(in_dim, out_dim, device=None, **kwargs):
    """
    Convenience function to create HNLQLinear with E8 lattice.
    
    Args:
        in_dim: Input dimension (must be divisible by 8)
        out_dim: Output dimension
        device: Device to place the layer on (defaults to cuda if available, else CPU)
        **kwargs: Additional arguments passed to HNLQLinear
        
    Returns:
        HNLQLinear instance configured for E8 lattice
    """
    # Import here to avoid circular imports
    from ..vq_layers import HNLQLinear, get_generators
    from .codecs import e8_quantize
    
    # Ensure input dimension is divisible by 8
    if in_dim % 8 != 0:
        raise ValueError(f"Input dimension {in_dim} must be divisible by 8 for E8 lattice")
    
    # Default to cuda if available and device not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create E8 lattice and get generators
    lattice = E8Lattice(device=device)
    G, Ginv = get_generators(lattice)
    
    # Set default E8 parameters
    defaults = {
        'G': G,
        'Ginv': Ginv,
        'quantize_fn': e8_quantize,  # Pass E8-specific quantizer
        'block_size': 8,
        'lattice_type': 'E8',
        'q': 4,
        'M': 2,
        'Delta0': 1.5,
    }
    
    # Update with provided kwargs
    defaults.update(kwargs)
    
    return HNLQLinear(in_dim, out_dim, **defaults)