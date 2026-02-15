"""
E8 Lattice Codecs Module

This module provides optimized encode/decode functions specifically for E8 lattice
quantization, for E8 lattice quantization.
"""

import warnings
from typing import Tuple, Optional
import torch

from ..base import LatticeConfig, Lattice
from .lattice import E8Lattice
from enum import Enum


def e8_encode(
    x: torch.Tensor, 
    config: LatticeConfig,
    lattice: Optional[Lattice] = None,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """
    E8 hierarchical encoding.
    
    Encode vectors using hierarchical nested lattice quantization with M levels
    and handle overload by scaling the vector until quantization succeeds.
    
    Args:
        x: Input vector(s) to be encoded (shape [d] or [batch_size, d])
        config: LatticeConfig configuration
        lattice: Optional lattice instance (defaults to E8Lattice)
        dither: Optional dither vector for randomized quantization
        device: Device to perform computation on (defaults to x's device)
        
    Returns:
        Tuple of (encoding_vectors, T) where:
        - encoding_vectors: Tensor of shape [batch_size, M, d] or [M, d] containing M encoding vectors
        - T: Number of scaling operations performed to handle overload
    """
    # Determine device
    if device is None:
        device = x.device
    
    if lattice is None:
        lattice = E8Lattice(device=device)
    
    # Handle both single vector and batch cases
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Ensure x is on the correct device
    x = x.to(device)
    batch_size, d = x.shape
    if d != 8:
        raise ValueError(f"Input dimension {d} doesn't match E8 dimension 8")
        
    
    # Apply scaling and dithering
    x_scaled = x / config.beta
    if config.with_dither and dither is not None:
        dither = dither.to(device)
        if dither.dim() == 1:
            dither = dither.unsqueeze(0)  # Add batch dimension
        x_scaled = x_scaled + dither
    
    # Perform hierarchical encoding
    x_l = x_scaled.clone()
    encoding_vectors = []
    
    for _ in range(config.M):
        # encode to E8 lattice (vectorized for batch)
        encoded = lattice.projection(x_l)  # [batch_size, 8]
        encoded = lattice.encode_coords(encoded, config.q)  # [batch_size, 8]
        encoding_vectors.append(encoded)
        # Scale down for next level
        x_l = x_l / config.q
    
    encoding_vectors = torch.stack(encoding_vectors, dim=1)  # [batch_size, M, 8]
    
    # Remove batch dimension if input was single vector
    if squeeze_output:
        encoding_vectors = encoding_vectors.squeeze(0)  # [M, 8]
    
    return encoding_vectors, 0  # No scaling iterations needed


class DecodingMethod(Enum):
    """Enumeration of available decoding methods."""
    
    FULL = "full"
    # APPROXIMATE and PROGRESSIVE will be added later


class E8Decoder:
    """
    E8 lattice decoder with multiple decoding methods.
    
    This class provides different decoding strategies for E8 lattice quantization,
    allowing users to choose between accuracy and speed based on their needs.
    
    Methods:
    - full: Complete hierarchical decoding (default, most accurate)
    - approximate: Quick approximation for real-time applications (to be added)
    - progressive: Incremental decoding with intermediate results (to be added)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize E8 decoder.
        
        Args:
            device: Device to perform computation on (defaults to CPU)
        """
        self.device = device if device is not None else torch.device('cpu')
        self.lattice = E8Lattice(device=self.device)
    
    def decode(self, b: torch.Tensor, config: LatticeConfig, method: DecodingMethod = DecodingMethod.FULL) -> torch.Tensor:
        """
        Decode E8 lattice encoding vectors.
        
        Args:
            b: Encoding vectors of shape [batch_size, M, 8] or [M, 8]
            config: Lattice quantization configuration
            method: Decoding method to use (default: FULL)
            
        Returns:
            Decoded tensor of shape [batch_size, 8] or [8]
        """
        if method == DecodingMethod.FULL:
            return self._full_decode(b, config)
        else:
            raise ValueError(f"Unknown decoding method: {method}. Only 'full' is currently supported.")
    
    def _full_decode(self, b: torch.Tensor, config: LatticeConfig) -> torch.Tensor:
        """
        Full hierarchical decoding (most accurate).
        
        Performs complete hierarchical reconstruction with all levels.
        """
        # Handle both single vector and batch cases
        if b.dim() == 2:
            b = b.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, M, d = b.shape
        device = b.device
        
        # Ensure lattice is on correct device
        if self.lattice.device != device:
            self.lattice = E8Lattice(device=device)
        
        # Hierarchical reconstruction
        x_hat_list = []
        for i in range(M):
            b_i = b[:, i, :]  # [batch_size, 8]
            # Convert encoding coordinates to lattice point
            Gb = self.lattice.decode_coords(b_i, config.q)
            # Compute quantization error
            x_i_hat = Gb - config.q * self.lattice.projection(Gb / config.q)
            x_hat_list.append(x_i_hat)
        
        # Sum with appropriate weights
        x_hat = torch.zeros_like(x_hat_list[0])
        for i, x_i in enumerate(x_hat_list):
            x_hat += (config.q ** i) * x_i
        
        # Apply scaling compensation
        x_hat = x_hat * config.beta
        
        if squeeze_output:
            x_hat = x_hat.squeeze(0)
        
        return x_hat
    
    


def e8_decode(
    b: torch.Tensor,
    config: LatticeConfig,
    method: str = "full",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    E8 hierarchical decoding wrapper function.
    
    This function provides a simple interface to the E8Decoder class,
    allowing users to choose different decoding methods.
    
    Args:
        b: Encoding vectors of shape [batch_size, M, 8] or [M, 8]
        config: Lattice quantization configuration
        method: Decoding method ("full" - only method currently supported)
        device: Device to perform computation on (defaults to b's device)
        
    Returns:
        Decoded tensor of shape [batch_size, 8] or [8]
    """
    # Convert string method to enum
    method_map = {
        "full": DecodingMethod.FULL,
        # "approximate": DecodingMethod.APPROXIMATE,  # To be added later
        # "progressive": DecodingMethod.PROGRESSIVE,  # To be added later
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown decoding method: {method}. Available: {list(method_map.keys())}")
    
    # Create decoder and decode
    decoder = E8Decoder(device=device)
    return decoder.decode(b, config, method_map[method])


def e8_quantize(x: torch.Tensor, q: int, lattice: Optional[E8Lattice] = None) -> torch.Tensor:
    """
    E8 quantization by combining encoding and decoding.
    
    This function first encodes the input using hierarchical nested lattice quantization,
    then decodes it back to get a valid E8 lattice point.
    
    Args:
        x: Input tensor to quantize (shape [8] or [batch_size, 8])
        q: Quantization parameter
        lattice: Optional E8Lattice instance (creates new one if None)
        
    Returns:
        Quantized tensor (valid E8 lattice point)
    """
    if lattice is None:
        lattice = E8Lattice(device=x.device)
    
    # Handle batch inputs - use vectorized processing instead of Python loop
    if x.dim() == 2:
        # Batch processing with vectorized operations
        batch_size = x.shape[0]
        
        # Create a simple config for quantization
        config = LatticeConfig(
            lattice_type='E8',
            q=q,
            M=2,  # Default M=2 for hierarchical quantization
            beta=1.0,  # No additional scaling
            alpha=1.0,
            max_scaling_iterations=10,
            with_dither=False,
            disable_overload_protection=True
        )
        
        # Step 1: vectorized encoding for all vectors at once
        encoding_vectors, t = e8_encode(x, config, lattice=lattice)
        
        # Step 2: vectorized decoding for all vectors at once
        x_final = e8_decode(encoding_vectors, config, method="full")
        
        return x_final
    
    # Single vector processing
    if x.shape[0] != 8:
        raise ValueError(f"Input dimension {x.shape[0]} doesn't match E8 dimension 8")
    
    # Create a simple config for quantization
    config = LatticeConfig(
        lattice_type='E8',
        q=q,
        M=2,  # Default M=2 for hierarchical quantization
        beta=1.0,  # No additional scaling
        alpha=1.0,
        max_scaling_iterations=10,
        with_dither=False,
        disable_overload_protection=True
    )
    
    # Step 1: encoding
    encoding_vectors, t = e8_encode(x, config, lattice=lattice)
    
    # Step 2: decoding
    x_final = e8_decode(encoding_vectors, config, method="full")
    
    return x_final

    