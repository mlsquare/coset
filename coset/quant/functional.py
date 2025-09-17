"""
Core quantization functions for hierarchical nested-lattice quantization.

This module implements the core algorithms for encoding, decoding, and quantization
as specified in the paper by Kaplan & Ordentlich (2025).
"""

import warnings
from typing import Tuple, Optional
import torch
from .params import QuantizationConfig
from ..lattices import Lattice


def encode(
    x: torch.Tensor, 
    lattice: Lattice, 
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, int]:
    """
    Algorithm 1: Hierarchical encoding.
    
    Encode a vector using hierarchical nested lattice quantization with M levels
    and handle overload by scaling the vector until quantization succeeds.
    
    Args:
        x: Input vector to be quantized
        lattice: Lattice instance for quantization
        config: Quantization configuration
        dither: Optional dither vector for randomized quantization
        
    Returns:
        Tuple of (encoding_vectors, T) where:
        - encoding_vectors: Tensor of shape [M, d] containing M encoding vectors
        - T: Number of scaling operations performed to handle overload
    """
    # Ensure x is a vector
    x = x.flatten()
    if x.shape[0] != lattice.d:
        raise ValueError(f"Input dimension {x.shape[0]} doesn't match lattice dimension {lattice.d}")
    
    # Apply scaling and dithering
    x_scaled = x / config.beta
    if config.with_dither and dither is not None:
        x_scaled = x_scaled + dither.flatten()
    
    # Perform hierarchical encoding
    encoding_vectors, did_overload = _encode_internal(x_scaled, lattice, config)
    t = 0
    
    # Handle overload with scaling
    if not config.disable_overload_protection:
        while did_overload and t < config.max_scaling_iterations:
            t += 1
            x_scaled = x_scaled / (2 ** config.alpha)
            encoding_vectors, did_overload = _encode_internal(x_scaled, lattice, config)
        
        if did_overload:
            warnings.warn(
                f"Overload not resolved after {config.max_scaling_iterations} iterations. "
                "Consider increasing max_scaling_iterations or adjusting parameters."
            )
    
    return encoding_vectors, t


def _encode_internal(
    x: torch.Tensor, 
    lattice: Lattice, 
    config: QuantizationConfig
) -> Tuple[torch.Tensor, bool]:
    """
    Internal encoding function that performs hierarchical quantization.
    
    Args:
        x: Input vector (already scaled and dithered)
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Tuple of (encoding_vectors, overload_error)
    """
    x_l = x.clone()
    encoding_vectors = []
    
    for _ in range(config.M):
        # Quantize to lattice
        x_l = lattice.Q(x_l)
        
        # Convert to encoding coordinates
        b_i = lattice.encode_coords(x_l, config.q)
        encoding_vectors.append(b_i)
        
        # Scale down for next level
        x_l = x_l / config.q
    
    # Check for overload
    overload_error = not torch.allclose(lattice.Q(x_l), torch.zeros_like(x_l), atol=1e-8)
    
    return torch.stack(encoding_vectors), overload_error


def decode(
    b: torch.Tensor, 
    lattice: Lattice, 
    config: QuantizationConfig,
    T: int = 0,
    dither: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Algorithm 2: Hierarchical decoding.
    
    Decode hierarchical encoding vectors back to the original space,
    accounting for any scaling that was applied during encoding.
    
    Args:
        b: Encoding vectors of shape [M, d]
        lattice: Lattice instance
        config: Quantization configuration
        T: Number of scaling operations that were applied during encoding
        dither: Optional dither vector (if used during encoding)
        
    Returns:
        Reconstructed vector
    """
    if b.shape[0] != config.M:
        raise ValueError(f"Number of encoding levels {b.shape[0]} doesn't match M={config.M}")
    if b.shape[1] != lattice.d:
        raise ValueError(f"Encoding dimension {b.shape[1]} doesn't match lattice dimension {lattice.d}")
    
    # Perform hierarchical reconstruction
    x_hat_list = []
    for i in range(config.M):
        b_i = b[i]
        
        # Convert encoding coordinates to lattice point
        Gb = lattice.decode_coords(b_i, config.q)
        
        # Compute quantization error
        x_i_hat = Gb - config.q * lattice.Q(Gb / config.q)
        x_hat_list.append(x_i_hat)
    
    # Sum with appropriate weights
    x_hat = torch.zeros_like(x_hat_list[0])
    for i, x_i in enumerate(x_hat_list):
        x_hat += (config.q ** i) * x_i
    
    # Remove dither if it was applied
    if config.with_dither and dither is not None:
        x_hat = x_hat - dither.flatten()
    
    # Apply scaling compensation
    x_hat = config.beta * x_hat * (2 ** (config.alpha * T))
    
    return x_hat


def quantize(
    x: torch.Tensor, 
    lattice: Lattice, 
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Complete hierarchical quantization process: encode and decode a vector.
    
    This is a convenience method that performs both encoding and decoding
    in a single call, returning the quantized version of the input vector.
    
    Args:
        x: Input vector to be quantized
        lattice: Lattice instance
        config: Quantization configuration
        dither: Optional dither vector for randomized quantization
        
    Returns:
        Quantized version of the input vector
    """
    b, T = encode(x, lattice, config, dither)
    return decode(b, lattice, config, T, dither)


def mac_modq(x: torch.Tensor, y: torch.Tensor, q: int) -> torch.Tensor:
    """
    Modular multiply-accumulate: <x,y> mod q.
    
    Compute the inner product of two vectors modulo q.
    
    Args:
        x: First vector
        y: Second vector
        q: Modulus
        
    Returns:
        Inner product modulo q
    """
    return torch.sum(x * y) % q


def accumulate_modq(acc: torch.Tensor, x: torch.Tensor, q: int) -> torch.Tensor:
    """
    Modular accumulation: (acc + x) mod q.
    
    Add x to accumulator modulo q.
    
    Args:
        acc: Accumulator tensor
        x: Value to add
        q: Modulus
        
    Returns:
        Updated accumulator modulo q
    """
    return (acc + x) % q


def batch_encode(
    X: torch.Tensor, 
    lattice: Lattice, 
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode multiple vectors efficiently.
    
    Args:
        X: Input matrix where each row is a vector to encode
        lattice: Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        
    Returns:
        Tuple of (encoded_vectors, scaling_counts) where:
        - encoded_vectors: Tensor of shape [batch_size, M, d]
        - scaling_counts: Tensor of shape [batch_size] with scaling counts
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    batch_size = X.shape[0]
    encoded_vectors = []
    scaling_counts = []
    
    for i in range(batch_size):
        b, T = encode(X[i], lattice, config, dither)
        encoded_vectors.append(b)
        scaling_counts.append(T)
    
    return torch.stack(encoded_vectors), torch.tensor(scaling_counts)


def batch_decode(
    encoded_vectors: torch.Tensor, 
    scaling_counts: torch.Tensor,
    lattice: Lattice, 
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Decode multiple vectors efficiently.
    
    Args:
        encoded_vectors: Tensor of shape [batch_size, M, d]
        scaling_counts: Tensor of shape [batch_size]
        lattice: Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        
    Returns:
        Matrix where each row is a decoded vector
    """
    batch_size = encoded_vectors.shape[0]
    decoded_vectors = []
    
    for i in range(batch_size):
        decoded = decode(encoded_vectors[i], lattice, config, scaling_counts[i].item(), dither)
        decoded_vectors.append(decoded)
    
    return torch.stack(decoded_vectors)
