"""
CUDA-accelerated E8 operations with automatic fallback.

This module provides CUDA kernel wrappers for E8 quantization operations.
If CUDA is not available or kernels are not compiled, it falls back to
the PyTorch GPU implementations in e8_gpu.py.
"""

import torch
from typing import Optional, Tuple
from .e8_gpu import batch_e8_quantize, batch_encode_e8, batch_decode_e8, batch_quantize_e8
from ..lattices import E8Lattice
from .params import QuantizationConfig

# Try to load CUDA extensions (will be None if not available)
try:
    # Future: Load compiled CUDA extensions here
    # from .e8_cuda_cpp import e8_quantize_cuda, e8_encode_cuda, e8_decode_cuda
    _CUDA_AVAILABLE = False
except ImportError:
    _CUDA_AVAILABLE = False


def is_cuda_available() -> bool:
    """Check if CUDA kernels are available."""
    return _CUDA_AVAILABLE and torch.cuda.is_available()


def e8_quantize_cuda(
    X: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    CUDA-accelerated E8 quantization.
    
    Falls back to PyTorch GPU implementation if CUDA kernels not available.
    
    Args:
        X: Input tensor [batch_size, 8]
        device: Device to perform computation on
        
    Returns:
        Quantized tensor [batch_size, 8]
    """
    if is_cuda_available():
        # Future: Call actual CUDA kernel
        # return e8_quantize_cuda_kernel(X, device)
        pass
    
    # Fallback to PyTorch GPU implementation
    return batch_e8_quantize(X, device=device)


def e8_encode_cuda(
    X: torch.Tensor,
    lattice: E8Lattice,
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated E8 batch encoding.
    
    Falls back to PyTorch GPU implementation if CUDA kernels not available.
    
    Args:
        X: Input matrix [batch_size, 8]
        lattice: E8Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        device: Device to perform computation on
        
    Returns:
        Tuple of (encodings [batch_size, M, 8], T_values [batch_size])
    """
    if is_cuda_available():
        # Future: Call actual CUDA kernel
        # return e8_encode_cuda_kernel(X, lattice, config, dither, device)
        pass
    
    # Fallback to PyTorch GPU implementation
    return batch_encode_e8(X, lattice, config, dither, device)


def e8_decode_cuda(
    encodings: torch.Tensor,
    T_values: torch.Tensor,
    lattice: E8Lattice,
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    CUDA-accelerated E8 batch decoding.
    
    Falls back to PyTorch GPU implementation if CUDA kernels not available.
    
    Args:
        encodings: Encoding vectors [batch_size, M, 8]
        T_values: Scaling counts [batch_size]
        lattice: E8Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        device: Device to perform computation on
        
    Returns:
        Decoded matrix [batch_size, 8]
    """
    if is_cuda_available():
        # Future: Call actual CUDA kernel
        # return e8_decode_cuda_kernel(encodings, T_values, lattice, config, dither, device)
        pass
    
    # Fallback to PyTorch GPU implementation
    return batch_decode_e8(encodings, T_values, lattice, config, dither, device)


def e8_quantize_cuda_wrapper(
    X: torch.Tensor,
    lattice: E8Lattice,
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    CUDA-accelerated complete quantization (encode + decode).
    
    Falls back to PyTorch GPU implementation if CUDA kernels not available.
    
    Args:
        X: Input matrix [batch_size, 8]
        lattice: E8Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        device: Device to perform computation on
        
    Returns:
        Quantized matrix [batch_size, 8]
    """
    encodings, T_values = e8_encode_cuda(X, lattice, config, dither, device)
    return e8_decode_cuda(encodings, T_values, lattice, config, dither, device)
