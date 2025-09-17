"""
PyTorch bindings for CUDA kernels.

This module provides PyTorch autograd functions and convenient wrappers
for the CUDA-accelerated quantization operations.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional
import os
import warnings

# Try to load CUDA extensions
try:
    from torch.utils.cpp_extension import load
    
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load CUDA extensions
    cuda_encode_ext = load(
        name="cuda_encode",
        sources=[os.path.join(current_dir, "encode_kernel.cu")],
        verbose=False
    )
    
    cuda_decode_ext = load(
        name="cuda_decode", 
        sources=[os.path.join(current_dir, "decode_kernel.cu")],
        verbose=False
    )
    
    cuda_quantize_ext = load(
        name="cuda_quantize",
        sources=[os.path.join(current_dir, "quantize_kernel.cu")],
        verbose=False
    )
    
    cuda_mac_ext = load(
        name="cuda_mac",
        sources=[os.path.join(current_dir, "mac_kernel.cu")],
        verbose=False
    )
    
    cuda_accumulate_ext = load(
        name="cuda_accumulate",
        sources=[os.path.join(current_dir, "accumulate_kernel.cu")],
        verbose=False
    )
    
    cuda_lut_ext = load(
        name="cuda_lut",
        sources=[os.path.join(current_dir, "lut_kernel.cu")],
        verbose=False
    )
    
    CUDA_AVAILABLE = True
    
except Exception as e:
    warnings.warn(f"Failed to load CUDA extensions: {e}")
    CUDA_AVAILABLE = False
    cuda_encode_ext = None
    cuda_decode_ext = None
    cuda_quantize_ext = None
    cuda_mac_ext = None
    cuda_accumulate_ext = None
    cuda_lut_ext = None


class CudaEncodeFunction(Function):
    """Autograd function for CUDA-accelerated encoding."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, G_inv: torch.Tensor, q: int, M: int, 
                beta: float, alpha: float, max_scaling_iterations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CUDA encoding.
        
        Args:
            x: Input vectors [batch_size, d]
            G_inv: Inverse generator matrix [d, d]
            q: Quantization parameter
            M: Number of hierarchical levels
            beta: Scaling parameter
            alpha: Overload scaling parameter
            max_scaling_iterations: Maximum scaling iterations
            
        Returns:
            Tuple of (encodings, t_values)
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extensions not available")
        
        # Ensure tensors are on the same device
        device = x.device
        G_inv = G_inv.to(device)
        
        # Call CUDA kernel
        result = cuda_encode_ext.cuda_encode_forward(
            x, G_inv, q, M, beta, alpha, max_scaling_iterations
        )
        
        # Split result into encodings and t_values
        batch_size = x.size(0)
        d = x.size(1)
        encodings = result[:batch_size * M * d].view(batch_size, M, d)
        t_values = result[batch_size * M * d:].view(batch_size)
        
        # Save for backward pass
        ctx.save_for_backward(x, G_inv)
        ctx.q = q
        ctx.M = M
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.max_scaling_iterations = max_scaling_iterations
        
        return encodings, t_values
    
    @staticmethod
    def backward(ctx, grad_encodings: torch.Tensor, grad_t_values: torch.Tensor):
        """Backward pass (straight-through estimator)."""
        # Straight-through estimator: pass gradients through unchanged
        x, G_inv = ctx.saved_tensors
        grad_x = grad_encodings.sum(dim=1) if grad_encodings is not None else None
        return grad_x, None, None, None, None, None, None


class CudaDecodeFunction(Function):
    """Autograd function for CUDA-accelerated decoding."""
    
    @staticmethod
    def forward(ctx, encodings: torch.Tensor, t_values: torch.Tensor, G: torch.Tensor,
                q: int, M: int, beta: float, alpha: float) -> torch.Tensor:
        """
        Forward pass for CUDA decoding.
        
        Args:
            encodings: Encoded vectors [batch_size, M, d]
            t_values: Scaling counts [batch_size]
            G: Generator matrix [d, d]
            q: Quantization parameter
            M: Number of hierarchical levels
            beta: Scaling parameter
            alpha: Overload scaling parameter
            
        Returns:
            Decoded vectors [batch_size, d]
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extensions not available")
        
        # Ensure tensors are on the same device
        device = encodings.device
        t_values = t_values.to(device)
        G = G.to(device)
        
        # Call CUDA kernel
        x_hat = cuda_decode_ext.cuda_decode_forward(
            encodings, t_values, G, q, M, beta, alpha
        )
        
        # Save for backward pass
        ctx.save_for_backward(encodings, t_values, G)
        ctx.q = q
        ctx.M = M
        ctx.beta = beta
        ctx.alpha = alpha
        
        return x_hat
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass (straight-through estimator)."""
        # Straight-through estimator: pass gradients through unchanged
        return grad_output, None, None, None, None, None, None


class CudaQuantizeFunction(Function):
    """Autograd function for CUDA-accelerated combined quantization."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, G: torch.Tensor, G_inv: torch.Tensor,
                q: int, M: int, beta: float, alpha: float, max_scaling_iterations: int) -> torch.Tensor:
        """
        Forward pass for CUDA quantization.
        
        Args:
            x: Input vectors [batch_size, d]
            G: Generator matrix [d, d]
            G_inv: Inverse generator matrix [d, d]
            q: Quantization parameter
            M: Number of hierarchical levels
            beta: Scaling parameter
            alpha: Overload scaling parameter
            max_scaling_iterations: Maximum scaling iterations
            
        Returns:
            Quantized vectors [batch_size, d]
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extensions not available")
        
        # Ensure tensors are on the same device
        device = x.device
        G = G.to(device)
        G_inv = G_inv.to(device)
        
        # Call CUDA kernel
        x_hat = cuda_quantize_ext.cuda_quantize_forward(
            x, G, G_inv, q, M, beta, alpha, max_scaling_iterations
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, G, G_inv)
        ctx.q = q
        ctx.M = M
        ctx.beta = beta
        ctx.alpha = alpha
        ctx.max_scaling_iterations = max_scaling_iterations
        
        return x_hat
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass (straight-through estimator)."""
        # Straight-through estimator: pass gradients through unchanged
        return grad_output, None, None, None, None, None, None, None


# Convenience functions
def cuda_encode(x: torch.Tensor, lattice, config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated encoding.
    
    Args:
        x: Input vectors [batch_size, d]
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Tuple of (encodings, t_values)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extensions not available")
    
    return CudaEncodeFunction.apply(
        x, lattice.G_inv, config.q, config.M, config.beta, 
        config.alpha, config.max_scaling_iterations
    )


def cuda_decode(encodings: torch.Tensor, t_values: torch.Tensor, lattice, config) -> torch.Tensor:
    """
    CUDA-accelerated decoding.
    
    Args:
        encodings: Encoded vectors [batch_size, M, d]
        t_values: Scaling counts [batch_size]
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Decoded vectors [batch_size, d]
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extensions not available")
    
    return CudaDecodeFunction.apply(
        encodings, t_values, lattice.G, config.q, config.M, 
        config.beta, config.alpha
    )


def cuda_quantize(x: torch.Tensor, lattice, config) -> torch.Tensor:
    """
    CUDA-accelerated combined quantization.
    
    Args:
        x: Input vectors [batch_size, d]
        lattice: Lattice instance
        config: Quantization configuration
        
    Returns:
        Quantized vectors [batch_size, d]
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extensions not available")
    
    return CudaQuantizeFunction.apply(
        x, lattice.G, lattice.G_inv, config.q, config.M, 
        config.beta, config.alpha, config.max_scaling_iterations
    )


# Modulo Arithmetic Autograd Functions

class CudaLUTInnerProductFunction(Function):
    """Autograd function for CUDA-accelerated LUT inner product."""
    
    @staticmethod
    def forward(ctx, encodings_x, encodings_y, lut, q, d):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Store for backward pass
        ctx.save_for_backward(encodings_x, encodings_y, lut)
        ctx.q = q
        ctx.d = d
        
        # Call CUDA kernel
        return cuda_mac_ext.cuda_lut_inner_product(encodings_x, encodings_y, lut, q, d)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator for gradients
        return grad_output, None, None, None, None


class CudaMACEncodingSpaceFunction(Function):
    """Autograd function for CUDA-accelerated MAC in encoding space."""
    
    @staticmethod
    def forward(ctx, encodings_x, encodings_y, lut, q, d):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Store for backward pass
        ctx.save_for_backward(encodings_x, encodings_y, lut)
        ctx.q = q
        ctx.d = d
        
        # Call CUDA kernel
        return cuda_mac_ext.cuda_mac_encoding_space(encodings_x, encodings_y, lut, q, d)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator for gradients
        return grad_output, None, None, None, None


class CudaCarryAwareAccumulateFunction(Function):
    """Autograd function for CUDA-accelerated carry-aware accumulation."""
    
    @staticmethod
    def forward(ctx, encodings, layer_sums, q, d):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Store for backward pass
        ctx.save_for_backward(encodings, layer_sums)
        ctx.q = q
        ctx.d = d
        
        # Call CUDA kernel
        cuda_accumulate_ext.cuda_carry_aware_accumulate(encodings, layer_sums, q, d)
        return cuda_accumulate_ext.cuda_normalize_with_carry(layer_sums, q, d)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator for gradients
        return grad_output, None, None, None


class CudaBatchMACFunction(Function):
    """Autograd function for CUDA-accelerated batch MAC operations."""
    
    @staticmethod
    def forward(ctx, encodings_batch_x, encodings_batch_y, lut, q, d):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Store for backward pass
        ctx.save_for_backward(encodings_batch_x, encodings_batch_y, lut)
        ctx.q = q
        ctx.d = d
        
        # Call CUDA kernel
        return cuda_mac_ext.cuda_batch_mac(encodings_batch_x, encodings_batch_y, lut, q, d)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator for gradients
        return grad_output, None, None, None, None


class CudaAdaptiveMACFunction(Function):
    """Autograd function for CUDA-accelerated adaptive MAC operations."""
    
    @staticmethod
    def forward(ctx, encodings_x, encodings_y, lut, max_layers, q, d):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Store for backward pass
        ctx.save_for_backward(encodings_x, encodings_y, lut)
        ctx.max_layers = max_layers
        ctx.q = q
        ctx.d = d
        
        # Call CUDA kernel
        return cuda_mac_ext.cuda_adaptive_mac(encodings_x, encodings_y, lut, max_layers, q, d)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator for gradients
        return grad_output, None, None, None, None, None


# High-level CUDA functions for modulo arithmetic

def cuda_lut_inner_product(encodings_x, encodings_y, lut, q, d):
    """CUDA-accelerated LUT inner product."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return CudaLUTInnerProductFunction.apply(encodings_x, encodings_y, lut, q, d)


def cuda_mac_encoding_space(encodings_x, encodings_y, lut, q, d):
    """CUDA-accelerated MAC in encoding space."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return CudaMACEncodingSpaceFunction.apply(encodings_x, encodings_y, lut, q, d)


def cuda_carry_aware_accumulate(encodings, layer_sums, q, d):
    """CUDA-accelerated carry-aware accumulation."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return CudaCarryAwareAccumulateFunction.apply(encodings, layer_sums, q, d)


def cuda_batch_mac(encodings_batch_x, encodings_batch_y, lut, q, d):
    """CUDA-accelerated batch MAC operations."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return CudaBatchMACFunction.apply(encodings_batch_x, encodings_batch_y, lut, q, d)


def cuda_adaptive_mac(encodings_x, encodings_y, lut, max_layers, q, d):
    """CUDA-accelerated adaptive MAC operations."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return CudaAdaptiveMACFunction.apply(encodings_x, encodings_y, lut, max_layers, q, d)


# LUT management functions

def cuda_build_lut(lattice_points, lut_size, d):
    """CUDA-accelerated LUT building."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return cuda_lut_ext.cuda_build_lut(lattice_points, lut_size, d)


def cuda_build_one_sided_lut(query_vector, lattice_points, lut_size, d):
    """CUDA-accelerated one-sided LUT building."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return cuda_lut_ext.cuda_build_one_sided_lut(query_vector, lattice_points, lut_size, d)


def cuda_lut_lookup(indices_x, indices_y, lut):
    """CUDA-accelerated LUT lookup."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return cuda_lut_ext.cuda_lut_lookup(indices_x, indices_y, lut)


def cuda_one_sided_lut_lookup(indices, lut):
    """CUDA-accelerated one-sided LUT lookup."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return cuda_lut_ext.cuda_one_sided_lut_lookup(indices, lut)


def cuda_batch_lut_lookup(indices_batch_x, indices_batch_y, lut, M, q):
    """CUDA-accelerated batch LUT lookup."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA kernels not available")
    
    return cuda_lut_ext.cuda_batch_lut_lookup(indices_batch_x, indices_batch_y, lut, M, q)


# Check CUDA availability
def is_cuda_available() -> bool:
    """Check if CUDA kernels are available."""
    return CUDA_AVAILABLE and torch.cuda.is_available()


def get_cuda_info() -> dict:
    """Get information about CUDA availability and capabilities."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_extensions_available': CUDA_AVAILABLE,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name()
        info['cuda_version'] = torch.version.cuda
    
    return info
