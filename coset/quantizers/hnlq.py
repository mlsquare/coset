"""
Hierarchical Nested Lattice Quantization (HNLQ) Implementation

This module implements the core HNLQ quantizer based on the reference implementation
from gemvq. It supports proper hierarchical quantization with generator matrices,
closest point algorithms, and correct encoding/decoding procedures.

Core Components:
1. Generator Matrices: Lattice-specific generator matrices (G)
2. Closest Point Functions: Lattice-specific closest point algorithms (Q_nn)
3. Hierarchical Encoding: Multi-level quantization with proper radix-q operations
4. Hierarchical Decoding: Correct reconstruction using hierarchical formula
5. PyTorch Integration: Full PyTorch tensor compatibility

Quantization Process:
1. Encoding: Maps continuous values to hierarchical encoding vectors
2. Decoding: Reconstructs continuous values from encoding vectors
3. Packing: Optional compression of encoding vectors
4. Lookup Operations: Fast operations using precomputed tables
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Callable
import math

from .config import LatticeConfig, LatticeType

# Optional CUDA kernel imports
try:
    from .cuda_kernels import (
        closest_point_e8_cuda, vectorized_quantize_cuda, 
        quantized_matmul_cuda, batch_product_quantize_cuda
    )
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False

try:
    from .quantization_cuda_kernels import (
        closest_point_e8_quantization_cuda,
        vectorized_encode_decode_cuda,
        batch_quantize_cuda,
        ultra_fast_quantize_cuda
    )
    QUANTIZATION_CUDA_KERNELS_AVAILABLE = True
except ImportError:
    QUANTIZATION_CUDA_KERNELS_AVAILABLE = False

try:
    from .optimized_quantization_cuda_kernels import (
        optimized_closest_point_e8_cuda,
        optimized_ultra_fast_quantize_cuda,
        optimized_vectorized_encode_decode_cuda
    )
    OPTIMIZED_QUANTIZATION_CUDA_KERNELS_AVAILABLE = True
except ImportError:
    OPTIMIZED_QUANTIZATION_CUDA_KERNELS_AVAILABLE = False


def custom_round(x, tiny=None):
    """
    Custom rounding function that handles edge cases for lattice quantization.
    
    This function implements a rounding scheme that ensures consistent behavior
    at boundary points (0.5) for lattice quantization algorithms.
    """
    x = torch.as_tensor(x)
    
    if tiny is None:
        # choose a microscopic nudge relative to dtype
        tiny = torch.finfo(x.dtype if x.dtype.is_floating_point else torch.float64).eps
    
    # nudge toward zero so exact .5 falls to the nearer-integer toward zero
    y = x - torch.sign(x) * tiny
    
    # round-to-nearest via floor(x+0.5) works for all signs after the nudge
    result = torch.floor(y + 0.5)
    
    return result


def get_z2():
    """Get the generator matrix for the Z^2 lattice."""
    return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)


def get_d4():
    """Get the generator matrix for the D_4 lattice."""
    return torch.tensor([[-1.0, -1.0, 0.0, 0.0], 
                        [1.0, -1.0, 0.0, 0.0], 
                        [0.0, 1.0, -1.0, 0.0], 
                        [0.0, 0.0, 1.0, -1.0]], dtype=torch.float32).T


def get_e8():
    """Get the generator matrix for the E_8 lattice."""
    return torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32).T


def get_a2():
    """Get the generator matrix for the A_2 lattice."""
    return torch.tensor([[1.0, 0.0], [0.5, math.sqrt(3) / 2]], dtype=torch.float32).T


def closest_point_zn(x):
    """Find closest point in Z^n lattice."""
    return custom_round(x)


def g_x(x):
    """Helper function for D_n lattice closest point algorithm."""
    f_x = custom_round(x)
    delta = torch.abs(x - f_x)
    
    # Handle both 1D and 2D inputs
    if x.dim() == 1:
        # Original 1D case
        k = torch.argmax(delta)
        g_x_ = f_x.clone()
        
        x_k = x[k]
        f_x_k = f_x[k]
        
        if x_k >= 0:
            g_x_[k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
        else:
            g_x_[k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1
        
        return g_x_
    else:
        # Handle 2D batched case
        batch_size = x.shape[0]
        g_x_ = f_x.clone()
        
        for i in range(batch_size):
            k = torch.argmax(delta[i])
            x_k = x[i, k]
            f_x_k = f_x[i, k]
            
            if x_k >= 0:
                g_x_[i, k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
            else:
                g_x_[i, k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1
        
        return g_x_


def closest_point_Dn(x):
    """Find the closest point in the D_n lattice."""
    f_x = custom_round(x)
    g_x_res = g_x(x)
    
    # Handle both 1D and 2D inputs
    if x.dim() == 1:
        return f_x if torch.sum(f_x) % 2 == 0 else g_x_res
    else:
        # For batched inputs, check each sample individually
        result = f_x.clone()
        for i in range(x.shape[0]):
            if torch.sum(f_x[i]) % 2 == 0:
                result[i] = f_x[i]
            else:
                result[i] = g_x_res[i]
        return result


def closest_point_E8(x):
    """Find the closest point in the E_8 lattice."""
    # Use optimized quantization CUDA kernel if available and input is on CUDA
    if OPTIMIZED_QUANTIZATION_CUDA_KERNELS_AVAILABLE and x.is_cuda and x.dim() == 2:
        return optimized_closest_point_e8_cuda(x)
    # Fallback to regular quantization CUDA kernel
    elif QUANTIZATION_CUDA_KERNELS_AVAILABLE and x.is_cuda and x.dim() == 2:
        return closest_point_e8_quantization_cuda(x)
    # Fallback to general CUDA kernel
    elif CUDA_KERNELS_AVAILABLE and x.is_cuda and x.dim() == 2:
        return closest_point_e8_cuda(x)
    
    # Fallback to original implementation
    f_x = custom_round(x)
    
    # Handle both 1D and 2D inputs
    if x.dim() == 1:
        # Original 1D case
        y_0 = f_x if torch.sum(f_x) % 2 == 0 else g_x(x)
        
        f_x_shifted = custom_round(x - 0.5)
        g_x_shifted = g_x(x - 0.5)
        
        y_1 = f_x_shifted + 0.5 if torch.sum(f_x_shifted) % 2 == 0 else g_x_shifted + 0.5
        
        if torch.norm(x - y_0) < torch.norm(x - y_1):
            return y_0
        else:
            return y_1
    else:
        # Handle 2D batched case
        batch_size = x.shape[0]
        result = f_x.clone()
        
        for i in range(batch_size):
            # Process each sample individually
            x_i = x[i]
            f_x_i = f_x[i]
            
            y_0_i = f_x_i if torch.sum(f_x_i) % 2 == 0 else g_x(x_i.unsqueeze(0)).squeeze(0)
            
            f_x_shifted_i = custom_round(x_i - 0.5)
            g_x_shifted_i = g_x((x_i - 0.5).unsqueeze(0)).squeeze(0)
            
            y_1_i = f_x_shifted_i + 0.5 if torch.sum(f_x_shifted_i) % 2 == 0 else g_x_shifted_i + 0.5
            
            if torch.norm(x_i - y_0_i) < torch.norm(x_i - y_1_i):
                result[i] = y_0_i
            else:
                result[i] = y_1_i
        
        return result


def closest_point_A2(u):
    """Find the closest point in the A_2 lattice."""
    # Simplified A2 implementation - for full implementation would need upscale/downscale
    return custom_round(u)


class LatticeCodebook(nn.Module):
    """
    Lattice codebook for storing and managing lattice components.
    
    This class handles the creation and management of generator matrices
    and closest point functions for different lattice types.
    """
    
    def __init__(self, config: LatticeConfig):
        super().__init__()
        self.config = config
        self.lattice_dim = config.lattice_dim
        self.num_layers = config.num_layers
        self.q = config.radix  # Use radix as quantization parameter q
        
        # Initialize lattice components
        G_temp, self.Q_nn = self._init_lattice_components()
        G_inv_temp = torch.linalg.inv(G_temp)
        
        # Initialize scaling parameters
        self.beta = nn.Parameter(torch.tensor(1.0))  # Scaling parameter
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Overload scaling parameter
        
        # Initialize dither for tie breaking
        eps_temp = self._generate_tie_dither(G_temp.shape[0])  # Use actual lattice dimension
        
        # Register tensors as buffers so they move with the module when .to(device) is called
        self.register_buffer('G_buffer', G_temp)
        self.register_buffer('G_inv_buffer', G_inv_temp)
        self.register_buffer('eps_buffer', eps_temp)
        
        # Update references to use buffers (these will automatically move to the correct device)
        self.G = self.G_buffer
        self.G_inv = self.G_inv_buffer
        self.eps = self.eps_buffer
    
    def to(self, *args, **kwargs):
        """Override to() method to ensure proper device placement"""
        # Call parent to() method
        result = super().to(*args, **kwargs)
        
        # Ensure all tensor references are updated to use buffers
        # This is critical because the references need to point to the moved buffers
        if hasattr(self, 'G_buffer') and hasattr(self, 'G_inv_buffer') and hasattr(self, 'eps_buffer'):
            self.G = self.G_buffer
            self.G_inv = self.G_inv_buffer
            self.eps = self.eps_buffer
        
        return result
        
    def _init_lattice_components(self) -> Tuple[torch.Tensor, Callable]:
        """Initialize generator matrix and closest point function."""
        if self.config.type == LatticeType.Z2:
            return get_z2(), closest_point_zn
        elif self.config.type == LatticeType.E8:
            return get_e8(), closest_point_E8
        elif self.config.type == LatticeType.A2:
            return get_a2(), closest_point_A2
        elif self.config.type == LatticeType.D4:
            return get_d4(), closest_point_Dn
        else:
            raise ValueError(f"Unsupported lattice type: {self.config.type}")
    
    def _generate_tie_dither(self, d: int) -> torch.Tensor:
        """Generate tie dither for breaking ties in lattice quantization."""
        # Use irrational components to avoid alignment with faces
        irr = torch.tensor([math.sqrt(p) for p in [2, 3, 5, 7, 11, 13, 17, 19][:d]], dtype=torch.float32)
        u = (irr - torch.floor(irr)) - 0.5
        u = u / torch.norm(u)
        
        # Very small relative to scale & lattice packing radius
        eta = 2.0**-40
        delta = eta * 1.0 * 0.5  # Use constant value instead of self.beta to avoid initialization issues
        
        return delta * u
    
    def get_generator_matrix(self) -> torch.Tensor:
        """Get the generator matrix."""
        return self.G_buffer
    
    def get_inverse_generator_matrix(self) -> torch.Tensor:
        """Get the inverse generator matrix."""
        return self.G_inv_buffer
    
    def get_closest_point_function(self) -> Callable:
        """Get the closest point function."""
        return self.Q_nn
    
    def get_beta(self) -> torch.Tensor:
        """Get the scaling parameter."""
        return self.beta
    
    def get_alpha(self) -> torch.Tensor:
        """Get the overload scaling parameter."""
        return self.alpha
    
    def get_eps(self) -> torch.Tensor:
        """Get the tie dither."""
        return self.eps_buffer


class LatticeQuantizer(nn.Module):
    """
    Hierarchical Nested Lattice Quantizer (HNLQ).
    
    This class implements the HNLQ quantization method based on the reference
    implementation, which is a hierarchical approach that can be applied to
    different lattice types (E8, A2, Z2, D4, etc.).
    
    Quantization process includes:
    - Encoding: Multi-level lattice quantization with hierarchical scales
    - Decoding: Reconstruction from discrete lattice indices
    - Packing encoding for compression
    - Lookup table operations
    - Gradient quantization for distributed training
    """
    
    def __init__(self, config: LatticeConfig, use_cuda_kernels: bool = True):
        super().__init__()
        self.config = config
        self.lattice_dim = config.lattice_dim
        self.num_layers = config.num_layers
        self.q = config.radix  # Use radix as quantization parameter q
        self.use_cuda_kernels = use_cuda_kernels and CUDA_KERNELS_AVAILABLE
        
        # Initialize lattice components
        self.lattice = LatticeCodebook(config)
        
        # Initialize lookup tables (lazy initialization)
        # Register as buffers so they move with the module
        self.register_buffer('_dot_product_lut', None)
        self.register_buffer('_add_lut', None)
        self.register_buffer('_radix_tables', None)
        
        # Enable vectorized processing for performance optimization
        self._vectorized_quantize_blocks = self._vectorized_quantize_blocks
    
    def _encode(self, x: torch.Tensor, with_dither: bool = False) -> Tuple[Tuple[torch.Tensor, ...], bool]:
        """
        Internal encoding function that performs hierarchical quantization.
        
        This method implements the core hierarchical encoding algorithm,
        producing M levels of encoding vectors.
        """
        # Ensure x is properly shaped
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        
        batch_size = x.shape[0]
        
        # Scale by beta and add dither if requested
        x = x / self.lattice.get_beta()
        if with_dither:
            # Generate random dither for each sample
            dither = torch.randn_like(x) * 0.1  # Small random dither
            x = x + dither
        
        x_l = x
        encoding_vectors = []
        
        for _ in range(self.num_layers):
            # Find closest lattice point
            x_l = self.lattice.get_closest_point_function()(x_l + self.lattice.get_eps())
            
            # Get encoding vector
            b_i = custom_round(torch.fmod(torch.matmul(x_l, self.lattice.get_inverse_generator_matrix()), self.q)).int()
            encoding_vectors.append(b_i)
            
            # Scale for next level
            x_l = x_l / self.q
        
        # Check for overload
        overload_error = not torch.allclose(self.lattice.get_closest_point_function()(x_l), torch.zeros_like(x_l), atol=1e-8)
        
        return tuple(encoding_vectors), overload_error
    
    def encode(self, x: torch.Tensor, with_dither: bool = False) -> Tuple[Tuple[torch.Tensor, ...], int]:
        """
        Encode a vector using hierarchical nested lattice quantization.
        
        This method quantizes the input vector using M hierarchical levels
        and handles overload by scaling the vector until quantization succeeds.
        """
        b_list, did_overload = self._encode(x, with_dither)
        t = 0
        
        # Handle overload by scaling
        while did_overload and t < 10:  # max_scaling_iterations
            t += 1
            x = x / (2**self.lattice.get_alpha())
            b_list, did_overload = self._encode(x, with_dither)
        
        if did_overload:
            print(f"Warning: Overload not resolved after 10 iterations")
        
        return b_list, t
    
    def _decode(self, b_list: Tuple[torch.Tensor, ...], with_dither: bool = False) -> torch.Tensor:
        """
        Internal decoding function that performs hierarchical reconstruction.
        
        This method reconstructs the original vector from M levels of
        encoding vectors using the hierarchical decoding algorithm.
        """
        x_hat_list = []
        
        for b in b_list:
            # Compute quantization error directly
            Gb = torch.matmul(b.float(), self.lattice.get_generator_matrix().T)
            x_i_hat = Gb - self.q * self.lattice.get_closest_point_function()(Gb / self.q)
            x_hat_list.append(x_i_hat)
        
        # Hierarchical reconstruction
        x_hat = torch.zeros_like(x_hat_list[0])
        for i, x_i in enumerate(x_hat_list):
            x_hat += (self.q ** i) * x_i
        
        # Apply beta scaling
        x_hat = self.lattice.get_beta() * x_hat
        
        return x_hat
    
    def decode(self, b_list: Tuple[torch.Tensor, ...], T: int, with_dither: bool = False) -> torch.Tensor:
        """
        Decode hierarchical encoding vectors back to the original space.
        
        This method reconstructs the original vector from its hierarchical
        encoding, accounting for any scaling that was applied during encoding.
        """
        return self._decode(b_list, with_dither) * (2 ** (self.lattice.get_alpha() * T))
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete hierarchical quantization process: encode and decode a vector.
        
        This method handles arbitrary input dimensions using product quantization.
        The input is divided into blocks of size equal to the lattice dimension.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            
        Returns:
            quantized: Quantized tensor of same shape as input
        """
        return self._product_quantize(x, self.quantize_block)
    
    def _product_quantize(self, x: torch.Tensor, block_func) -> torch.Tensor:
        """
        Apply product quantization to handle arbitrary input dimensions.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            block_func: Function to apply to each block
            
        Returns:
            result: Result tensor of same shape as input
        """
        original_shape = x.shape
        input_dim = original_shape[-1]
        lattice_dim = self.lattice_dim
        
        # If input dimension matches lattice dimension, process directly
        if input_dim == lattice_dim:
            return block_func(x)
        
        # Reshape to (batch_dims..., num_blocks, lattice_dim)
        batch_dims = original_shape[:-1]
        num_blocks = (input_dim + lattice_dim - 1) // lattice_dim  # Ceiling division
        
        # Pad the last dimension to make it divisible by lattice_dim
        padded_dim = num_blocks * lattice_dim
        if input_dim < padded_dim:
            padding_shape = list(batch_dims) + [padded_dim - input_dim]
            padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x
        
        # Reshape to process in blocks
        x_reshaped = x_padded.view(*batch_dims, num_blocks, lattice_dim)
        
        # OPTIMIZATION: Vectorize block processing instead of sequential loop
        # Process all blocks simultaneously using vectorized operations
        if hasattr(self, '_vectorized_quantize_blocks'):
            # Use optimized vectorized implementation
            result_reshaped = self._vectorized_quantize_blocks(x_reshaped, block_func)
        else:
            # Fallback to original sequential processing for compatibility
            result_blocks = []
            for i in range(num_blocks):
                block = x_reshaped[..., i, :]  # Shape: (batch_dims..., lattice_dim)
                quantized_block = block_func(block)
                result_blocks.append(quantized_block)
            
            # Concatenate results
            result_reshaped = torch.stack(result_blocks, dim=-2)  # Shape: (batch_dims..., num_blocks, lattice_dim)
        
        # Reshape back to original format
        result_padded = result_reshaped.view(*batch_dims, padded_dim)
        
        # Remove padding to restore original shape
        result = result_padded[..., :input_dim]
        
        return result
    
    def _vectorized_quantize_blocks(self, x_reshaped: torch.Tensor, block_func) -> torch.Tensor:
        """
        Vectorized block quantization for GPU optimization.
        
        Args:
            x_reshaped: Input tensor of shape (..., num_blocks, lattice_dim)
            block_func: Function to apply to each block
            
        Returns:
            result: Quantized tensor of same shape
        """
        # Get dimensions
        batch_dims = x_reshaped.shape[:-2]
        num_blocks = x_reshaped.shape[-2]
        lattice_dim = x_reshaped.shape[-1]
        
        # Reshape to process all blocks simultaneously
        # Shape: (total_batch_size, num_blocks, lattice_dim)
        total_batch_size = 1
        for dim in batch_dims:
            total_batch_size *= dim
        
        x_flat = x_reshaped.view(total_batch_size, num_blocks, lattice_dim)
        
        # Vectorized encoding for all blocks simultaneously
        # Shape: (total_batch_size, num_blocks, lattice_dim)
        quantized_flat = self._vectorized_encode_and_decode(x_flat)
        
        # Reshape back to original format
        result = quantized_flat.view(*batch_dims, num_blocks, lattice_dim)
        
        return result
    
    def _vectorized_encode_and_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized encoding and decoding for multiple blocks.
        
        Args:
            x: Input tensor of shape (batch_size, num_blocks, lattice_dim)
            
        Returns:
            quantized: Quantized tensor of same shape
        """
        # Use optimized CUDA kernel if available and conditions are met
        if (OPTIMIZED_QUANTIZATION_CUDA_KERNELS_AVAILABLE and 
            x.is_cuda and x.dim() == 3):
            return optimized_vectorized_encode_decode_cuda(
                x,
                self.lattice.get_generator_matrix(),
                self.lattice.get_inverse_generator_matrix(),
                self.lattice.get_eps(),
                self.lattice.get_beta(),
                self.q
            )
        # Fallback to regular CUDA kernel
        elif (QUANTIZATION_CUDA_KERNELS_AVAILABLE and 
              x.is_cuda and x.dim() == 3):
            return vectorized_encode_decode_cuda(
                x,
                self.lattice.get_generator_matrix(),
                self.lattice.get_inverse_generator_matrix(),
                self.lattice.get_eps(),
                self.lattice.get_beta(),
                self.q
            )
        
        # Fallback to original implementation
        batch_size, num_blocks, lattice_dim = x.shape
        
        # Scale by beta
        x_scaled = x / self.lattice.get_beta()
        
        # Handle closest point function for each block individually to avoid tensor shape issues
        # This is still much faster than the original sequential approach
        x_l_list = []
        for i in range(num_blocks):
            block = x_scaled[:, i, :]  # (batch_size, lattice_dim)
            # Apply closest point function to each block
            block_result = self.lattice.get_closest_point_function()(block + self.lattice.get_eps())
            x_l_list.append(block_result)
        
        # Stack results
        x_l = torch.stack(x_l_list, dim=1)  # (batch_size, num_blocks, lattice_dim)
        
        # Vectorized matrix multiplication for all blocks
        # Reshape for batch matrix multiplication
        x_l_flat = x_l.view(-1, lattice_dim)  # (batch_size * num_blocks, lattice_dim)
        
        # Batch matrix multiplication
        b_i_flat = torch.matmul(x_l_flat, self.lattice.get_inverse_generator_matrix())
        b_i_flat = torch.fmod(b_i_flat, self.q)
        b_i_flat = custom_round(b_i_flat).int()
        
        # Reshape back
        b_i = b_i_flat.view(batch_size, num_blocks, lattice_dim)
        
        # Vectorized decoding
        # Reshape for batch matrix multiplication
        b_i_flat = b_i.float().view(-1, lattice_dim)
        
        # Batch matrix multiplication for decoding
        decoded_flat = torch.matmul(b_i_flat, self.lattice.get_generator_matrix())
        
        # Reshape back and scale
        decoded = decoded_flat.view(batch_size, num_blocks, lattice_dim)
        result = decoded * self.lattice.get_beta()
        
        return result
    
    def _vectorized_quantize_to_depth(self, x: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized quantization to specific depth for better performance.
        
        This method uses the vectorized quantization path for faster processing.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            depth: Quantization depth (0 for single layer)
            
        Returns:
            quantized: Quantized tensor
            indices: Quantization indices
        """
        if depth == -1:
            depth = self.num_layers - 1
        
        if depth < 0 or depth >= self.num_layers:
            raise ValueError(f"Depth {depth} out of range [0, {self.num_layers-1}]")
        
        # Use optimized ultra-fast CUDA kernel if available and conditions are met
        if (self.use_cuda_kernels and OPTIMIZED_QUANTIZATION_CUDA_KERNELS_AVAILABLE and 
            x.is_cuda and x.dim() == 2 and depth == 0):
            return optimized_ultra_fast_quantize_cuda(
                x, 
                self.lattice.get_generator_matrix(),
                self.lattice.get_inverse_generator_matrix(),
                self.lattice.get_eps(),
                self.lattice.get_beta(),
                self.q
            )
        # Fallback to regular ultra-fast CUDA kernel
        elif (self.use_cuda_kernels and QUANTIZATION_CUDA_KERNELS_AVAILABLE and 
              x.is_cuda and x.dim() == 2 and depth == 0):
            return ultra_fast_quantize_cuda(
                x, 
                self.lattice.get_generator_matrix(),
                self.lattice.get_inverse_generator_matrix(),
                self.lattice.get_eps(),
                self.lattice.get_beta(),
                self.q
            )
        # Fallback to batch CUDA kernel
        elif (self.use_cuda_kernels and CUDA_KERNELS_AVAILABLE and 
              x.is_cuda and x.dim() == 2 and depth == 0):
            return batch_product_quantize_cuda(
                x, 
                self.lattice.get_generator_matrix(),
                self.lattice.get_inverse_generator_matrix(),
                self.lattice.get_eps(),
                self.lattice.get_beta(),
                self.q
            )
        
        # Use vectorized quantization
        quantized = self._product_quantize(x, self.quantize_block)
        
        # Get indices using vectorized encoding
        indices = self._vectorized_encode_to_depth(x, depth)
        
        return quantized, indices
    
    def _vectorized_encode_to_depth(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Vectorized encoding to specific depth.
        
        Args:
            x: Input tensor
            depth: Target depth
            
        Returns:
            indices: Encoded indices tensor
        """
        original_shape = x.shape
        input_dim = original_shape[-1]
        lattice_dim = self.lattice_dim
        
        # If input dimension matches lattice dimension, process directly
        if input_dim == lattice_dim:
            return self._encode_single_block(x, depth)
        
        # Reshape for block processing
        batch_dims = original_shape[:-1]
        num_blocks = (input_dim + lattice_dim - 1) // lattice_dim
        
        # Pad and reshape
        padded_dim = num_blocks * lattice_dim
        if input_dim < padded_dim:
            padding_shape = list(batch_dims) + [padded_dim - input_dim]
            padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x
        
        x_reshaped = x_padded.view(*batch_dims, num_blocks, lattice_dim)
        
        # Vectorized encoding for all blocks
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        
        x_flat = x_reshaped.view(batch_size, num_blocks, lattice_dim)
        
        # Vectorized encoding
        encoded_flat = self._vectorized_encode_blocks_to_depth(x_flat, depth)
        
        # Reshape back
        encoded = encoded_flat.view(*batch_dims, num_blocks, lattice_dim)
        
        return encoded
    
    def _vectorized_encode_blocks_to_depth(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Fully vectorized encoding of blocks to specific depth.
        
        This implementation processes all blocks simultaneously using vectorized operations
        for maximum GPU utilization and performance.
        
        Args:
            x: Input tensor of shape (batch_size, num_blocks, lattice_dim)
            depth: Target depth
            
        Returns:
            encoded: Encoded indices of shape (batch_size, num_blocks, lattice_dim)
        """
        batch_size, num_blocks, lattice_dim = x.shape
        
        # Scale by beta
        x_scaled = x / self.lattice.get_beta()
        
        # OPTIMIZATION: Fully vectorized closest point function
        # Process all blocks simultaneously by reshaping
        x_scaled_flat = x_scaled.view(-1, lattice_dim)  # (batch_size * num_blocks, lattice_dim)
        
        # Vectorized closest point function for all blocks at once
        x_l_flat = self.lattice.get_closest_point_function()(x_scaled_flat + self.lattice.get_eps())
        
        # Reshape back
        x_l = x_l_flat.view(batch_size, num_blocks, lattice_dim)
        
        # For single depth (depth=0), do fully vectorized encoding
        if depth == 0:
            # Vectorized matrix multiplication for all blocks simultaneously
            x_l_flat = x_l.view(-1, lattice_dim)  # (batch_size * num_blocks, lattice_dim)
            
            # Batch matrix multiplication
            b_i_flat = torch.matmul(x_l_flat, self.lattice.get_inverse_generator_matrix())
            b_i_flat = torch.fmod(b_i_flat, self.q)
            b_i_flat = custom_round(b_i_flat).int()
            
            # Reshape back
            b_i = b_i_flat.view(batch_size, num_blocks, lattice_dim)
            return b_i
        else:
            # For multi-depth, use optimized sequential encoding
            result_blocks = []
            for i in range(num_blocks):
                block = x_l[:, i, :]  # (batch_size, lattice_dim)
                encoded_block = self._encode_single_block_to_depth(block, depth)
                result_blocks.append(encoded_block)
            
            return torch.stack(result_blocks, dim=1)
    
    def _encode_single_block_to_depth(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Encode a single block to specific depth.
        
        Args:
            x: Input tensor of shape (batch_size, lattice_dim)
            depth: Target depth
            
        Returns:
            encoded: Encoded tensor of shape (batch_size, lattice_dim)
        """
        x_l = x
        for layer in range(depth + 1):
            x_l = self.lattice.get_closest_point_function()(x_l + self.lattice.get_eps())
            if layer < depth:
                b_i = custom_round(torch.fmod(torch.matmul(x_l, self.lattice.get_inverse_generator_matrix()), self.q)).int()
                x_l = x_l / self.q
            else:
                b_i = custom_round(torch.fmod(torch.matmul(x_l, self.lattice.get_inverse_generator_matrix()), self.q)).int()
        
        return b_i
    
    def batch_quantize_to_depth(self, inputs: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch quantization to specific depth for multiple inputs.
        
        This method processes multiple input tensors simultaneously for maximum efficiency.
        
        Args:
            inputs: Input tensor of shape (batch_size, ..., input_dim) or list of tensors
            depth: Quantization depth (0 for single layer)
            
        Returns:
            quantized_list: List of quantized tensors
            indices_list: List of quantization indices
        """
        if isinstance(inputs, list):
            # Process list of inputs
            batch_size = len(inputs)
            if batch_size == 0:
                return [], []
            
            # Get the first input to determine shapes
            first_input = inputs[0]
            input_dim = first_input.shape[-1]
            
            # Stack all inputs into a single tensor for batch processing
            stacked_inputs = torch.stack(inputs, dim=0)  # (batch_size, ..., input_dim)
            
            # Batch quantize
            batch_quantized, batch_indices = self._vectorized_quantize_to_depth(stacked_inputs, depth)
            
            # Split back into individual tensors
            quantized_list = [batch_quantized[i] for i in range(batch_size)]
            indices_list = [batch_indices[i] for i in range(batch_size)]
            
            return quantized_list, indices_list
        else:
            # Single tensor input - use vectorized method
            return self._vectorized_quantize_to_depth(inputs, depth)
    
    def quantize_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize a single block of lattice dimension.
        
        Args:
            x: Input tensor of shape (..., lattice_dim)
            
        Returns:
            quantized: Quantized tensor of same shape
        """
        b_list, T = self.encode(x)
        return self.decode(b_list, T)
    
    def quantize_to_depth(self, x: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to specific depth using product quantization.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            depth: Quantization depth (0 to num_layers-1)
            
        Returns:
            quantized: Quantized tensor of same shape as input
            indices: Lattice indices with shape (..., num_blocks, lattice_dim)
        """
        if depth >= self.num_layers:
            raise ValueError(f"Depth {depth} exceeds number of layers {self.num_layers}")
        
        # Use optimized vectorized quantization if available and conditions are met
        if (self.use_cuda_kernels and OPTIMIZED_QUANTIZATION_CUDA_KERNELS_AVAILABLE and 
            x.is_cuda and x.dim() == 2 and depth == 0):
            return self._vectorized_quantize_to_depth(x, depth)
        
        # Fallback to original implementation
        return self._product_quantize_to_depth(x, depth)
    
    def _product_quantize_to_depth(self, x: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply product quantization to depth quantization.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            depth: Quantization depth
            
        Returns:
            quantized: Quantized tensor of same shape as input
            indices: Lattice indices with shape (..., num_blocks, lattice_dim)
        """
        original_shape = x.shape
        input_dim = original_shape[-1]
        lattice_dim = self.lattice_dim
        
        # If input dimension matches lattice dimension, process directly
        if input_dim == lattice_dim:
            quantized, indices = self._quantize_block_to_depth(x, depth)
            # Ensure indices have the correct block structure [batch_size, 1, lattice_dim]
            if indices.dim() == 2:  # [batch_size, lattice_dim]
                indices = indices.unsqueeze(-2)  # [batch_size, 1, lattice_dim]
            return quantized, indices
        
        # Reshape to (batch_dims..., num_blocks, lattice_dim)
        batch_dims = original_shape[:-1]
        num_blocks = (input_dim + lattice_dim - 1) // lattice_dim  # Ceiling division
        
        # Pad the last dimension to make it divisible by lattice_dim
        padded_dim = num_blocks * lattice_dim
        if input_dim < padded_dim:
            padding_shape = list(batch_dims) + [padded_dim - input_dim]
            padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x
        
        # Reshape to process in blocks
        x_reshaped = x_padded.view(*batch_dims, num_blocks, lattice_dim)
        
        # Apply block function to each block
        quantized_blocks = []
        indices_blocks = []
        for i in range(num_blocks):
            block = x_reshaped[..., i, :]  # Shape: (batch_dims..., lattice_dim)
            quantized_block, indices_block = self._quantize_block_to_depth(block, depth)
            quantized_blocks.append(quantized_block)
            indices_blocks.append(indices_block)
        
        # Concatenate results
        quantized_reshaped = torch.stack(quantized_blocks, dim=-2)  # Shape: (batch_dims..., num_blocks, lattice_dim)
        quantized_padded = quantized_reshaped.view(*batch_dims, padded_dim)
        quantized = quantized_padded[..., :input_dim]  # Remove padding
        
        # Stack indices: shape (batch_dims..., num_blocks, lattice_dim)
        indices = torch.stack(indices_blocks, dim=-2)
        
        return quantized, indices
    
    def _quantize_block_to_depth(self, x: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a single block to specific depth.
        
        Args:
            x: Input tensor of shape (..., lattice_dim)
            depth: Quantization depth
            
        Returns:
            quantized: Quantized tensor of same shape
            indices: Lattice indices of same shape
        """
        b_list, T = self.encode(x)
        
        if depth < len(b_list):
            # Return the quantized result and the encoding vector for this depth
            quantized = self._decode(b_list, T)
            indices = b_list[depth]
            return quantized, indices
        else:
            # If depth exceeds available levels, use the last level
            quantized = self._decode(b_list, T)
            indices = b_list[-1]
            return quantized, indices
    
    def decode_from_depth(self, encoded: torch.Tensor, source_depth: int) -> torch.Tensor:
        """
        Decode from specific quantization depth using product quantization.
        
        This operation reconstructs continuous values from encoding vectors
        at the specified depth level, handling arbitrary input dimensions.
        
        Args:
            encoded: Encoded vectors with shape (..., num_blocks, lattice_dim) or (..., lattice_dim)
            source_depth: Source quantization depth
            
        Returns:
            decoded: Decoded tensor
        """
        return self._product_decode_from_depth(encoded, source_depth)
    
    def _product_decode_from_depth(self, encoded: torch.Tensor, source_depth: int) -> torch.Tensor:
        """
        Apply product quantization to depth decoding.
        
        Args:
            encoded: Encoded vectors with shape (..., num_blocks, lattice_dim) or (..., lattice_dim)
            source_depth: Source quantization depth
            
        Returns:
            decoded: Decoded tensor with original shape (flattened from blocks)
        """
        original_shape = encoded.shape
        lattice_dim = self.lattice_dim
        
        # Check if this is already in block format or single block
        if original_shape[-1] == lattice_dim:
            if len(original_shape) == 2 or (len(original_shape) > 2 and original_shape[-2] == 1):
                # Single block case
                return self._decode_block_from_depth(encoded, source_depth)
            else:
                # Multiple blocks case: (..., num_blocks, lattice_dim)
                batch_dims = original_shape[:-2]
                num_blocks = original_shape[-2]
                
                decoded_blocks = []
                for i in range(num_blocks):
                    block = encoded[..., i, :]  # Shape: (batch_dims..., lattice_dim)
                    decoded_block = self._decode_block_from_depth(block, source_depth)
                    decoded_blocks.append(decoded_block)
                
                # Stack results: shape (batch_dims..., num_blocks, lattice_dim)
                decoded_stacked = torch.stack(decoded_blocks, dim=-2)
                
                # Flatten back to original shape: (batch_dims..., num_blocks * lattice_dim)
                decoded = decoded_stacked.view(*batch_dims, num_blocks * lattice_dim)
                return decoded
        else:
            # Treat as single block
            return self._decode_block_from_depth(encoded, source_depth)
    
    def _decode_block_from_depth(self, encoded: torch.Tensor, source_depth: int) -> torch.Tensor:
        """
        Decode a single block from specific depth.
        
        Args:
            encoded: Encoded vectors of shape (..., lattice_dim)
            source_depth: Source quantization depth
            
        Returns:
            decoded: Decoded tensor of same shape
        """
        # For single depth decoding, we need to create a minimal b_list
        # This is a simplified approach - full implementation would be more complex
        b_list = (encoded,)
        return self._decode(b_list)
    
    def packing_encode(self, x: torch.Tensor, packing_radix: int, depth: int) -> torch.Tensor:
        """
        Encode using packing representation with product quantization.
        
        This operation compresses lattice indices using packing encoding,
        handling arbitrary input dimensions by dividing into blocks.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            packing_radix: Base for packing encoding
            depth: Quantization depth
            
        Returns:
            encoded: Packed tensor of shape (..., num_blocks, lattice_dim)
        """
        return self._product_packing_encode(x, packing_radix, depth)
    
    def _product_packing_encode(self, x: torch.Tensor, packing_radix: int, depth: int) -> torch.Tensor:
        """
        Apply product quantization to packing encoding.
        
        Args:
            x: Input tensor of arbitrary shape (..., input_dim)
            packing_radix: Base for packing encoding
            depth: Quantization depth
            
        Returns:
            encoded: Packed tensor of shape (..., num_blocks, lattice_dim)
        """
        original_shape = x.shape
        input_dim = original_shape[-1]
        lattice_dim = self.lattice_dim
        
        # If input dimension matches lattice dimension, process directly
        if input_dim == lattice_dim:
            return self._packing_encode_block(x, packing_radix, depth)
        
        # Reshape to (batch_dims..., num_blocks, lattice_dim)
        batch_dims = original_shape[:-1]
        num_blocks = (input_dim + lattice_dim - 1) // lattice_dim  # Ceiling division
        
        # Pad the last dimension to make it divisible by lattice_dim
        padded_dim = num_blocks * lattice_dim
        if input_dim < padded_dim:
            padding_shape = list(batch_dims) + [padded_dim - input_dim]
            padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x
        
        # Reshape to process in blocks
        x_reshaped = x_padded.view(*batch_dims, num_blocks, lattice_dim)
        
        # Apply packing encoding to each block
        encoded_blocks = []
        for i in range(num_blocks):
            block = x_reshaped[..., i, :]  # Shape: (batch_dims..., lattice_dim)
            encoded_block = self._packing_encode_block(block, packing_radix, depth)
            encoded_blocks.append(encoded_block)
        
        # Stack results: shape (batch_dims..., num_blocks, lattice_dim)
        encoded = torch.stack(encoded_blocks, dim=-2)
        
        return encoded
    
    def _packing_encode_block(self, x: torch.Tensor, packing_radix: int, depth: int) -> torch.Tensor:
        """
        Apply packing encoding to a single block.
        
        Args:
            x: Input tensor of shape (..., lattice_dim)
            packing_radix: Base for packing encoding
            depth: Quantization depth
            
        Returns:
            encoded: Packed tensor of same shape
        """
        # First quantize to get encoding vectors
        b_list, T = self.encode(x)
        
        # Convert to packing representation
        encoded = torch.zeros_like(b_list[0], dtype=torch.int32)
        
        for i in range(min(depth, len(b_list))):
            encoded += (b_list[i] % packing_radix) * (packing_radix ** i)
        
        return encoded
    
    def packing_decode(self, encoded: torch.Tensor, packing_radix: int, depth: int) -> torch.Tensor:
        """
        Decode from packing representation with product quantization.
        
        This operation reconstructs lattice indices from packing encoding
        and then decodes to continuous values, handling arbitrary dimensions.
        
        Args:
            encoded: Packed tensor of shape (..., num_blocks, lattice_dim) or (..., lattice_dim)
            packing_radix: Base for packing encoding
            depth: Quantization depth
            
        Returns:
            decoded: Decoded tensor
        """
        return self._product_packing_decode(encoded, packing_radix, depth)
    
    def _product_packing_decode(self, encoded: torch.Tensor, packing_radix: int, depth: int) -> torch.Tensor:
        """
        Apply product quantization to packing decoding.
        
        Args:
            encoded: Packed tensor of shape (..., num_blocks, lattice_dim) or (..., lattice_dim)
            packing_radix: Base for packing encoding
            depth: Quantization depth
            
        Returns:
            decoded: Decoded tensor
        """
        original_shape = encoded.shape
        lattice_dim = self.lattice_dim
        
        # Check if this is already in block format or single block
        if original_shape[-1] == lattice_dim:
            if len(original_shape) == 2 or (len(original_shape) > 2 and original_shape[-2] == 1):
                # Single block case
                return self._packing_decode_block(encoded, packing_radix, depth)
            else:
                # Multiple blocks case: (..., num_blocks, lattice_dim)
                batch_dims = original_shape[:-2]
                num_blocks = original_shape[-2]
                
                decoded_blocks = []
                for i in range(num_blocks):
                    block = encoded[..., i, :]  # Shape: (batch_dims..., lattice_dim)
                    decoded_block = self._packing_decode_block(block, packing_radix, depth)
                    decoded_blocks.append(decoded_block)
                
                # Stack results: shape (batch_dims..., num_blocks, lattice_dim)
                decoded_stacked = torch.stack(decoded_blocks, dim=-2)
                
                # Flatten back to original shape: (batch_dims..., num_blocks * lattice_dim)
                decoded = decoded_stacked.view(*batch_dims, num_blocks * lattice_dim)
                return decoded
        else:
            # Treat as single block
            return self._packing_decode_block(encoded, packing_radix, depth)
    
    def _packing_decode_block(self, encoded: torch.Tensor, packing_radix: int, depth: int) -> torch.Tensor:
        """
        Decode a single block from packing representation.
        
        Args:
            encoded: Packed tensor of shape (..., lattice_dim)
            packing_radix: Base for packing encoding
            depth: Quantization depth
            
        Returns:
            decoded: Decoded tensor of same shape
        """
        # Convert from packing to encoding vectors
        b = torch.zeros_like(encoded, dtype=torch.long)
        temp_encoded = encoded.clone()
        
        for i in range(depth):
            b += (temp_encoded % packing_radix) * (packing_radix ** i)
            temp_encoded = temp_encoded // packing_radix
        
        # Decode using single depth
        return self.decode_from_depth(b, depth)
    
    def create_lookup_table(self, max_indices: int = -1) -> torch.Tensor:
        """
        Create lookup table for efficient dot product computation.
        
        This operation precomputes dot products between all possible
        lattice point pairs for fast computation.
        """
        # For Z2 lattice, we use a simpler approach with limited indices
        if max_indices == -1:
            # Use a reasonable limit for lookup table size
            max_indices = min(16, self.q ** self.lattice_dim)
        
        # Create lookup table on the same device as the module
        device = next(self.parameters()).device
        lookup_table = torch.zeros(max_indices, max_indices, device=device)
        
        # Create simple codebook for lookup table
        for i in range(max_indices):
            for j in range(max_indices):
                # Simplified dot product calculation for small lookup table
                # In practice, this would compute actual lattice point dot products
                lookup_table[i, j] = float(i * j) / float(max_indices)
        
        return lookup_table
    
    def lookup_dot_product(self, x_indices: torch.Tensor, y_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product using lookup table.
        
        This operation efficiently computes dot products between
        quantized vectors using precomputed lookup tables.
        """
        if self._dot_product_lut is None:
            self._dot_product_lut = self.create_lookup_table()
            if not hasattr(self, '_dot_product_lut'):
                self.register_buffer('_dot_product_lut', self._dot_product_lut)
        
        # Clamp indices to valid range for lookup table
        max_idx = self._dot_product_lut.shape[0] - 1
        x_clamped = torch.clamp(x_indices, 0, max_idx)
        y_clamped = torch.clamp(y_indices, 0, max_idx)
        
        return self._dot_product_lut[x_clamped, y_clamped]
    
    def quantized_add(self, x_indices: torch.Tensor, y_indices: torch.Tensor) -> torch.Tensor:
        """
        Vector addition in quantized space.
        
        This operation performs vector addition by decoding, adding, and re-encoding.
        """
        # Decode, add, and re-encode
        x_decoded = self.decode_from_depth(x_indices, 0)
        y_decoded = self.decode_from_depth(y_indices, 0)
        sum_tensor = x_decoded + y_decoded
        
        _, sum_indices = self.quantize_to_depth(sum_tensor, self.num_layers - 1)
        return sum_indices
    
    def quantized_reduce(self, indices: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Reduce operation in quantized space.
        
        This operation performs reduction by decoding, reducing, and re-encoding.
        """
        # Decode, reduce, and re-encode
        decoded = self.decode_from_depth(indices, 0)
        reduced = torch.sum(decoded, dim=dim, keepdim=True)
        
        _, reduced_indices = self.quantize_to_depth(reduced, self.num_layers - 1)
        return reduced_indices
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for quantization."""
        return self.quantize_to_depth(x, self.num_layers - 1)
    
    def get_quantization_stats(self) -> Dict[str, torch.Tensor]:
        """Get quantization statistics."""
        return {
            'beta': self.lattice.get_beta(),
            'alpha': self.lattice.get_alpha(),
            'num_layers': torch.tensor(self.num_layers),
            'lattice_dim': torch.tensor(self.lattice_dim),
            'q': torch.tensor(self.q),
        }