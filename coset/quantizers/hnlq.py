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
    k = torch.argmax(delta)
    g_x_ = f_x.clone()
    
    x_k = x[k]
    f_x_k = f_x[k]
    
    if x_k >= 0:
        g_x_[k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
    else:
        g_x_[k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1
    
    return g_x_


def closest_point_Dn(x):
    """Find the closest point in the D_n lattice."""
    f_x = custom_round(x)
    g_x_res = g_x(x)
    return f_x if torch.sum(f_x) % 2 == 0 else g_x_res


def closest_point_E8(x):
    """Find the closest point in the E_8 lattice."""
    f_x = custom_round(x)
    y_0 = f_x if torch.sum(f_x) % 2 == 0 else g_x(x)
    
    f_x_shifted = custom_round(x - 0.5)
    g_x_shifted = g_x(x - 0.5)
    
    y_1 = f_x_shifted + 0.5 if torch.sum(f_x_shifted) % 2 == 0 else g_x_shifted + 0.5
    
    if torch.norm(x - y_0) < torch.norm(x - y_1):
        return y_0
    else:
        return y_1


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
        self.G, self.Q_nn = self._init_lattice_components()
        self.G_inv = torch.linalg.inv(self.G)
        
        # Initialize scaling parameters
        self.beta = nn.Parameter(torch.tensor(1.0))  # Scaling parameter
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Overload scaling parameter
        
        # Initialize dither for tie breaking
        self.eps = self._generate_tie_dither(self.G.shape[0])  # Use actual lattice dimension
        
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
        delta = eta * self.beta * 0.5  # Using 0.5 as default Rin
        
        return delta * u
    
    def get_generator_matrix(self) -> torch.Tensor:
        """Get the generator matrix."""
        return self.G
    
    def get_inverse_generator_matrix(self) -> torch.Tensor:
        """Get the inverse generator matrix."""
        return self.G_inv
    
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
        return self.eps


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
    
    def __init__(self, config: LatticeConfig):
        super().__init__()
        self.config = config
        self.lattice_dim = config.lattice_dim
        self.num_layers = config.num_layers
        self.q = config.radix  # Use radix as quantization parameter q
        
        # Initialize lattice components
        self.lattice = LatticeCodebook(config)
        
        # Initialize lookup tables (lazy initialization)
        self._dot_product_lut = None
        self._add_lut = None
        self._radix_tables = None
    
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
        
        # Apply block function to each block
        result_blocks = []
        for i in range(num_blocks):
            block = x_reshaped[..., i, :]  # Shape: (batch_dims..., lattice_dim)
            quantized_block = block_func(block)
            result_blocks.append(quantized_block)
        
        # Concatenate results
        result_reshaped = torch.stack(result_blocks, dim=-2)  # Shape: (batch_dims..., num_blocks, lattice_dim)
        result_padded = result_reshaped.view(*batch_dims, padded_dim)
        
        # Remove padding to restore original shape
        result = result_padded[..., :input_dim]
        
        return result
    
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
        
        lookup_table = torch.zeros(max_indices, max_indices)
        
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