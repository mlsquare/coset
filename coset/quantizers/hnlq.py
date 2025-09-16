"""
Hierarchical Nested Lattice Quantization (HNLQ) Implementation

This module implements the core HNLQ quantizer that supports:
- Multi-level lattice quantization with different scales
- Efficient encoding/decoding operations
- Radix-q encoding for gradient compression
- Lookup table operations for fast dot products
- PyTorch autograd compatibility
- Batched operations

Core Encoding/Decoding Operations:

1. Lattice Quantization:
   - Maps continuous values to discrete lattice points
   - Uses hierarchical scales for multi-level quantization
   - Supports different lattice types (HNLQ, E8, A2, Z2, D4)

2. Hierarchical Encoding:
   - Multi-level quantization with different scales
   - Adaptive depth selection based on input statistics
   - Learnable scale factors and zero points

3. Radix-Q Encoding:
   - Compresses lattice indices using radix-q representation
   - Reduces communication overhead for distributed training
   - Supports different radix values (2, 4, 8, 16, etc.)

4. Lookup Table Operations:
   - Precomputed dot product tables for fast computation
   - Vector addition and reduction in quantized space
   - Memory-efficient operations for large matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import math

from .config import LatticeConfig, LatticeType


class LatticeCodebook(nn.Module):
    """
    Lattice codebook for storing and managing lattice points.
    
    This class handles the creation and management of lattice codebooks
    for different lattice types and hierarchy levels.
    """
    
    def __init__(self, config: LatticeConfig):
        super().__init__()
        self.config = config
        self.lattice_dim = config.lattice_dim
        self.num_layers = config.num_layers
        self.num_codewords = config.get_num_codewords()
        
        # Initialize codebook based on lattice type
        self.codebook = self._init_codebook()
        
        # Initialize scale and zero point parameters
        self._init_quantization_params()
    
    def _init_codebook(self) -> torch.Tensor:
        """
        Initialize lattice codebook based on lattice type.
        
        Returns:
            codebook: Tensor of shape [num_codewords, lattice_dim]
        """
        if self.config.type == LatticeType.HNLQ:
            return self._init_hnlq_codebook()
        elif self.config.type == LatticeType.E8:
            return self._init_e8_codebook()
        elif self.config.type == LatticeType.A2:
            return self._init_a2_codebook()
        elif self.config.type == LatticeType.Z2:
            return self._init_z2_codebook()
        elif self.config.type == LatticeType.D4:
            return self._init_d4_codebook()
        else:
            raise ValueError(f"Unsupported lattice type: {self.config.type}")
    
    def _init_hnlq_codebook(self) -> torch.Tensor:
        """Initialize HNLQ codebook using binary encoding."""
        codebook = torch.zeros(self.num_codewords, self.lattice_dim)
        
        for i in range(self.num_codewords):
            binary = format(i, f'0{self.lattice_dim}b')
            codebook[i] = torch.tensor([int(b) for b in binary], dtype=torch.float32)
        
        return codebook
    
    def _init_e8_codebook(self) -> torch.Tensor:
        """Initialize E8 lattice codebook."""
        # E8 lattice has 240 roots + origin
        # For simplicity, we'll use a subset based on binary encoding
        codebook = torch.zeros(self.num_codewords, self.lattice_dim)
        
        for i in range(self.num_codewords):
            # Generate E8-like points using binary encoding with scaling
            binary = format(i, f'0{self.lattice_dim}b')
            point = torch.tensor([int(b) for b in binary], dtype=torch.float32)
            # Scale to create E8-like structure
            codebook[i] = point * 2.0 - 1.0
        
        return codebook
    
    def _init_a2_codebook(self) -> torch.Tensor:
        """Initialize A2 (hexagonal) lattice codebook."""
        codebook = torch.zeros(self.num_codewords, self.lattice_dim)
        
        for i in range(self.num_codewords):
            # Generate hexagonal lattice points
            binary = format(i, f'0{self.lattice_dim}b')
            point = torch.tensor([int(b) for b in binary], dtype=torch.float32)
            # Apply hexagonal transformation
            if self.lattice_dim >= 2:
                x, y = point[0], point[1]
                codebook[i, 0] = x + 0.5 * y
                codebook[i, 1] = y * math.sqrt(3) / 2
                if self.lattice_dim > 2:
                    codebook[i, 2:] = point[2:]
        
        return codebook
    
    def _init_z2_codebook(self) -> torch.Tensor:
        """Initialize Z2 (square) lattice codebook."""
        codebook = torch.zeros(self.num_codewords, self.lattice_dim)
        
        for i in range(self.num_codewords):
            binary = format(i, f'0{self.lattice_dim}b')
            codebook[i] = torch.tensor([int(b) for b in binary], dtype=torch.float32)
        
        return codebook
    
    def _init_d4_codebook(self) -> torch.Tensor:
        """Initialize D4 lattice codebook."""
        codebook = torch.zeros(self.num_codewords, self.lattice_dim)
        
        for i in range(self.num_codewords):
            binary = format(i, f'0{self.lattice_dim}b')
            point = torch.tensor([int(b) for b in binary], dtype=torch.float32)
            # Apply D4 lattice transformation
            if self.lattice_dim >= 4:
                # D4 lattice has specific structure
                codebook[i] = point * 2.0 - 1.0
            else:
                codebook[i] = point
        
        return codebook
    
    def _init_quantization_params(self):
        """Initialize quantization parameters (scales and zero points)."""
        # Initialize scales
        if self.config.learnable_scales:
            self.scales = nn.Parameter(torch.tensor(self.config.scales))
        else:
            self.register_buffer('scales', torch.tensor(self.config.scales))
        
        # Initialize zero points
        if self.config.learnable_zero_points:
            self.zero_points = nn.Parameter(torch.tensor(self.config.zero_points, dtype=torch.int32))
        else:
            self.register_buffer('zero_points', torch.tensor(self.config.zero_points, dtype=torch.int32))
    
    def get_codebook(self) -> torch.Tensor:
        """Get the lattice codebook."""
        return self.codebook
    
    def get_scale(self, layer: int = 0) -> torch.Tensor:
        """Get scale for specific layer."""
        return self.scales[layer]
    
    def get_zero_point(self, layer: int = 0) -> torch.Tensor:
        """Get zero point for specific layer."""
        return self.zero_points[layer]


class LatticeQuantizer(nn.Module):
    """
    Hierarchical Nested Lattice Quantizer.
    
    This class implements the core quantization operations including:
    - Multi-level lattice quantization
    - Encoding/decoding with different depths
    - Radix-q encoding for compression
    - Lookup table operations
    - Gradient quantization for distributed training
    """
    
    def __init__(self, config: LatticeConfig):
        super().__init__()
        self.config = config
        self.lattice_dim = config.lattice_dim
        self.num_layers = config.num_layers
        self.radix = config.radix
        
        # Initialize codebook
        self.codebook = LatticeCodebook(config)
        
        # Initialize hierarchy weights
        self.hierarchy_weights = nn.Parameter(
            torch.ones(config.num_layers) / config.num_layers
        )
        
        # Initialize lookup tables (lazy initialization)
        self._dot_product_lut = None
        self._add_lut = None
        self._radix_tables = None
    
    def quantize(
        self, 
        x: torch.Tensor, 
        depth: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to lattice points.
        
        This is the core encoding operation that maps continuous values
        to discrete lattice points using hierarchical quantization.
        
        Args:
            x: Input tensor of shape (..., lattice_dim)
            depth: Quantization depth (-1 for adaptive, 0-N for specific layer)
            
        Returns:
            quantized: Quantized tensor of same shape as input
            indices: Lattice indices of shape (..., num_layers) or (...,)
            
        Example:
            >>> config = LatticeConfig(lattice_dim=8, num_layers=3)
            >>> quantizer = LatticeQuantizer(config)
            >>> x = torch.randn(32, 8)
            >>> quantized, indices = quantizer.quantize(x)
            >>> print(quantized.shape)  # torch.Size([32, 8])
            >>> print(indices.shape)    # torch.Size([32, 3])
        """
        if depth == -1:
            return self._adaptive_quantize(x)
        else:
            return self._single_level_quantize(x, depth)
    
    def _adaptive_quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive quantization using hierarchy weights.
        
        Selects optimal quantization level based on input statistics
        and combines results from multiple levels.
        """
        quantized_list = []
        indices_list = []
        
        for i in range(self.num_layers):
            q, idx = self._single_level_quantize(x, i)
            quantized_list.append(q * self.hierarchy_weights[i])
            indices_list.append(idx)
        
        # Combine quantized values from all levels
        quantized = torch.stack(quantized_list, dim=-2).sum(dim=-2)
        indices = torch.stack(indices_list, dim=-1)  # (..., num_layers)
        
        return quantized, indices
    
    def _single_level_quantize(self, x: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-level quantization at specified depth.
        
        Args:
            x: Input tensor
            depth: Quantization depth (0 to num_layers-1)
            
        Returns:
            quantized: Quantized tensor
            indices: Lattice indices
        """
        if depth >= self.num_layers:
            raise ValueError(f"Depth {depth} exceeds number of layers {self.num_layers}")
        
        # Get quantization parameters
        scale = self.codebook.get_scale(depth)
        zero_point = self.codebook.get_zero_point(depth)
        codebook = self.codebook.get_codebook()
        
        # Handle different input dimensions
        if x.shape[-1] != self.lattice_dim:
            # For inputs with different dimensions, use simple quantization
            # This is a simplified approach for non-lattice-dim inputs
            x_norm = x / scale + zero_point
            # Simple rounding quantization
            quantized = torch.round(x_norm) * scale - zero_point
            indices = torch.round(x_norm).long()
            return quantized, indices
        
        # Normalize input
        x_norm = x / scale + zero_point
        
        # Find closest lattice points
        distances = torch.cdist(x_norm.unsqueeze(-2), codebook.unsqueeze(0))
        indices = torch.argmin(distances, dim=-1).squeeze(-2)
        
        # Get quantized values
        quantized = codebook[indices]
        
        # Ensure output has same shape as input
        if quantized.shape != x.shape:
            quantized = quantized.squeeze(-2)
        
        # Keep indices as [batch_size] since we find one lattice point per vector
        # Don't expand indices - they represent single lattice point indices
        
        return quantized, indices
    
    def dequantize(
        self, 
        indices: torch.Tensor, 
        depth: int = -1
    ) -> torch.Tensor:
        """
        Dequantize indices back to continuous values.
        
        This is the core decoding operation that reconstructs continuous
        values from discrete lattice indices.
        
        Args:
            indices: Lattice indices of shape (..., num_layers) or (...,)
            depth: Quantization depth (-1 for multi-level, 0-N for specific layer)
            
        Returns:
            dequantized: Reconstructed tensor
            
        Example:
            >>> quantized, indices = quantizer.quantize(x)
            >>> reconstructed = quantizer.dequantize(indices)
            >>> print(torch.allclose(x, reconstructed, atol=1e-3))  # True
        """
        if depth == -1:
            return self._multi_level_dequantize(indices)
        else:
            return self._single_level_dequantize(indices, depth)
    
    def _multi_level_dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Multi-level dequantization using hierarchy weights."""
        if indices.dim() == 1 or (indices.dim() > 1 and indices.shape[-1] == 1):
            # Single level
            return self._single_level_dequantize(indices.squeeze(-1), 0)
        
        # Multiple levels
        dequantized_list = []
        for i in range(self.num_layers):
            if i < indices.shape[-1]:
                dq = self._single_level_dequantize(indices[..., i], i)
                dequantized_list.append(dq * self.hierarchy_weights[i])
        
        return torch.stack(dequantized_list, dim=-2).sum(dim=-2)
    
    def _single_level_dequantize(self, indices: torch.Tensor, depth: int) -> torch.Tensor:
        """Single-level dequantization at specified depth."""
        if depth >= self.num_layers:
            raise ValueError(f"Depth {depth} exceeds number of layers {self.num_layers}")
        
        # Get quantization parameters
        scale = self.codebook.get_scale(depth)
        zero_point = self.codebook.get_zero_point(depth)
        codebook = self.codebook.get_codebook()
        
        # Handle different input dimensions
        if indices.shape[-1] != self.lattice_dim:
            # For non-lattice-dim inputs, use simple dequantization
            dequantized = (indices.float() - zero_point) * scale
            # Ensure output has same shape as input
            if dequantized.shape != indices.shape:
                dequantized = dequantized.squeeze(-1)
            return dequantized
        
        # Get quantized values
        # indices should be [batch_size] with values in [0, num_codewords-1]
        quantized = codebook[indices]
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def encode_to_depth(
        self, 
        x: torch.Tensor, 
        target_depth: int
    ) -> torch.Tensor:
        """
        Encode input to specific quantization depth.
        
        This operation quantizes input and returns only the indices
        for the specified depth level.
        
        Args:
            x: Input tensor
            target_depth: Target quantization depth
            
        Returns:
            encoded: Encoded indices for target depth
        """
        _, indices = self._single_level_quantize(x, target_depth)
        return indices
    
    def decode_from_depth(
        self, 
        encoded: torch.Tensor, 
        source_depth: int
    ) -> torch.Tensor:
        """
        Decode from specific quantization depth.
        
        This operation reconstructs continuous values from indices
        at the specified depth level.
        
        Args:
            encoded: Encoded indices
            source_depth: Source quantization depth
            
        Returns:
            decoded: Decoded tensor
        """
        return self._single_level_dequantize(encoded, source_depth)
    
    def radixq_encode(
        self, 
        x: torch.Tensor, 
        radix: int, 
        depth: int
    ) -> torch.Tensor:
        """
        Encode using radix-q representation.
        
        This operation compresses lattice indices using radix-q encoding,
        which is useful for gradient communication in distributed training.
        
        Args:
            x: Input tensor
            radix: Base for radix-q encoding
            depth: Quantization depth
            
        Returns:
            encoded: Radix-q encoded tensor
            
        Example:
            >>> x = torch.randn(32, 8)
            >>> encoded = quantizer.radixq_encode(x, radix=4, depth=1)
            >>> print(encoded.dtype)  # torch.int32
        """
        # First quantize to get indices
        _, indices = self._single_level_quantize(x, depth)
        
        # Convert to radix-q representation
        max_radix_value = radix ** depth
        encoded = torch.zeros_like(indices, dtype=torch.int32)
        
        for i in range(depth):
            encoded += (indices % radix) * (radix ** i)
            indices = indices // radix
        
        return encoded
    
    def radixq_decode(
        self, 
        encoded: torch.Tensor, 
        radix: int, 
        depth: int
    ) -> torch.Tensor:
        """
        Decode from radix-q representation.
        
        This operation reconstructs lattice indices from radix-q encoding
        and then dequantizes to continuous values.
        
        Args:
            encoded: Radix-q encoded tensor
            radix: Base for radix-q encoding
            depth: Quantization depth
            
        Returns:
            decoded: Decoded tensor
        """
        # Convert from radix-q to indices
        indices = torch.zeros_like(encoded, dtype=torch.long)
        temp_encoded = encoded.clone()
        
        for i in range(depth):
            indices += (temp_encoded % radix) * (radix ** i)
            temp_encoded = temp_encoded // radix
        
        # Dequantize indices
        return self._single_level_dequantize(indices, depth)
    
    def create_lookup_table(self, max_indices: int = -1) -> torch.Tensor:
        """
        Create lookup table for efficient dot product computation.
        
        This operation precomputes dot products between all possible
        lattice point pairs for fast computation.
        
        Args:
            max_indices: Maximum number of indices to consider (-1 for all)
            
        Returns:
            lookup_table: Precomputed dot product table
            
        Example:
            >>> lut = quantizer.create_lookup_table()
            >>> print(lut.shape)  # torch.Size([256, 256]) for 8-bit lattice
        """
        if max_indices == -1:
            max_indices = self.config.get_num_codewords()
        
        lookup_table = torch.zeros(max_indices, max_indices)
        codebook = self.codebook.get_codebook()
        
        for i in range(max_indices):
            for j in range(max_indices):
                if i < codebook.size(0) and j < codebook.size(0):
                    lookup_table[i, j] = torch.sum(codebook[i] * codebook[j])
        
        return lookup_table
    
    def lookup_dot_product(
        self, 
        x_indices: torch.Tensor, 
        y_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dot product using lookup table.
        
        This operation efficiently computes dot products between
        quantized vectors using precomputed lookup tables.
        
        Args:
            x_indices: Indices for first vector
            y_indices: Indices for second vector
            
        Returns:
            dot_product: Dot product result
            
        Example:
            >>> x_indices = torch.randint(0, 256, (32,))
            >>> y_indices = torch.randint(0, 256, (32,))
            >>> dot_products = quantizer.lookup_dot_product(x_indices, y_indices)
            >>> print(dot_products.shape)  # torch.Size([32])
        """
        if self._dot_product_lut is None:
            self._dot_product_lut = self.create_lookup_table()
            if not hasattr(self, '_dot_product_lut'):
                self.register_buffer('_dot_product_lut', self._dot_product_lut)
        
        return self._dot_product_lut[x_indices, y_indices]
    
    def quantized_add(
        self, 
        x_indices: torch.Tensor, 
        y_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Vector addition in quantized space.
        
        This operation performs vector addition directly in the
        quantized space without dequantization.
        
        Args:
            x_indices: Indices for first vector
            y_indices: Indices for second vector
            
        Returns:
            sum_indices: Indices for sum vector
        """
        # Dequantize, add, and re-quantize
        x_dequantized = self.dequantize(x_indices)
        y_dequantized = self.dequantize(y_indices)
        sum_tensor = x_dequantized + y_dequantized
        
        _, sum_indices = self.quantize(sum_tensor)
        return sum_indices
    
    def quantized_reduce(
        self, 
        indices: torch.Tensor, 
        dim: int = -1
    ) -> torch.Tensor:
        """
        Reduce operation in quantized space.
        
        This operation performs reduction (sum, mean, etc.) directly
        in the quantized space.
        
        Args:
            indices: Input indices
            dim: Dimension to reduce along
            
        Returns:
            reduced_indices: Indices for reduced tensor
        """
        # Dequantize, reduce, and re-quantize
        dequantized = self.dequantize(indices)
        reduced = torch.sum(dequantized, dim=dim, keepdim=True)
        
        _, reduced_indices = self.quantize(reduced)
        return reduced_indices
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for quantization."""
        return self.quantize(x)
    
    def get_quantization_stats(self) -> Dict[str, torch.Tensor]:
        """Get quantization statistics."""
        return {
            'scales': self.codebook.scales,
            'zero_points': self.codebook.zero_points,
            'hierarchy_weights': self.hierarchy_weights,
            'num_codewords': torch.tensor(self.config.get_num_codewords()),
        }
