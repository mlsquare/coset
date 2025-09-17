"""
Sparse and compressed lookup table implementations for memory efficiency.

This module provides optimized lookup table implementations that use sparse
representations and compression techniques to reduce memory usage while
maintaining performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
import numpy as np

class SparseLookupTable(nn.Module):
    """
    Sparse lookup table implementation for memory efficiency.
    
    This class implements a sparse lookup table that only stores non-zero
    entries, significantly reducing memory usage for sparse patterns.
    """
    
    def __init__(self, max_indices: int, sparsity_threshold: float = 0.1):
        """
        Initialize sparse lookup table.
        
        Args:
            max_indices: Maximum number of indices
            sparsity_threshold: Threshold for considering entries as sparse
        """
        super().__init__()
        self.max_indices = max_indices
        self.sparsity_threshold = sparsity_threshold
        
        # Sparse representation using indices and values
        self.register_buffer('_sparse_indices', None)
        self.register_buffer('_sparse_values', None)
        self.register_buffer('_dense_fallback', None)
        
        # Statistics
        self._is_sparse = False
        self._compression_ratio = 1.0
    
    def create_sparse_lookup_table(self, generator_matrix: torch.Tensor, 
                                 inverse_generator_matrix: torch.Tensor,
                                 beta: float, q: int) -> None:
        """
        Create sparse lookup table from generator matrices.
        
        Args:
            generator_matrix: Generator matrix [lattice_dim, lattice_dim]
            inverse_generator_matrix: Inverse generator matrix [lattice_dim, lattice_dim]
            beta: Beta scaling factor
            q: Quantization parameter
        """
        device = generator_matrix.device
        lattice_dim = generator_matrix.shape[0]
        
        # Create full lookup table first
        full_lut = torch.zeros(self.max_indices, self.max_indices, device=device)
        
        # Fill lookup table with actual dot products
        for i in range(self.max_indices):
            for j in range(self.max_indices):
                # Create lattice points from indices
                point_i = self._index_to_lattice_point(i, generator_matrix, beta, q)
                point_j = self._index_to_lattice_point(j, generator_matrix, beta, q)
                
                # Compute dot product
                dot_product = torch.dot(point_i, point_j)
                full_lut[i, j] = dot_product
        
        # Analyze sparsity
        non_zero_mask = torch.abs(full_lut) > 1e-8
        sparsity = 1.0 - (non_zero_mask.sum().float() / full_lut.numel())
        
        if sparsity > self.sparsity_threshold:
            # Use sparse representation
            self._is_sparse = True
            sparse_indices = torch.nonzero(non_zero_mask, as_tuple=False)
            sparse_values = full_lut[non_zero_mask]
            
            self._sparse_indices = sparse_indices
            self._sparse_values = sparse_values
            
            # Calculate compression ratio
            original_size = full_lut.numel()
            sparse_size = sparse_indices.numel() + sparse_values.numel()
            self._compression_ratio = original_size / sparse_size
            
            print(f"Sparse lookup table created: {sparsity:.2%} sparsity, "
                  f"{self._compression_ratio:.1f}x compression")
        else:
            # Use dense representation
            self._is_sparse = False
            self._dense_fallback = full_lut
            self._compression_ratio = 1.0
            
            print(f"Dense lookup table used: {sparsity:.2%} sparsity")
    
    def _index_to_lattice_point(self, index: int, generator_matrix: torch.Tensor, 
                               beta: float, q: int) -> torch.Tensor:
        """Convert index to lattice point."""
        # Convert index to encoding vector
        encoding_vector = torch.zeros(generator_matrix.shape[0], device=generator_matrix.device)
        
        # Simple index to encoding conversion (can be optimized)
        for i in range(generator_matrix.shape[0]):
            encoding_vector[i] = (index // (q ** i)) % q
        
        # Convert to lattice point
        lattice_point = torch.matmul(encoding_vector, generator_matrix) * beta
        return lattice_point
    
    def lookup(self, input_indices: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """
        Perform sparse lookup operation.
        
        Args:
            input_indices: Input indices [batch_size, num_blocks, lattice_dim]
            weight_indices: Weight indices [out_features, num_blocks, lattice_dim]
            
        Returns:
            output: Output tensor [batch_size, out_features]
        """
        if self._is_sparse:
            return self._sparse_lookup(input_indices, weight_indices)
        else:
            return self._dense_lookup(input_indices, weight_indices)
    
    def _sparse_lookup(self, input_indices: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """Perform sparse lookup operation."""
        batch_size, num_blocks, lattice_dim = input_indices.shape
        out_features = weight_indices.shape[0]
        
        # Flatten indices
        input_flat = input_indices.view(batch_size, -1)
        weight_flat = weight_indices.view(out_features, -1)
        
        # Clamp indices to valid range
        input_clamped = torch.clamp(input_flat, 0, self.max_indices - 1)
        weight_clamped = torch.clamp(weight_flat, 0, self.max_indices - 1)
        
        # Create lookup indices for all combinations
        input_expanded = input_clamped.unsqueeze(1)  # [batch_size, 1, num_blocks * lattice_dim]
        weight_expanded = weight_clamped.unsqueeze(0)  # [1, out_features, num_blocks * lattice_dim]
        
        # Broadcast for all combinations
        input_broadcast = input_expanded.expand(batch_size, out_features, -1)
        weight_broadcast = weight_expanded.expand(batch_size, out_features, -1)
        
        # Flatten for sparse lookup
        input_flat_lookup = input_broadcast.view(-1)
        weight_flat_lookup = weight_broadcast.view(-1)
        
        # Create lookup pairs
        lookup_pairs = torch.stack([input_flat_lookup, weight_flat_lookup], dim=1)
        
        # Find matching sparse indices
        output = torch.zeros(batch_size, out_features, device=input_indices.device)
        
        # Use vectorized operations for sparse lookup
        for i, (idx_i, idx_j) in enumerate(self._sparse_indices):
            # Find matching pairs
            mask = (lookup_pairs[:, 0] == idx_i) & (lookup_pairs[:, 1] == idx_j)
            
            if mask.any():
                # Get the value
                value = self._sparse_values[i]
                
                # Add to output
                output.view(-1)[mask] += value
        
        return output
    
    def _dense_lookup(self, input_indices: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """Perform dense lookup operation."""
        batch_size, num_blocks, lattice_dim = input_indices.shape
        out_features = weight_indices.shape[0]
        
        # Flatten indices
        input_flat = input_indices.view(batch_size, -1)
        weight_flat = weight_indices.view(out_features, -1)
        
        # Clamp indices to valid range
        input_clamped = torch.clamp(input_flat, 0, self.max_indices - 1)
        weight_clamped = torch.clamp(weight_flat, 0, self.max_indices - 1)
        
        # Create lookup indices for all combinations
        input_expanded = input_clamped.unsqueeze(1)  # [batch_size, 1, num_blocks * lattice_dim]
        weight_expanded = weight_clamped.unsqueeze(0)  # [1, out_features, num_blocks * lattice_dim]
        
        # Broadcast for all combinations
        input_broadcast = input_expanded.expand(batch_size, out_features, -1)
        weight_broadcast = weight_expanded.expand(batch_size, out_features, -1)
        
        # Vectorized lookup
        lookup_values = self._dense_fallback[input_broadcast, weight_broadcast]
        
        # Sum over the last dimension
        output = torch.sum(lookup_values, dim=-1)
        
        return output
    
    def get_compression_stats(self) -> Dict[str, Union[float, bool]]:
        """Get compression statistics."""
        return {
            'is_sparse': self._is_sparse,
            'compression_ratio': self._compression_ratio,
            'sparsity_threshold': self.sparsity_threshold,
            'max_indices': self.max_indices
        }

class CompressedLookupTable(nn.Module):
    """
    Compressed lookup table using quantization and compression techniques.
    
    This class implements a compressed lookup table that uses quantization
    and other compression techniques to reduce memory usage.
    """
    
    def __init__(self, max_indices: int, compression_bits: int = 8):
        """
        Initialize compressed lookup table.
        
        Args:
            max_indices: Maximum number of indices
            compression_bits: Number of bits for compression (8, 16, or 32)
        """
        super().__init__()
        self.max_indices = max_indices
        self.compression_bits = compression_bits
        
        # Compressed representation
        self.register_buffer('_compressed_table', None)
        self.register_buffer('_scale_factor', None)
        self.register_buffer('_zero_point', None)
        
        # Statistics
        self._compression_ratio = 1.0
    
    def create_compressed_lookup_table(self, generator_matrix: torch.Tensor,
                                     inverse_generator_matrix: torch.Tensor,
                                     beta: float, q: int) -> None:
        """
        Create compressed lookup table from generator matrices.
        
        Args:
            generator_matrix: Generator matrix [lattice_dim, lattice_dim]
            inverse_generator_matrix: Inverse generator matrix [lattice_dim, lattice_dim]
            beta: Beta scaling factor
            q: Quantization parameter
        """
        device = generator_matrix.device
        lattice_dim = generator_matrix.shape[0]
        
        # Create full lookup table first
        full_lut = torch.zeros(self.max_indices, self.max_indices, device=device)
        
        # Fill lookup table with actual dot products
        for i in range(self.max_indices):
            for j in range(self.max_indices):
                # Create lattice points from indices
                point_i = self._index_to_lattice_point(i, generator_matrix, beta, q)
                point_j = self._index_to_lattice_point(j, generator_matrix, beta, q)
                
                # Compute dot product
                dot_product = torch.dot(point_i, point_j)
                full_lut[i, j] = dot_product
        
        # Compress the lookup table
        self._compress_table(full_lut)
    
    def _index_to_lattice_point(self, index: int, generator_matrix: torch.Tensor,
                               beta: float, q: int) -> torch.Tensor:
        """Convert index to lattice point."""
        # Convert index to encoding vector
        encoding_vector = torch.zeros(generator_matrix.shape[0], device=generator_matrix.device)
        
        # Simple index to encoding conversion (can be optimized)
        for i in range(generator_matrix.shape[0]):
            encoding_vector[i] = (index // (q ** i)) % q
        
        # Convert to lattice point
        lattice_point = torch.matmul(encoding_vector, generator_matrix) * beta
        return lattice_point
    
    def _compress_table(self, full_lut: torch.Tensor) -> None:
        """Compress the lookup table using quantization."""
        # Compute scale factor and zero point
        min_val = full_lut.min()
        max_val = full_lut.max()
        
        if self.compression_bits == 8:
            qmin, qmax = 0, 255
        elif self.compression_bits == 16:
            qmin, qmax = 0, 65535
        else:  # 32 bits
            qmin, qmax = 0, 4294967295
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize the table
        quantized = torch.round(full_lut / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Store compressed representation
        if self.compression_bits == 8:
            self._compressed_table = quantized.to(torch.uint8)
        elif self.compression_bits == 16:
            self._compressed_table = quantized.to(torch.int16)
        else:
            self._compressed_table = quantized.to(torch.int32)
        
        self._scale_factor = scale
        self._zero_point = zero_point
        
        # Calculate compression ratio
        original_size = full_lut.numel() * 4  # 4 bytes per float32
        compressed_size = self._compressed_table.numel() * (self.compression_bits // 8)
        self._compression_ratio = original_size / compressed_size
        
        print(f"Compressed lookup table created: {self.compression_bits}-bit compression, "
              f"{self._compression_ratio:.1f}x compression")
    
    def lookup(self, input_indices: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """
        Perform compressed lookup operation.
        
        Args:
            input_indices: Input indices [batch_size, num_blocks, lattice_dim]
            weight_indices: Weight indices [out_features, num_blocks, lattice_dim]
            
        Returns:
            output: Output tensor [batch_size, out_features]
        """
        batch_size, num_blocks, lattice_dim = input_indices.shape
        out_features = weight_indices.shape[0]
        
        # Flatten indices
        input_flat = input_indices.view(batch_size, -1)
        weight_flat = weight_indices.view(out_features, -1)
        
        # Clamp indices to valid range
        input_clamped = torch.clamp(input_flat, 0, self.max_indices - 1)
        weight_clamped = torch.clamp(weight_flat, 0, self.max_indices - 1)
        
        # Create lookup indices for all combinations
        input_expanded = input_clamped.unsqueeze(1)  # [batch_size, 1, num_blocks * lattice_dim]
        weight_expanded = weight_clamped.unsqueeze(0)  # [1, out_features, num_blocks * lattice_dim]
        
        # Broadcast for all combinations
        input_broadcast = input_expanded.expand(batch_size, out_features, -1)
        weight_broadcast = weight_expanded.expand(batch_size, out_features, -1)
        
        # Vectorized lookup with decompression
        compressed_values = self._compressed_table[input_broadcast, weight_broadcast]
        lookup_values = (compressed_values.float() - self._zero_point) * self._scale_factor
        
        # Sum over the last dimension
        output = torch.sum(lookup_values, dim=-1)
        
        return output
    
    def get_compression_stats(self) -> Dict[str, Union[float, int]]:
        """Get compression statistics."""
        return {
            'compression_ratio': self._compression_ratio,
            'compression_bits': self.compression_bits,
            'max_indices': self.max_indices
        }

def create_optimized_lookup_table(max_indices: int, generator_matrix: torch.Tensor,
                                inverse_generator_matrix: torch.Tensor, beta: float, q: int,
                                optimization_type: str = "auto") -> Union[SparseLookupTable, CompressedLookupTable]:
    """
    Create an optimized lookup table based on the specified optimization type.
    
    Args:
        max_indices: Maximum number of indices
        generator_matrix: Generator matrix
        inverse_generator_matrix: Inverse generator matrix
        beta: Beta scaling factor
        q: Quantization parameter
        optimization_type: Type of optimization ("sparse", "compressed", or "auto")
        
    Returns:
        Optimized lookup table
    """
    if optimization_type == "sparse":
        lut = SparseLookupTable(max_indices)
        lut.create_sparse_lookup_table(generator_matrix, inverse_generator_matrix, beta, q)
        return lut
    elif optimization_type == "compressed":
        lut = CompressedLookupTable(max_indices)
        lut.create_compressed_lookup_table(generator_matrix, inverse_generator_matrix, beta, q)
        return lut
    elif optimization_type == "auto":
        # Try sparse first, fall back to compressed if not sparse enough
        sparse_lut = SparseLookupTable(max_indices)
        sparse_lut.create_sparse_lookup_table(generator_matrix, inverse_generator_matrix, beta, q)
        
        if sparse_lut._is_sparse:
            return sparse_lut
        else:
            compressed_lut = CompressedLookupTable(max_indices)
            compressed_lut.create_compressed_lookup_table(generator_matrix, inverse_generator_matrix, beta, q)
            return compressed_lut
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")

# Export the classes and functions
__all__ = [
    'SparseLookupTable',
    'CompressedLookupTable', 
    'create_optimized_lookup_table'
]
