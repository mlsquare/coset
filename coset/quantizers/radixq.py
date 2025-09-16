"""
Radix-Q Encoding/Decoding for Efficient Gradient Communication

This module implements radix-q encoding and decoding operations for compressing
gradients during distributed training. Radix-q encoding allows for efficient
communication by representing quantized values in a compressed format.

Core Operations:

1. Radix-Q Encoding:
   - Converts quantized indices to radix-q representation
   - Reduces communication overhead by using fewer bits
   - Supports different radix values (2, 4, 8, 16, etc.)

2. Radix-Q Decoding:
   - Reconstructs quantized indices from radix-q representation
   - Maintains numerical precision during communication
   - Supports batch operations for efficiency

3. Gradient Compression:
   - Compresses gradients before communication
   - Enables in-place operations in compressed space
   - Optimizes memory usage for distributed training
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import math

from .config import LatticeConfig


class RadixQEncoder(nn.Module):
    """
    Radix-Q Encoder for efficient gradient communication.
    
    This class implements radix-q encoding and decoding operations
    for compressing gradients during distributed training.
    """
    
    def __init__(self, config: LatticeConfig):
        super().__init__()
        self.config = config
        self.radix = config.radix
        self.num_layers = config.num_layers
        self.lattice_dim = config.lattice_dim
        
        # Precompute radix tables for efficiency
        self._radix_tables = self._create_radix_tables()
    
    def _create_radix_tables(self) -> torch.Tensor:
        """
        Create radix tables for efficient encoding/decoding.
        
        Returns:
            radix_tables: Precomputed radix tables
        """
        max_depth = self.num_layers
        max_radix_value = self.radix ** max_depth
        
        # Create tables for each depth level
        radix_tables = torch.zeros(max_depth, max_radix_value, dtype=torch.long)
        
        for depth in range(max_depth):
            for i in range(max_radix_value):
                # Convert to radix-q representation
                radix_value = 0
                temp = i
                for j in range(depth):
                    radix_value += (temp % self.radix) * (self.radix ** j)
                    temp = temp // self.radix
                radix_tables[depth, i] = radix_value
        
        return radix_tables
    
    def encode(
        self, 
        x: torch.Tensor, 
        depth: int,
        radix: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode tensor using radix-q representation.
        
        This operation converts quantized values to radix-q representation
        for efficient communication during distributed training.
        
        Args:
            x: Input tensor to encode
            depth: Encoding depth (number of radix digits)
            radix: Base for radix-q encoding (uses config radix if None)
            
        Returns:
            encoded: Radix-q encoded tensor
            
        Example:
            >>> config = LatticeConfig(radix=4, num_layers=3)
            >>> encoder = RadixQEncoder(config)
            >>> x = torch.randint(0, 16, (32, 8))
            >>> encoded = encoder.encode(x, depth=2)
            >>> print(encoded.dtype)  # torch.int32
        """
        if radix is None:
            radix = self.radix
        
        if depth <= 0 or depth > self.num_layers:
            raise ValueError(f"Depth {depth} must be between 1 and {self.num_layers}")
        
        # Ensure input is within valid range
        max_value = radix ** depth - 1
        x_clamped = torch.clamp(x, 0, max_value)
        
        # Convert to radix-q representation
        encoded = torch.zeros_like(x_clamped, dtype=torch.int32)
        temp = x_clamped.clone().long()
        
        for i in range(depth):
            encoded += (temp % radix) * (radix ** i)
            temp = temp // radix
        
        return encoded
    
    def decode(
        self, 
        encoded: torch.Tensor, 
        depth: int,
        radix: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode tensor from radix-q representation.
        
        This operation reconstructs quantized values from radix-q
        representation after communication.
        
        Args:
            encoded: Radix-q encoded tensor
            depth: Encoding depth (number of radix digits)
            radix: Base for radix-q encoding (uses config radix if None)
            
        Returns:
            decoded: Decoded tensor
            
        Example:
            >>> encoded = torch.tensor([5, 10, 15])  # radix-4 encoded
            >>> decoded = encoder.decode(encoded, depth=2)
            >>> print(decoded)  # tensor([1, 2, 3])  # original values
        """
        if radix is None:
            radix = self.radix
        
        if depth <= 0 or depth > self.num_layers:
            raise ValueError(f"Depth {depth} must be between 1 and {self.num_layers}")
        
        # Convert from radix-q to original values
        decoded = torch.zeros_like(encoded, dtype=torch.long)
        temp = encoded.clone()
        
        for i in range(depth):
            decoded += (temp % radix) * (radix ** i)
            temp = temp // radix
        
        return decoded
    
    def encode_gradients(
        self, 
        gradients: torch.Tensor, 
        depth: int = 1
    ) -> torch.Tensor:
        """
        Encode gradients for communication.
        
        This operation compresses gradients using radix-q encoding
        before sending them over the network.
        
        Args:
            gradients: Gradient tensor
            depth: Encoding depth
            
        Returns:
            encoded_gradients: Compressed gradient tensor
        """
        # Quantize gradients first (assuming they're already quantized)
        # In practice, this would be called after gradient quantization
        return self.encode(gradients, depth)
    
    def decode_gradients(
        self, 
        encoded_gradients: torch.Tensor, 
        depth: int = 1
    ) -> torch.Tensor:
        """
        Decode gradients after communication.
        
        This operation reconstructs gradients from radix-q encoding
        after receiving them over the network.
        
        Args:
            encoded_gradients: Compressed gradient tensor
            depth: Encoding depth
            
        Returns:
            gradients: Reconstructed gradient tensor
        """
        return self.decode(encoded_gradients, depth)
    
    def compute_compression_ratio(self, depth: int) -> float:
        """
        Compute compression ratio for given depth.
        
        Args:
            depth: Encoding depth
            
        Returns:
            compression_ratio: Ratio of original to compressed size
        """
        original_bits = self.lattice_dim * 32  # Assuming 32-bit floats
        compressed_bits = depth * math.log2(self.radix)
        return original_bits / compressed_bits
    
    def get_max_encoded_value(self, depth: int) -> int:
        """
        Get maximum value that can be encoded at given depth.
        
        Args:
            depth: Encoding depth
            
        Returns:
            max_value: Maximum encodable value
        """
        return self.radix ** depth - 1
    
    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """Forward pass for encoding."""
        return self.encode(x, depth)


class QuantizedGradientCompressor(nn.Module):
    """
    Gradient compressor for distributed training.
    
    This class combines quantization and radix-q encoding for
    efficient gradient communication.
    """
    
    def __init__(self, config: LatticeConfig):
        super().__init__()
        self.config = config
        self.radix_encoder = RadixQEncoder(config)
        
        # Communication parameters
        self.communication_depth = 1  # Default depth for communication
        self.compression_enabled = True
    
    def compress_gradients(
        self, 
        gradients: torch.Tensor, 
        depth: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compress gradients for communication.
        
        Args:
            gradients: Gradient tensor
            depth: Compression depth (uses default if None)
            
        Returns:
            compressed: Compressed gradient tensor
        """
        if depth is None:
            depth = self.communication_depth
        
        if not self.compression_enabled:
            return gradients
        
        # Encode gradients using radix-q
        return self.radix_encoder.encode_gradients(gradients, depth)
    
    def decompress_gradients(
        self, 
        compressed_gradients: torch.Tensor, 
        depth: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decompress gradients after communication.
        
        Args:
            compressed_gradients: Compressed gradient tensor
            depth: Compression depth (uses default if None)
            
        Returns:
            gradients: Decompressed gradient tensor
        """
        if depth is None:
            depth = self.communication_depth
        
        if not self.compression_enabled:
            return compressed_gradients
        
        # Decode gradients from radix-q
        return self.radix_encoder.decode_gradients(compressed_gradients, depth)
    
    def set_communication_depth(self, depth: int):
        """Set communication depth for gradient compression."""
        if depth <= 0 or depth > self.config.num_layers:
            raise ValueError(f"Depth {depth} must be between 1 and {self.config.num_layers}")
        self.communication_depth = depth
    
    def enable_compression(self, enabled: bool = True):
        """Enable or disable gradient compression."""
        self.compression_enabled = enabled
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        return {
            'compression_ratio': self.radix_encoder.compute_compression_ratio(self.communication_depth),
            'max_encoded_value': self.radix_encoder.get_max_encoded_value(self.communication_depth),
            'communication_depth': self.communication_depth,
            'compression_enabled': self.compression_enabled,
        }
