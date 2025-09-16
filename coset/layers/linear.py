"""
Quantized linear layer implementation

This module implements quantized linear layers that use hierarchical nested
lattice quantization for efficient matrix operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..quantizers.config import LatticeConfig
from ..quantizers.hnlq import LatticeQuantizer
from ..quantizers.radixq import QuantizedGradientCompressor
from .autograd import quantized_linear, ste_quantize


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer using hierarchical nested lattice quantization.
    
    This layer implements efficient matrix multiplication in quantized space
    with support for:
    - Multi-level quantization
    - Lookup table operations
    - Straight-through estimators for gradient flow
    - Gradient compression for distributed training
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LatticeConfig,
        bias: bool = True,
        use_ste: bool = True,
        use_lookup_tables: bool = True
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            config: Lattice quantization configuration
            bias: Whether to use bias term
            use_ste: Whether to use straight-through estimator
            use_lookup_tables: Whether to use lookup tables for computation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.use_ste = use_ste
        self.use_lookup_tables = use_lookup_tables
        
        # Initialize weight parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Initialize bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize quantizer
        self.quantizer = LatticeQuantizer(config)
        
        # Initialize gradient compressor
        self.gradient_compressor = QuantizedGradientCompressor(config)
        
        # Initialize lookup table (lazy initialization)
        self._lookup_table = None
        
        # Initialize quantization parameters
        self._init_quantization_params()
    
    def _init_quantization_params(self):
        """Initialize quantization parameters based on weight statistics."""
        with torch.no_grad():
            # Initialize scales based on weight statistics
            weight_std = torch.std(self.weight)
            weight_mean = torch.mean(self.weight)
            
            # Set initial scale to cover weight range
            initial_scale = weight_std / (2.0 * self.config.radix)
            
            # Update quantizer scales
            for i in range(self.config.num_layers):
                self.quantizer.codebook.scales.data[i] = initial_scale * (2.0 ** i)
    
    def _get_lookup_table(self) -> torch.Tensor:
        """Get or create lookup table for efficient computation."""
        if self._lookup_table is None:
            self._lookup_table = self.quantizer.create_lookup_table()
            if not hasattr(self, '_lookup_table'):
                self.register_buffer('_lookup_table', self._lookup_table)
        return self._lookup_table
    
    def forward(self, input: torch.Tensor, depth: int = -1) -> torch.Tensor:
        """
        Forward pass for quantized linear layer.
        
        Args:
            input: Input tensor of shape [batch_size, in_features]
            depth: Quantization depth (-1 for adaptive, 0-N for specific layer)
            
        Returns:
            output: Output tensor of shape [batch_size, out_features]
        """
        if self.use_lookup_tables:
            return self._forward_with_lookup_tables(input, depth)
        else:
            return self._forward_standard(input, depth)
    
    def _forward_with_lookup_tables(self, input: torch.Tensor, depth: int) -> torch.Tensor:
        """Forward pass using lookup tables for efficient computation."""
        # Quantize input
        input_quantized, input_indices = self.quantizer.quantize(input, depth)
        
        # Quantize weights
        weight_quantized, weight_indices = self.quantizer.quantize(self.weight, depth)
        
        # Get lookup table
        lookup_table = self._get_lookup_table()
        
        # Perform quantized matrix multiplication using lookup tables
        # For now, we'll use standard matrix multiplication
        # In CUDA implementation, this will use the lookup table
        output = torch.matmul(input_quantized, weight_quantized.t())
        
        # Apply STE if enabled
        if self.use_ste:
            # Compute standard matrix multiplication for STE
            standard_output = torch.matmul(input, self.weight.t())
            output = ste_quantize(standard_output, output)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _forward_standard(self, input: torch.Tensor, depth: int) -> torch.Tensor:
        """Standard forward pass without lookup tables."""
        # Use custom autograd function
        output = quantized_linear(
            input, self.weight, self.bias, self.quantizer, self.config, depth
        )
        
        return output
    
    def get_quantized_weights(self, depth: int = -1) -> torch.Tensor:
        """
        Get quantized weight tensor.
        
        Args:
            depth: Quantization depth
            
        Returns:
            quantized_weights: Quantized weight tensor
        """
        quantized, _ = self.quantizer.quantize(self.weight, depth)
        return quantized
    
    def get_quantized_gradients(self, depth: int = -1) -> torch.Tensor:
        """
        Get quantized gradients for communication.
        
        Args:
            depth: Quantization depth
            
        Returns:
            quantized_gradients: Quantized gradient tensor
        """
        if self.weight.grad is None:
            return None
        
        # Compress gradients
        return self.gradient_compressor.compress_gradients(self.weight.grad, depth)
    
    def set_quantized_gradients(self, quantized_gradients: torch.Tensor, depth: int = -1):
        """
        Set quantized gradients after communication.
        
        Args:
            quantized_gradients: Quantized gradient tensor
            depth: Quantization depth
        """
        # Decompress gradients
        gradients = self.gradient_compressor.decompress_gradients(quantized_gradients, depth)
        
        # Set gradients
        if self.weight.grad is None:
            self.weight.grad = gradients
        else:
            self.weight.grad += gradients
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization statistics for this layer."""
        stats = {
            'weight_shape': self.weight.shape,
            'bias_shape': self.bias.shape if self.bias is not None else None,
            'quantization_error': self._compute_quantization_error(),
            'compression_stats': self.gradient_compressor.get_compression_stats(),
        }
        
        # Add quantizer stats
        quantizer_stats = self.quantizer.get_quantization_stats()
        stats.update(quantizer_stats)
        
        return stats
    
    def _compute_quantization_error(self) -> torch.Tensor:
        """Compute quantization error for weights."""
        quantized_weights = self.get_quantized_weights()
        return torch.mean(torch.abs(self.weight - quantized_weights))
    
    def set_communication_depth(self, depth: int):
        """Set communication depth for gradient compression."""
        self.gradient_compressor.set_communication_depth(depth)
    
    def enable_compression(self, enabled: bool = True):
        """Enable or disable gradient compression."""
        self.gradient_compressor.enable_compression(enabled)
    
    def extra_repr(self) -> str:
        """Extra representation for the layer."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, use_ste={self.use_ste}, '
                f'use_lookup_tables={self.use_lookup_tables}')


class QuantizedMLP(nn.Module):
    """
    Multi-layer perceptron using quantized linear layers.
    
    This class provides a convenient way to create MLPs with quantized
    linear layers for efficient computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        config: LatticeConfig,
        activation: str = "relu",
        dropout: float = 0.1,
        use_ste: bool = True,
        use_lookup_tables: bool = True
    ):
        """
        Initialize quantized MLP.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output features
            config: Lattice quantization configuration
            activation: Activation function name
            dropout: Dropout probability
            use_ste: Whether to use straight-through estimator
            use_lookup_tables: Whether to use lookup tables
        """
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(QuantizedLinear(
            input_dim, hidden_dims[0], config, use_ste=use_ste, use_lookup_tables=use_lookup_tables
        ))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(QuantizedLinear(
                hidden_dims[i], hidden_dims[i + 1], config, use_ste=use_ste, use_lookup_tables=use_lookup_tables
            ))
        
        # Output layer
        self.layers.append(QuantizedLinear(
            hidden_dims[-1], output_dim, config, use_ste=use_ste, use_lookup_tables=use_lookup_tables
        ))
        
        # Activation and dropout
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, depth: int = -1) -> torch.Tensor:
        """
        Forward pass for quantized MLP.
        
        Args:
            x: Input tensor
            depth: Quantization depth
            
        Returns:
            output: Output tensor
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, depth)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x, depth)
        return x
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization statistics for all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_quantization_stats()
        return stats
    
    def set_communication_depth(self, depth: int):
        """Set communication depth for all layers."""
        for layer in self.layers:
            layer.set_communication_depth(depth)
    
    def enable_compression(self, enabled: bool = True):
        """Enable or disable gradient compression for all layers."""
        for layer in self.layers:
            layer.enable_compression(enabled)
