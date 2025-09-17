"""
Quantized linear layer for quantization-aware training.

This module provides a PyTorch linear layer that integrates hierarchical
nested-lattice quantization for both weights and activations.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..quant import QuantizationConfig, quantize
from ..lattices import Lattice, D4Lattice


class QLinear(nn.Module):
    """
    Quantized linear layer with hierarchical nested-lattice quantization.
    
    This layer maintains FP32 shadow weights and can quantize weights,
    activations, or both during training. It supports straight-through
    estimation for gradient flow.
    
    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        config: Quantization configuration
        lattice: Lattice instance for quantization
        quantize_weights: Whether to quantize weights
        quantize_activations: Whether to quantize activations
        quantize_every: Frequency of quantization (1 = every step)
        use_lut: Whether to use lookup table for matrix multiplication
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantizationConfig,
        lattice: Optional[Lattice] = None,
        quantize_weights: bool = True,
        quantize_activations: bool = False,
        quantize_every: int = 1,
        use_lut: bool = False,
        bias: bool = True
    ):
        """
        Initialize the quantized linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            config: Quantization configuration
            lattice: Lattice instance (defaults to D4Lattice)
            quantize_weights: Whether to quantize weights
            quantize_activations: Whether to quantize activations
            quantize_every: Frequency of quantization (1 = every step)
            use_lut: Whether to use lookup table for matrix multiplication
            bias: Whether to include bias term
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.lattice = lattice if lattice is not None else D4Lattice()
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_every = quantize_every
        self.use_lut = use_lut
        
        # Initialize FP32 shadow weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Quantization step counter
        self._step_count = 0
        
        # Validate dimensions
        if in_features % self.lattice.d != 0:
            raise ValueError(
                f"Input features {in_features} must be divisible by lattice dimension {self.lattice.d}"
            )
        if out_features % self.lattice.d != 0:
            raise ValueError(
                f"Output features {out_features} must be divisible by lattice dimension {self.lattice.d}"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Check if we should quantize this step
        should_quantize = (
            self.training and 
            self.quantize_every > 0 and 
            self._step_count % self.quantize_every == 0
        )
        
        if should_quantize:
            return self._quantized_forward(x)
        else:
            return self._standard_forward(x)
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without quantization."""
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Quantize weights if enabled
        if self.quantize_weights:
            weight_quantized = self._quantize_weight()
        else:
            weight_quantized = self.weight
        
        # Ensure weight_quantized is on the same device as x
        weight_quantized = weight_quantized.to(x.device)
        
        # Quantize activations if enabled
        if self.quantize_activations:
            x_quantized = self._quantize_activation(x)
        else:
            x_quantized = x
        
        # Get bias on the same device
        bias = self.bias.to(x.device) if self.bias is not None else None
        
        # Perform matrix multiplication
        if self.use_lut:
            output = self._lut_matmul(x_quantized, weight_quantized)
        else:
            output = torch.nn.functional.linear(x_quantized, weight_quantized, bias)
        
        # Increment step counter
        self._step_count += 1
        
        return output
    
    def _quantize_weight(self) -> torch.Tensor:
        """Quantize the weight matrix."""
        # Reshape weight to [out_features * in_features // d, d]
        weight_flat = self.weight.view(-1, self.lattice.d)
        
        # Quantize each d-dimensional vector
        weight_quantized = torch.zeros_like(weight_flat)
        for i in range(weight_flat.shape[0]):
            weight_quantized[i] = quantize(weight_flat[i], self.lattice, self.config)
        
        return weight_quantized.view(self.out_features, self.in_features)
    
    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the activation tensor."""
        batch_size = x.shape[0]
        x_flat = x.view(-1, self.lattice.d)
        
        # Quantize each d-dimensional vector
        x_quantized = torch.zeros_like(x_flat)
        for i in range(x_flat.shape[0]):
            x_quantized[i] = quantize(x_flat[i], self.lattice, self.config)
        
        return x_quantized.view(batch_size, self.in_features)
    
    def _lut_matmul(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication using lookup tables.
        
        This is a placeholder implementation. In practice, this would use
        precomputed lookup tables for efficient inner product computation.
        """
        # For now, fall back to standard matrix multiplication
        # TODO: Implement actual LUT-based matrix multiplication
        return torch.nn.functional.linear(x, weight, self.bias)
    
    def get_quantization_stats(self) -> dict:
        """Get statistics about quantization."""
        return {
            "step_count": self._step_count,
            "quantize_weights": self.quantize_weights,
            "quantize_activations": self.quantize_activations,
            "quantize_every": self.quantize_every,
            "use_lut": self.use_lut,
            "lattice_type": self.lattice.name,
            "config": self.config.to_dict(),
        }
    
    def reset_step_count(self):
        """Reset the quantization step counter."""
        self._step_count = 0
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, lattice={self.lattice.name}, "
            f"quantize_weights={self.quantize_weights}, "
            f"quantize_activations={self.quantize_activations}, "
            f"quantize_every={self.quantize_every}"
        )
