"""
Autograd functions for quantized operations with Straight-Through Estimators (STE)

This module implements custom autograd functions that enable gradient flow
through quantized operations using straight-through estimators.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional

from ..quantizers.config import LatticeConfig


class STEFunction(Function):
    """
    Straight-Through Estimator (STE) function.
    
    This function allows gradients to flow through quantization operations
    by using the identity function for the backward pass.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: return quantized values.
        
        Args:
            input: Original input tensor
            quantized: Quantized tensor
            
        Returns:
            quantized: Quantized tensor (same as input for forward pass)
        """
        ctx.save_for_backward(input, quantized)
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: use identity function for gradients.
        
        This is the key of STE - gradients flow through as if no
        quantization happened.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            grad_input: Gradient for input (identity)
            grad_quantized: None (no gradient for quantized)
        """
        input, quantized = ctx.saved_tensors
        return grad_output, None


class QuantizedLinearFunction(Function):
    """
    Custom autograd function for quantized linear layer.
    
    This function implements the forward and backward passes for
    quantized matrix multiplication with STE.
    """
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        quantizer,
        config: LatticeConfig,
        depth: int = -1
    ) -> torch.Tensor:
        """
        Forward pass for quantized linear layer.
        
        Args:
            input: Input tensor [batch_size, in_features]
            weight: Weight tensor [out_features, in_features]
            bias: Bias tensor [out_features] or None
            quantizer: Lattice quantizer instance
            config: Lattice configuration
            depth: Quantization depth
            
        Returns:
            output: Output tensor [batch_size, out_features]
        """
        # Quantize input
        input_quantized, input_indices = quantizer.quantize_to_depth(input, depth)
        
        # Quantize weights
        weight_quantized, weight_indices = quantizer.quantize_to_depth(weight, depth)
        
        # Perform quantized matrix multiplication
        # For now, we'll use standard matrix multiplication
        # In the CUDA implementation, this will use lookup tables
        output = torch.matmul(input_quantized, weight_quantized.t())
        
        # Add bias if present
        if bias is not None:
            output = output + bias
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, bias, input_quantized, weight_quantized)
        ctx.quantizer = quantizer
        ctx.config = config
        ctx.depth = depth
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass for quantized linear layer.
        
        Uses STE to allow gradients to flow through quantization.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            grad_input: Gradient for input
            grad_weight: Gradient for weight
            grad_bias: Gradient for bias
            None, None, None: No gradients for quantizer, config, depth
        """
        input, weight, bias, input_quantized, weight_quantized = ctx.saved_tensors
        quantizer = ctx.quantizer
        config = ctx.config
        depth = ctx.depth
        
        # Compute gradients using STE
        # Gradients flow through as if no quantization happened
        
        # Gradient for input
        grad_input = torch.matmul(grad_output, weight_quantized)
        
        # Gradient for weight
        grad_weight = torch.matmul(grad_output.t(), input_quantized)
        
        # Gradient for bias
        grad_bias = None
        if bias is not None:
            grad_bias = torch.sum(grad_output, dim=0)
        
        return grad_input, grad_weight, grad_bias, None, None, None


class QuantizedMatMulFunction(Function):
    """
    Custom autograd function for quantized matrix multiplication.
    
    This function implements efficient matrix multiplication using
    lookup tables in the quantized space.
    """
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        lookup_table: torch.Tensor,
        config: LatticeConfig
    ) -> torch.Tensor:
        """
        Forward pass for quantized matrix multiplication.
        
        Args:
            input: Input tensor [batch_size, in_features]
            weight: Weight tensor [out_features, in_features]
            lookup_table: Precomputed lookup table
            config: Lattice configuration
            
        Returns:
            output: Output tensor [batch_size, out_features]
        """
        # For now, use standard matrix multiplication
        # In CUDA implementation, this will use lookup tables
        output = torch.matmul(input, weight.t())
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, lookup_table)
        ctx.config = config
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass for quantized matrix multiplication.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            grad_input: Gradient for input
            grad_weight: Gradient for weight
            None, None: No gradients for lookup_table, config
        """
        input, weight, lookup_table = ctx.saved_tensors
        
        # Compute gradients
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), input)
        
        return grad_input, grad_weight, None, None


class QuantizedGradientFunction(Function):
    """
    Custom autograd function for quantized gradient operations.
    
    This function handles gradient quantization and communication
    for distributed training.
    """
    
    @staticmethod
    def forward(
        ctx,
        gradients: torch.Tensor,
        compressor,
        depth: int
    ) -> torch.Tensor:
        """
        Forward pass for gradient quantization.
        
        Args:
            gradients: Gradient tensor
            compressor: Gradient compressor
            depth: Quantization depth
            
        Returns:
            quantized_gradients: Quantized gradient tensor
        """
        # Compress gradients
        quantized_gradients = compressor.compress_gradients(gradients, depth)
        
        # Save for backward pass
        ctx.save_for_backward(gradients, quantized_gradients)
        ctx.compressor = compressor
        ctx.depth = depth
        
        return quantized_gradients
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass for gradient quantization.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            grad_gradients: Gradient for original gradients
            None, None: No gradients for compressor, depth
        """
        gradients, quantized_gradients = ctx.saved_tensors
        
        # Use STE: gradients flow through as if no quantization happened
        return grad_output, None, None


# Convenience functions
def ste_quantize(input: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """
    Apply straight-through estimator to quantization.
    
    Args:
        input: Original input tensor
        quantized: Quantized tensor
        
    Returns:
        quantized: Quantized tensor with STE applied
    """
    return STEFunction.apply(input, quantized)


def quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    quantizer,
    config: LatticeConfig,
    depth: int = -1
) -> torch.Tensor:
    """
    Apply quantized linear transformation.
    
    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Bias tensor or None
        quantizer: Lattice quantizer
        config: Lattice configuration
        depth: Quantization depth
        
    Returns:
        output: Output tensor
    """
    return QuantizedLinearFunction.apply(input, weight, bias, quantizer, config, depth)


def quantized_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    lookup_table: torch.Tensor,
    config: LatticeConfig
) -> torch.Tensor:
    """
    Apply quantized matrix multiplication.
    
    Args:
        input: Input tensor
        weight: Weight tensor
        lookup_table: Precomputed lookup table
        config: Lattice configuration
        
    Returns:
        output: Output tensor
    """
    return QuantizedMatMulFunction.apply(input, weight, lookup_table, config)


def quantize_gradients(
    gradients: torch.Tensor,
    compressor,
    depth: int
) -> torch.Tensor:
    """
    Quantize gradients for communication.
    
    Args:
        gradients: Gradient tensor
        compressor: Gradient compressor
        depth: Quantization depth
        
    Returns:
        quantized_gradients: Quantized gradient tensor
    """
    return QuantizedGradientFunction.apply(gradients, compressor, depth)
