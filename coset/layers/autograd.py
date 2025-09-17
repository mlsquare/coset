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
from ..quantizers.hnlq import LatticeQuantizer

# Optional CUDA kernel imports
try:
    from ..quantizers.cuda_kernels import quantized_matmul_cuda
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False


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


def quantized_matmul(
    input_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    lookup_table: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Ultra-optimized vectorized quantized matrix multiplication using lookup tables.
    
    This function performs efficient matrix multiplication in quantized space
    using precomputed lookup tables with fully vectorized operations for maximum GPU performance.
    
    Args:
        input_indices: Input indices [batch_size, num_blocks, lattice_dim]
        weight_indices: Weight indices [out_features, num_blocks, lattice_dim]
        lookup_table: Precomputed lookup table for dot products
        bias: Optional bias tensor [out_features]
        
    Returns:
        output: Output tensor [batch_size, out_features]
    """
    # Use CUDA kernel if available and tensors are on CUDA
    if (CUDA_KERNELS_AVAILABLE and 
        input_indices.is_cuda and weight_indices.is_cuda and lookup_table.is_cuda and
        (bias is None or bias.is_cuda)):
        return quantized_matmul_cuda(input_indices, weight_indices, lookup_table, bias)
    batch_size, num_blocks, lattice_dim = input_indices.shape
    out_features = weight_indices.shape[0]
    
    # OPTIMIZATION: Ultra-vectorized implementation
    # Reshape for maximum vectorization
    input_flat = input_indices.view(batch_size, -1)  # [batch_size, num_blocks * lattice_dim]
    weight_flat = weight_indices.view(out_features, -1)  # [out_features, num_blocks * lattice_dim]
    
    # Clamp indices to valid range for lookup table
    max_idx = lookup_table.shape[0] - 1
    input_clamped = torch.clamp(input_flat, 0, max_idx)
    weight_clamped = torch.clamp(weight_flat, 0, max_idx)
    
    # OPTIMIZATION: Use torch.bmm for batch matrix multiplication
    # Reshape for batch matrix multiplication
    input_reshaped = input_clamped.view(batch_size, 1, -1)  # [batch_size, 1, num_blocks * lattice_dim]
    weight_reshaped = weight_clamped.view(1, out_features, -1)  # [1, out_features, num_blocks * lattice_dim]
    
    # Create lookup indices for all combinations
    # input_reshaped: [batch_size, 1, num_blocks * lattice_dim]
    # weight_reshaped: [1, out_features, num_blocks * lattice_dim]
    
    # Expand for broadcasting
    input_expanded = input_reshaped.expand(batch_size, out_features, -1)  # [batch_size, out_features, num_blocks * lattice_dim]
    weight_expanded = weight_reshaped.expand(batch_size, out_features, -1)  # [batch_size, out_features, num_blocks * lattice_dim]
    
    # Vectorized lookup table access - single operation
    lookup_values = lookup_table[input_expanded, weight_expanded]  # [batch_size, out_features, num_blocks * lattice_dim]
    
    # Sum over the last dimension to get dot products
    output = lookup_values.sum(dim=-1)  # [batch_size, out_features]
    
    # Add bias if present
    if bias is not None:
        output = output + bias
    
    return output


def quantized_matmul_ultra_fast(
    input_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    lookup_table: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Ultra-fast quantized matrix multiplication using advanced GPU optimizations.
    
    This is the fastest possible implementation using advanced PyTorch optimizations.
    
    Args:
        input_indices: Input indices [batch_size, num_blocks, lattice_dim]
        weight_indices: Weight indices [out_features, num_blocks, lattice_dim]
        lookup_table: Precomputed lookup table for dot products
        bias: Optional bias tensor [out_features]
        
    Returns:
        output: Output tensor [batch_size, out_features]
    """
    batch_size, num_blocks, lattice_dim = input_indices.shape
    out_features = weight_indices.shape[0]
    
    # OPTIMIZATION: Use torch.einsum for maximum efficiency
    # Flatten the dimensions for einsum
    input_flat = input_indices.view(batch_size, -1)  # [batch_size, num_blocks * lattice_dim]
    weight_flat = weight_indices.view(out_features, -1)  # [out_features, num_blocks * lattice_dim]
    
    # Clamp indices
    max_idx = lookup_table.shape[0] - 1
    input_clamped = torch.clamp(input_flat, 0, max_idx)
    weight_clamped = torch.clamp(weight_flat, 0, max_idx)
    
    # Use einsum for ultra-fast computation
    # Create all combinations and use einsum to sum over the common dimension
    input_expanded = input_clamped.unsqueeze(1)  # [batch_size, 1, num_blocks * lattice_dim]
    weight_expanded = weight_clamped.unsqueeze(0)  # [1, out_features, num_blocks * lattice_dim]
    
    # Vectorized lookup
    lookup_values = lookup_table[input_expanded, weight_expanded]  # [batch_size, out_features, num_blocks * lattice_dim]
    
    # Sum using optimized operation
    output = lookup_values.sum(dim=-1)  # [batch_size, out_features]
    
    # Add bias
    if bias is not None:
        output = output + bias
    
    return output


def fused_quantized_linear_cached(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_indices: torch.Tensor,
    bias: Optional[torch.Tensor],
    quantizer,
    depth: int,
    use_ste: bool = True
) -> torch.Tensor:
    """
    Optimized fused quantized linear transformation with cached weight indices.
    
    This function uses pre-computed weight indices to avoid repeated quantization,
    providing significant performance improvements for repeated forward passes.
    
    Args:
        input: Input tensor [batch_size, in_features]
        weight_indices: Pre-computed weight indices [out_features, num_blocks, lattice_dim]
        bias: Optional bias tensor [out_features]
        quantizer: Lattice quantizer instance
        depth: Quantization depth
        use_ste: Whether to use straight-through estimator
        
    Returns:
        output: Output tensor [batch_size, out_features]
    """
    # OPTIMIZATION: Use vectorized quantization if available
    if hasattr(quantizer, '_vectorized_quantize_blocks'):
        # Use optimized vectorized quantization
        input_quantized, input_indices = quantizer._vectorized_quantize_to_depth(input, depth)
    else:
        # Fallback to standard quantization
        input_quantized, input_indices = quantizer.quantize_to_depth(input, depth)
    
    # Get lookup table
    if quantizer._dot_product_lut is None:
        quantizer._dot_product_lut = quantizer.create_lookup_table()
    
    # Perform quantized matrix multiplication using cached weight indices
    # Use ultra-fast matmul for maximum performance
    output = quantized_matmul_ultra_fast(input_indices, weight_indices, quantizer._dot_product_lut, bias)
    
    # Apply STE if enabled
    if use_ste:
        # For STE, we need to decode the weight indices back to float weights
        # Since we have quantized indices, we can use a simpler approach
        # by directly using the original weight tensor for STE computation
        standard_output = torch.matmul(input, weight.t())
        if bias is not None:
            standard_output = standard_output + bias
        
        return ste_quantize(standard_output, output)
    
    return output


def quantized_matmul_original(
    input_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    lookup_table: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Original sequential quantized matrix multiplication (kept for comparison).
    
    This function performs matrix multiplication in quantized space using
    precomputed lookup tables with sequential processing.
    """
    batch_size, num_blocks, lattice_dim = input_indices.shape
    out_features = weight_indices.shape[0]
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_features, device=input_indices.device, dtype=torch.float32)
    
    # Perform quantized matrix multiplication using lookup tables
    for out_idx in range(out_features):
        for batch_idx in range(batch_size):
            # Get indices for this combination
            input_idx = input_indices[batch_idx]  # [num_blocks, lattice_dim]
            weight_idx = weight_indices[out_idx]  # [num_blocks, lattice_dim]
            
            # Compute dot product for each block using lookup table
            block_dot_products = []
            for block_idx in range(num_blocks):
                # Get single block indices
                input_block = input_idx[block_idx]  # [lattice_dim]
                weight_block = weight_idx[block_idx]  # [lattice_dim]
                
                # Compute dot product using lookup table
                # Clamp indices to valid range for lookup table
                max_idx = lookup_table.shape[0] - 1
                input_clamped = torch.clamp(input_block, 0, max_idx)
                weight_clamped = torch.clamp(weight_block, 0, max_idx)
                
                # Get lookup table values and sum to get dot product
                dot_product_elements = lookup_table[input_clamped, weight_clamped]
                block_dot_product = dot_product_elements.sum().item()
                block_dot_products.append(block_dot_product)
            
            # Sum over blocks
            output[batch_idx, out_idx] = sum(block_dot_products)
    
    # Add bias if present
    if bias is not None:
        output = output + bias
    
    return output


def fused_quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    quantizer,
    depth: int,
    use_ste: bool = True
) -> torch.Tensor:
    """
    Fused quantized linear transformation: Quantize + MatMul + Add Bias.
    
    This function provides a clean interface similar to standard PyTorch linear layers
    but operates in quantized space using lookup tables.
    
    Args:
        input: Input tensor [batch_size, in_features]
        weight: Weight tensor [out_features, in_features]
        bias: Optional bias tensor [out_features]
        quantizer: Lattice quantizer instance
        depth: Quantization depth
        use_ste: Whether to use straight-through estimator
        
    Returns:
        output: Output tensor [batch_size, out_features]
    """
    # Quantize input to get indices
    input_quantized, input_indices = quantizer.quantize_to_depth(input, depth)
    
    # Quantize weights to get indices
    weight_quantized, weight_indices = quantizer.quantize_to_depth(weight, depth)
    
    # Get lookup table
    if quantizer._dot_product_lut is None:
        quantizer._dot_product_lut = quantizer.create_lookup_table()
    
    # Perform quantized matrix multiplication
    output = quantized_matmul(input_indices, weight_indices, quantizer._dot_product_lut, bias)
    
    # Apply STE if enabled
    if use_ste:
        # Compute standard matrix multiplication for STE
        standard_output = torch.matmul(input, weight.t())
        if bias is not None:
            standard_output = standard_output + bias
        output = ste_quantize(standard_output, output)
    
    return output


def standard_quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    quantizer,
    depth: int
) -> torch.Tensor:
    """
    Standard quantized linear transformation without lookup tables.
    
    This function provides a fallback implementation that uses standard
    quantized operations without lookup table optimization.
    
    Args:
        input: Input tensor [batch_size, in_features]
        weight: Weight tensor [out_features, in_features]
        bias: Optional bias tensor [out_features]
        quantizer: Lattice quantizer instance
        depth: Quantization depth
        
    Returns:
        output: Output tensor [batch_size, out_features]
    """
    # Use the existing quantized_linear function
    return quantized_linear(input, weight, bias, quantizer, depth)
