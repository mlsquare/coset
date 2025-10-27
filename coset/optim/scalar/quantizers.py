"""
Scalar Quantization Functions

This module provides core scalar quantization functions supporting both
symmetric and asymmetric quantization with block-based grouping and
per-row norm scaling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


def _compute_row_norms(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute L2 norms for each row of the input tensor.
    
    Args:
        x: Input tensor of shape [..., features]
        
    Returns:
        Tuple of (row_norms, row_norms_safe) where row_norms_safe has minimum 1e-8
    """
    # Flatten to 2D: [num_rows, num_cols]
    if x.dim() == 1:
        x_2d = x.unsqueeze(0)  # [1, num_cols]
    else:
        # Flatten all but the last dimension to rows
        x_2d = x.view(-1, x.shape[-1])
    
    # Compute L2 norm of each row
    row_norms = torch.norm(x_2d, p=2, dim=1, keepdim=True)  # [num_rows, 1]
    # Avoid division by zero
    row_norms_safe = torch.clamp(row_norms, min=1e-8)
    
    return row_norms, row_norms_safe


def _reshape_to_blocks(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, Tuple]:
    """
    Reshape tensor to blocks of specified size.
    
    Args:
        x: Input tensor of shape [..., features]
        block_size: Size of each block
        
    Returns:
        Tuple of (reshaped_tensor, block_info)
    """
    original_shape = x.shape
    
    if block_size is None:
        # No blocking - treat each row as a single block
        # Return consistent tuple structure
        if x.dim() == 1:
            x_2d = x.unsqueeze(0)
            was_1d = True
        else:
            x_2d = x.view(-1, x.shape[-1])
            was_1d = False
        
        num_rows, num_cols = x_2d.shape
        return x_2d, (original_shape, was_1d, num_rows, num_cols, 1, num_cols)
    
    # Flatten to 2D: [num_rows, num_cols]
    if x.dim() == 1:
        x_2d = x.unsqueeze(0)  # [1, num_cols]
        was_1d = True
    else:
        x_2d = x.view(-1, x.shape[-1])
        was_1d = False
    
    num_rows, num_cols = x_2d.shape
    
    # Pad to multiple of block_size
    if num_cols % block_size != 0:
        padding_size = block_size - (num_cols % block_size)
        padding = torch.zeros(num_rows, padding_size, device=x.device, dtype=x.dtype)
        x_2d = torch.cat([x_2d, padding], dim=1)
        num_cols = x_2d.shape[1]
    
    # Reshape to blocks
    num_blocks = num_cols // block_size
    x_blocks = x_2d.view(num_rows, num_blocks, block_size)
    
    # Flatten to [num_rows * num_blocks, block_size]
    x_blocks = x_blocks.view(-1, block_size)
    
    return x_blocks, (original_shape, was_1d, num_rows, num_cols, num_blocks, block_size)


def _unreshape_from_blocks(x_blocks: torch.Tensor, original_info: Tuple, original_shape: Tuple) -> torch.Tensor:
    """
    Reshape tensor back from blocks to original shape.
    
    Args:
        x_blocks: Blocked tensor of shape [num_rows * num_blocks, block_size]
        original_info: Information from _reshape_to_blocks
        original_shape: Original shape before blocking
        
    Returns:
        Tensor restored to original shape
    """
    original_shape, was_1d, num_rows, num_cols, num_blocks, block_size = original_info
    
    # Reshape back to [num_rows, num_blocks, block_size]
    x_2d = x_blocks.view(num_rows, num_blocks, block_size)
    
    # Reshape to [num_rows, num_cols]
    x_2d = x_2d.view(num_rows, num_cols)
    
    # Remove padding if it was added
    if num_cols != original_shape[-1]:
        x_2d = x_2d[:, :original_shape[-1]]
    
    # Restore original shape
    if was_1d:
        x = x_2d.squeeze(0)
    else:
        x = x_2d.view(original_shape)
    
    return x


def symmetric_quantize(x: torch.Tensor, config) -> torch.Tensor:
    """
    Apply symmetric scalar quantization.
    
    Args:
        x: Input tensor
        config: ScalarConfig instance
        
    Returns:
        Quantized tensor
    """
    # Compute row norms and apply scaling
    row_norms, row_norms_safe = _compute_row_norms(x)
    
    # Apply per-row scaling
    if config.per_row_scaling:
        x_scaled = x / row_norms_safe
    else:
        x_scaled = x
    
    # Apply manual scale factor if provided
    if config.scale_factor is not None:
        x_scaled = x_scaled * config.scale_factor
    
    # Reshape to blocks
    x_blocks, block_info = _reshape_to_blocks(x_scaled, config.block_size)
    
    # Quantize each block
    if config.block_size is None:
        # Row-wise quantization
        x_quantized = _quantize_symmetric_block(x_blocks, config)
    else:
        # Block-wise quantization
        x_quantized = _quantize_symmetric_block(x_blocks, config)
    
    # Reshape back to original shape
    x_quantized = _unreshape_from_blocks(x_quantized, block_info, x.shape)
    
    # Scale back by row norms
    if config.per_row_scaling:
        x_quantized = x_quantized * row_norms_safe
    
    # Scale back by manual scale factor
    if config.scale_factor is not None:
        x_quantized = x_quantized / config.scale_factor
    
    return x_quantized


def asymmetric_quantize(x: torch.Tensor, config) -> torch.Tensor:
    """
    Apply asymmetric scalar quantization.
    
    Args:
        x: Input tensor
        config: ScalarConfig instance
        
    Returns:
        Quantized tensor
    """
    # Compute row norms and apply scaling
    row_norms, row_norms_safe = _compute_row_norms(x)
    
    # Apply per-row scaling
    if config.per_row_scaling:
        x_scaled = x / row_norms_safe
    else:
        x_scaled = x
    
    # Apply manual scale factor if provided
    if config.scale_factor is not None:
        x_scaled = x_scaled * config.scale_factor
    
    # Reshape to blocks
    x_blocks, block_info = _reshape_to_blocks(x_scaled, config.block_size)
    
    # Quantize each block
    if config.block_size is None:
        # Row-wise quantization
        x_quantized = _quantize_asymmetric_block(x_blocks, config)
    else:
        # Block-wise quantization
        x_quantized = _quantize_asymmetric_block(x_blocks, config)
    
    # Reshape back to original shape
    x_quantized = _unreshape_from_blocks(x_quantized, block_info, x.shape)
    
    # Scale back by row norms
    if config.per_row_scaling:
        x_quantized = x_quantized * row_norms_safe
    
    # Scale back by manual scale factor
    if config.scale_factor is not None:
        x_quantized = x_quantized / config.scale_factor
    
    return x_quantized


def _quantize_symmetric_block(x_block: torch.Tensor, config) -> torch.Tensor:
    """
    Quantize a single block using symmetric quantization.
    
    Args:
        x_block: Input block tensor of shape [num_blocks, block_size]
        config: ScalarConfig instance
        
    Returns:
        Quantized block tensor
    """
    # Find min and max values across the block
    min_val = x_block.min(dim=-1, keepdim=True)[0]  # [num_blocks, 1]
    max_val = x_block.max(dim=-1, keepdim=True)[0]  # [num_blocks, 1]
    
    # Compute scale factor
    range_val = max_val - min_val
    range_val = torch.clamp(range_val, min=1e-8)  # Avoid division by zero
    
    # Symmetric range: use the larger absolute value
    abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
    bits = config.effective_bits()  # Use default (no matrix shape)
    scale = abs_max / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize
    x_quantized = torch.round(x_block / scale)
    
    # Clamp to valid range
    min_val_quantized = config.min_value()
    max_val_quantized = config.max_value()
    x_quantized = torch.clamp(x_quantized, 
                             min=min_val_quantized, 
                             max=max_val_quantized)
    
    # Dequantize
    x_dequantized = x_quantized * scale
    
    return x_dequantized


def _quantize_asymmetric_block(x_block: torch.Tensor, config) -> torch.Tensor:
    """
    Quantize a single block using asymmetric quantization.
    
    Args:
        x_block: Input block tensor of shape [num_blocks, block_size]
        config: ScalarConfig instance
        
    Returns:
        Quantized block tensor
    """
    # Find min and max values across the block
    min_val = x_block.min(dim=-1, keepdim=True)[0]  # [num_blocks, 1]
    max_val = x_block.max(dim=-1, keepdim=True)[0]  # [num_blocks, 1]
    
    bits = config.effective_bits()  # Use default (no matrix shape)
    
    if config.asymmetric_method == "zero_point":
        # Zero-point method
        scale = (max_val - min_val) / (2 ** bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-min_val / scale)
        zero_point = torch.clamp(zero_point, min=0, max=2 ** bits - 1)
        
        # Quantize
        x_quantized = torch.round((x_block - min_val) / scale)
        x_quantized = torch.clamp(x_quantized, min=0, max=2 ** bits - 1)
        
        # Dequantize
        x_dequantized = x_quantized * scale + min_val
        
    else:  # affine method
        # Affine method
        scale = (max_val - min_val) / (2 ** bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        offset = torch.round(-min_val / scale)
        
        # Quantize
        x_quantized = torch.round(x_block / scale + offset)
        x_quantized = torch.clamp(x_quantized, min=0, max=2 ** bits - 1)
        
        # Dequantize
        x_dequantized = (x_quantized - offset) * scale
    
    return x_dequantized


def _quantize_symmetric_matrix(x_matrix: torch.Tensor, config, matrix_shape: Tuple[int, int], dtype_bits: int = 32) -> torch.Tensor:
    """
    Quantize a matrix using symmetric quantization at matrix level.
    
    Args:
        x_matrix: Input matrix tensor of shape [num_rows, num_cols]
        config: ScalarConfig instance
        matrix_shape: Tuple of (rows, cols) for bit adjustment
        dtype_bits: Number of bits in the data type (32 for fp32, 16 for fp16, etc.)
        
    Returns:
        Quantized matrix tensor
    """
    # Find min and max values across the entire matrix
    min_val = x_matrix.min()
    max_val = x_matrix.max()
    
    # Compute scale factor using adjusted bits
    bits = config.effective_bits(matrix_shape, dtype_bits)
    abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
    scale = abs_max / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize
    x_quantized = torch.round(x_matrix / scale)
    
    # Clamp to valid range
    min_val_quantized = config.min_value(matrix_shape, dtype_bits)
    max_val_quantized = config.max_value(matrix_shape, dtype_bits)
    x_quantized = torch.clamp(x_quantized, min=min_val_quantized, max=max_val_quantized)
    
    # Dequantize
    x_dequantized = x_quantized * scale
    
    return x_dequantized


def _quantize_asymmetric_matrix(x_matrix: torch.Tensor, config, matrix_shape: Tuple[int, int], dtype_bits: int = 32) -> torch.Tensor:
    """
    Quantize a matrix using asymmetric quantization at matrix level.
    
    Args:
        x_matrix: Input matrix tensor of shape [num_rows, num_cols]
        config: ScalarConfig instance
        matrix_shape: Tuple of (rows, cols) for bit adjustment
        dtype_bits: Number of bits in the data type (32 for fp32, 16 for fp16, etc.)
        
    Returns:
        Quantized matrix tensor
    """
    # Find min and max values across the entire matrix
    min_val = x_matrix.min()
    max_val = x_matrix.max()
    
    # Compute scale factor using adjusted bits
    bits = config.effective_bits(matrix_shape, dtype_bits)
    
    if config.asymmetric_method == "zero_point":
        # Zero-point method
        scale = (max_val - min_val) / (2 ** bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-min_val / scale)
        zero_point = torch.clamp(zero_point, min=0, max=2 ** bits - 1)
        
        # Quantize
        x_quantized = torch.round((x_matrix - min_val) / scale)
        x_quantized = torch.clamp(x_quantized, min=0, max=2 ** bits - 1)
        
        # Dequantize
        x_dequantized = x_quantized * scale + min_val
        
    else:  # affine method
        # Affine method
        scale = (max_val - min_val) / (2 ** bits - 1)
        scale = torch.clamp(scale, min=1e-8)
        offset = torch.round(-min_val / scale)
        
        # Quantize
        x_quantized = torch.round(x_matrix / scale + offset)
        x_quantized = torch.clamp(x_quantized, min=0, max=2 ** bits - 1)
        
        # Dequantize
        x_dequantized = (x_quantized - offset) * scale
    
    return x_dequantized


def matrix_level_quantize(x: torch.Tensor, config, dtype_bits: int = 32) -> torch.Tensor:
    """
    Apply matrix-level scalar quantization (whole matrix as one unit).
    
    Args:
        x: Input tensor
        config: ScalarConfig instance
        dtype_bits: Number of bits in the data type (32 for fp32, 16 for fp16, etc.)
        
    Returns:
        Quantized tensor
    """
    # Get matrix shape for bit adjustment
    if x.dim() == 1:
        matrix_shape = (1, x.shape[0])
    else:
        # Flatten all but the last dimension to rows
        x_2d = x.view(-1, x.shape[-1])
        matrix_shape = x_2d.shape
    
    # Compute row norms and apply scaling
    row_norms, row_norms_safe = _compute_row_norms(x)
    
    # Apply per-row scaling
    if config.per_row_scaling:
        x_scaled = x / row_norms_safe
    else:
        x_scaled = x
    
    # Apply manual scale factor if provided
    if config.scale_factor is not None:
        x_scaled = x_scaled * config.scale_factor
    
    # Flatten to 2D for matrix-level quantization
    if x_scaled.dim() == 1:
        x_2d = x_scaled.unsqueeze(0)
    else:
        x_2d = x_scaled.view(-1, x_scaled.shape[-1])
    
    # Matrix-level quantization
    if config.mode == "symmetric":
        x_quantized = _quantize_symmetric_matrix(x_2d, config, matrix_shape, dtype_bits)
    else:  # asymmetric
        x_quantized = _quantize_asymmetric_matrix(x_2d, config, matrix_shape, dtype_bits)
    
    # Reshape back to original shape
    x_quantized = x_quantized.view(x.shape)
    
    # Scale back by row norms
    if config.per_row_scaling:
        x_quantized = x_quantized * row_norms_safe
    
    # Scale back by manual scale factor
    if config.scale_factor is not None:
        x_quantized = x_quantized / config.scale_factor
    
    return x_quantized


def scalar_quantize(x: torch.Tensor, config, dtype_bits: int = 32) -> torch.Tensor:
    """
    Apply scalar quantization based on configuration.
    
    Args:
        x: Input tensor
        config: ScalarConfig instance
        dtype_bits: Number of bits in the data type (32 for fp32, 16 for fp16, etc.)
        
    Returns:
        Quantized tensor
    """
    if config.matrix_level_quantization:
        return matrix_level_quantize(x, config, dtype_bits)
    elif config.mode == "symmetric":
        return symmetric_quantize(x, config)
    else:  # asymmetric
        return asymmetric_quantize(x, config)


def batch_scalar_quantize(x: torch.Tensor, config) -> torch.Tensor:
    """
    Apply scalar quantization to a batch of tensors.
    
    Args:
        x: Input tensor of shape [batch_size, ...]
        config: ScalarConfig instance
        
    Returns:
        Quantized tensor
    """
    # For now, just apply scalar_quantize to the entire tensor
    # In the future, this could be optimized for batch processing
    return scalar_quantize(x, config)


def scalar_dequantize(x_quantized: torch.Tensor, config) -> torch.Tensor:
    """
    Dequantize a scalar quantized tensor.
    
    Note: This is mainly for completeness. In practice, the quantization
    functions already include dequantization in the forward pass.
    
    Args:
        x_quantized: Quantized tensor
        config: ScalarConfig instance
        
    Returns:
        Dequantized tensor (same as input for our implementation)
    """
    # In our implementation, quantization already includes dequantization
    return x_quantized
