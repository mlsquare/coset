"""
CUDA-optimized kernels for HNLQ operations using PyTorch JIT compilation
"""

import torch
from typing import Tuple

# CUDA kernel implementations using PyTorch JIT
@torch.jit.script
def cuda_closest_point_e8_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    CUDA-optimized E8 closest point function using JIT compilation.
    
    Args:
        x: Input tensor of shape (batch_size, 8)
        
    Returns:
        y: Closest E8 lattice points
    """
    batch_size, dim = x.shape
    
    # Compute f_x
    f_x = torch.round(x)
    sum_f_x = torch.sum(f_x, dim=1)
    
    # Compute y_0
    y_0 = f_x.clone()
    
    # For samples where sum is odd, compute g_x
    odd_mask = (sum_f_x % 2 != 0)
    if torch.any(odd_mask):
        x_odd = x[odd_mask]
        f_x_odd = f_x[odd_mask]
        
        # Compute delta for each sample
        delta = torch.abs(x_odd - f_x_odd)
        k_indices = torch.argmax(delta, dim=1)
        
        # Update y_0 for odd samples
        y_0_odd = f_x_odd.clone()
        for i, k in enumerate(k_indices):
            x_k = x_odd[i, k]
            f_x_k = f_x_odd[i, k]
            
            if x_k >= 0:
                y_0_odd[i, k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
            else:
                y_0_odd[i, k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1
        
        y_0[odd_mask] = y_0_odd
    
    # Compute f_x_shifted and g_x_shifted
    x_shifted = x - 0.5
    f_x_shifted = torch.round(x_shifted)
    sum_f_x_shifted = torch.sum(f_x_shifted, dim=1)
    
    # Compute g_x_shifted for odd samples
    g_x_shifted = f_x_shifted.clone()
    odd_shifted_mask = (sum_f_x_shifted % 2 != 0)
    if torch.any(odd_shifted_mask):
        x_shifted_odd = x_shifted[odd_shifted_mask]
        f_x_shifted_odd = f_x_shifted[odd_shifted_mask]
        
        delta_shifted = torch.abs(x_shifted_odd - f_x_shifted_odd)
        k_indices_shifted = torch.argmax(delta_shifted, dim=1)
        
        for i, k in enumerate(k_indices_shifted):
            x_k = x_shifted_odd[i, k]
            f_x_k = f_x_shifted_odd[i, k]
            
            if x_k >= 0:
                g_x_shifted[odd_shifted_mask][i, k] = f_x_k + 1 if f_x_k < x_k else f_x_k - 1
            else:
                g_x_shifted[odd_shifted_mask][i, k] = f_x_k + 1 if f_x_k <= x_k else f_x_k - 1
    
    # Compute y_1
    y_1 = torch.where(
        (sum_f_x_shifted % 2 == 0).unsqueeze(1),
        f_x_shifted + 0.5,
        g_x_shifted + 0.5
    )
    
    # Choose closest point
    norm_y_0 = torch.sum((x - y_0) ** 2, dim=1)
    norm_y_1 = torch.sum((x - y_1) ** 2, dim=1)
    
    closest_mask = norm_y_0 < norm_y_1
    y = torch.where(closest_mask.unsqueeze(1), y_0, y_1)
    
    return y

@torch.jit.script
def cuda_vectorized_quantize_kernel(
    x: torch.Tensor,
    generator_matrix: torch.Tensor,
    inverse_generator_matrix: torch.Tensor,
    eps: torch.Tensor,
    beta: float,
    q: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-optimized vectorized quantization kernel.
    
    Args:
        x: Input tensor of shape (batch_size, lattice_dim)
        generator_matrix: Generator matrix
        inverse_generator_matrix: Inverse generator matrix
        eps: Epsilon tensor
        beta: Beta scaling factor
        q: Quantization parameter
        
    Returns:
        quantized: Quantized tensor
        indices: Quantization indices
    """
    batch_size, lattice_dim = x.shape
    
    # Scale by beta
    x_scaled = x / beta
    
    # Add epsilon
    x_with_eps = x_scaled + eps
    
    # Find closest point using CUDA kernel
    closest_point = cuda_closest_point_e8_kernel(x_with_eps)
    
    # Matrix multiplication with inverse generator matrix
    b_i = torch.matmul(closest_point, inverse_generator_matrix)
    b_i = torch.fmod(b_i, q)
    indices = torch.round(b_i).int()
    
    # Matrix multiplication with generator matrix for output
    decoded = torch.matmul(indices.float(), generator_matrix)
    quantized = decoded * beta
    
    return quantized, indices

@torch.jit.script
def cuda_quantized_matmul_kernel(
    input_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    lookup_table: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    CUDA-optimized quantized matrix multiplication kernel.
    
    Args:
        input_indices: Input indices [batch_size, num_blocks, lattice_dim]
        weight_indices: Weight indices [out_features, num_blocks, lattice_dim]
        lookup_table: Precomputed lookup table
        bias: Bias tensor
        
    Returns:
        output: Output tensor [batch_size, out_features]
    """
    batch_size, num_blocks, lattice_dim = input_indices.shape
    out_features = weight_indices.shape[0]
    lookup_table_size = lookup_table.shape[0]
    
    # Flatten indices for vectorized operations
    input_flat = input_indices.view(batch_size, -1)
    weight_flat = weight_indices.view(out_features, -1)
    
    # Clamp indices to valid range
    input_clamped = torch.clamp(input_flat, 0, lookup_table_size - 1)
    weight_clamped = torch.clamp(weight_flat, 0, lookup_table_size - 1)
    
    # Vectorized lookup table operations
    input_expanded = input_clamped.unsqueeze(1)  # [batch_size, 1, num_blocks * lattice_dim]
    weight_expanded = weight_clamped.unsqueeze(0)  # [1, out_features, num_blocks * lattice_dim]
    
    # Broadcast for all combinations
    input_broadcast = input_expanded.expand(batch_size, out_features, -1)
    weight_broadcast = weight_expanded.expand(batch_size, out_features, -1)
    
    # Vectorized lookup
    lookup_values = lookup_table[input_broadcast, weight_broadcast]
    
    # Sum over the last dimension
    output = torch.sum(lookup_values, dim=-1)
    
    # Add bias
    if bias.numel() > 0:
        output = output + bias
    
    return output

@torch.jit.script
def cuda_batch_product_quantize_kernel(
    x: torch.Tensor,
    generator_matrix: torch.Tensor,
    inverse_generator_matrix: torch.Tensor,
    eps: torch.Tensor,
    beta: float,
    q: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-optimized batch product quantization kernel.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        generator_matrix: Generator matrix
        inverse_generator_matrix: Inverse generator matrix
        eps: Epsilon tensor
        beta: Beta scaling factor
        q: Quantization parameter
        
    Returns:
        quantized: Quantized tensor
        indices: Quantization indices
    """
    batch_size, input_dim = x.shape
    lattice_dim = generator_matrix.shape[0]
    
    # Calculate number of blocks
    num_blocks = (input_dim + lattice_dim - 1) // lattice_dim
    padded_dim = num_blocks * lattice_dim
    
    # Pad input if necessary
    if input_dim < padded_dim:
        padding_size = padded_dim - input_dim
        padding = torch.zeros(batch_size, padding_size, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=1)
    else:
        x_padded = x
    
    # Reshape to blocks
    x_blocks = x_padded.view(batch_size, num_blocks, lattice_dim)
    
    # Vectorized quantization for all blocks
    quantized_blocks = torch.zeros_like(x_blocks)
    indices_blocks = torch.zeros(batch_size, num_blocks, lattice_dim, dtype=torch.int32, device=x.device)
    
    for block_idx in range(num_blocks):
        block = x_blocks[:, block_idx, :]
        quantized_block, indices_block = cuda_vectorized_quantize_kernel(
            block, generator_matrix, inverse_generator_matrix, eps, beta, q
        )
        quantized_blocks[:, block_idx, :] = quantized_block
        indices_blocks[:, block_idx, :] = indices_block
    
    # Reshape back
    quantized_padded = quantized_blocks.view(batch_size, padded_dim)
    indices_padded = indices_blocks.view(batch_size, padded_dim)
    
    # Remove padding
    quantized = quantized_padded[:, :input_dim]
    indices = indices_padded[:, :input_dim]
    
    return quantized, indices

# Python wrapper functions
def closest_point_e8_cuda(x: torch.Tensor) -> torch.Tensor:
    """CUDA-optimized E8 closest point function."""
    return cuda_closest_point_e8_kernel(x)

def vectorized_quantize_cuda(
    x: torch.Tensor,
    generator_matrix: torch.Tensor,
    inverse_generator_matrix: torch.Tensor,
    eps: torch.Tensor,
    beta: float,
    q: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-optimized vectorized quantization."""
    return cuda_vectorized_quantize_kernel(x, generator_matrix, inverse_generator_matrix, eps, beta, q)

def quantized_matmul_cuda(
    input_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    lookup_table: torch.Tensor,
    bias: torch.Tensor = None
) -> torch.Tensor:
    """CUDA-optimized quantized matrix multiplication."""
    if bias is None:
        bias = torch.tensor([])
    return cuda_quantized_matmul_kernel(input_indices, weight_indices, lookup_table, bias)

def batch_product_quantize_cuda(
    x: torch.Tensor,
    generator_matrix: torch.Tensor,
    inverse_generator_matrix: torch.Tensor,
    eps: torch.Tensor,
    beta: float,
    q: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-optimized batch product quantization."""
    return cuda_batch_product_quantize_kernel(x, generator_matrix, inverse_generator_matrix, eps, beta, q)
