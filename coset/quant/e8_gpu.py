"""
E8 GPU-accelerated quantization module.

This module provides optimized PyTorch implementations of E8 lattice quantization
operations optimized for GPU batch processing. These are designed to be faster than
the CPU reference implementations and will be further accelerated by CUDA kernels.
"""

import torch
from typing import Optional, Tuple
from ..lattices import E8Lattice
from .params import QuantizationConfig


def batch_e8_quantize(X: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Vectorized E8 quantization for batch processing.
    
    Optimized for [batch_size, 8] tensors with vectorized operations.
    Uses PyTorch operations to maximize parallelism across batch dimension.
    
    Args:
        X: Input tensor of shape [batch_size, 8]
        device: Device to perform computation on (defaults to X's device)
        
    Returns:
        Quantized tensor of shape [batch_size, 8]
    """
    if device is None:
        device = X.device
    
    X = X.to(device)
    batch_size = X.shape[0]
    
    # Pre-allocate output
    Y = torch.zeros_like(X)
    
    # Helper function for custom rounding (vectorized)
    def custom_round_vec(x: torch.Tensor) -> torch.Tensor:
        """Vectorized custom rounding."""
        eps = torch.finfo(x.dtype).eps
        y = x - torch.sign(x) * eps
        return torch.floor(y + 0.5)
    
    # Compute both D8 candidates in parallel
    # Candidate 0: D8
    f_x = custom_round_vec(X)
    
    # Check sum parity for candidate 0
    sum_parity = torch.sum(f_x, dim=1) % 2  # [batch_size]
    
    # Find coordinate farthest from integer (vectorized)
    delta = torch.abs(X - f_x)  # [batch_size, 8]
    k_indices = torch.argmax(delta, dim=1)  # [batch_size]
    
    # Build g_x result for candidate 0
    g_x = f_x.clone()
    batch_indices = torch.arange(batch_size, device=device)
    
    x_k_vals = X[batch_indices, k_indices]
    f_x_k_vals = f_x[batch_indices, k_indices]
    
    # Vectorized conditional update
    positive_mask = x_k_vals >= 0
    flip_positive = positive_mask & (f_x_k_vals < x_k_vals)
    flip_negative = ~positive_mask & (f_x_k_vals <= x_k_vals)
    flip_mask = flip_positive | flip_negative
    
    g_x[batch_indices[flip_mask], k_indices[flip_mask]] += 1
    g_x[batch_indices[~flip_mask], k_indices[~flip_mask]] -= 1
    
    # Select between f_x and g_x based on parity
    # y_0 should use g_x when sum is odd, f_x when even
    odd_parity_mask = (sum_parity != 0).unsqueeze(1)  # [batch_size, 1]
    y_0 = torch.where(odd_parity_mask, g_x, f_x)
    
    # Candidate 1: D8 + (0.5)^8
    X_shifted = X - 0.5
    f_x_shifted = custom_round_vec(X_shifted)
    
    sum_parity_shifted = torch.sum(f_x_shifted, dim=1) % 2
    delta_shifted = torch.abs(X_shifted - f_x_shifted)
    k_indices_shifted = torch.argmax(delta_shifted, dim=1)
    
    g_x_shifted = f_x_shifted.clone()
    x_k_shifted_vals = X_shifted[batch_indices, k_indices_shifted]
    f_x_k_shifted_vals = f_x_shifted[batch_indices, k_indices_shifted]
    
    positive_mask_shifted = x_k_shifted_vals >= 0
    flip_positive_shifted = positive_mask_shifted & (f_x_k_shifted_vals < x_k_shifted_vals)
    flip_negative_shifted = ~positive_mask_shifted & (f_x_k_shifted_vals <= x_k_shifted_vals)
    flip_mask_shifted = flip_positive_shifted | flip_negative_shifted
    
    g_x_shifted[batch_indices[flip_mask_shifted], k_indices_shifted[flip_mask_shifted]] += 1
    g_x_shifted[batch_indices[~flip_mask_shifted], k_indices_shifted[~flip_mask_shifted]] -= 1
    
    odd_parity_mask_shifted = (sum_parity_shifted != 0).unsqueeze(1)
    y_1_mid = torch.where(odd_parity_mask_shifted, g_x_shifted, f_x_shifted)
    y_1 = y_1_mid + 0.5
    
    # Compute distances and select closer candidate
    dist_0 = torch.norm(X - y_0, dim=1)  # [batch_size]
    dist_1 = torch.norm(X - y_1, dim=1)  # [batch_size]
    
    closer_mask = (dist_0 < dist_1).unsqueeze(1)  # [batch_size, 1]
    Y = torch.where(closer_mask, y_0, y_1)
    
    return Y


def batch_encode_e8(
    X: torch.Tensor,
    lattice: E8Lattice,
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized batch encoding for E8 with vectorized operations.
    
    Args:
        X: Input matrix [batch_size, 8]
        lattice: E8Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        device: Device to perform computation on
        
    Returns:
        Tuple of (encoded_vectors [batch_size, M, 8], T_values [batch_size])
    """
    if device is None:
        device = X.device
    
    X = X.to(device)
    batch_size = X.shape[0]
    M = config.M
    q = config.q
    
    # Ensure lattice is on correct device
    lattice.G = lattice.G.to(device)
    lattice.G_inv = lattice.G_inv.to(device)
    
    # Apply scaling and dithering
    X_scaled = X / config.beta
    if config.with_dither and dither is not None:
        dither = dither.to(device).flatten()
        X_scaled = X_scaled + dither.unsqueeze(0)  # Broadcast to batch
    
    # Pre-allocate encoding tensor
    encodings = torch.zeros((batch_size, M, 8), dtype=torch.float32, device=device)
    T_values = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Current working tensor
    X_l = X_scaled.clone()
    
    # Process all M levels
    for m in range(M):
        # Quantize to lattice (vectorized)
        X_l = batch_e8_quantize(X_l, device=device)
        
        # Convert to encoding coordinates using matrix multiplication
        # encodings[b, m, :] = round(G_inv @ X_l[b, :]) % q
        # Use batched matrix multiplication
        G_inv_expanded = lattice.G_inv.unsqueeze(0).expand(batch_size, -1, -1)
        X_l_expanded = X_l.unsqueeze(2)  # [batch_size, 8, 1]
        coords = torch.bmm(G_inv_expanded, X_l_expanded).squeeze(2)  # [batch_size, 8]
        encodings[:, m, :] = torch.round(coords) % q
        
        # Scale down for next level
        X_l = X_l / q
    
    # Check for overload (vectorized across batch)
    X_l_check = batch_e8_quantize(X_l, device=device)
    zeros = torch.zeros_like(X_l_check)
    overload_mask = ~torch.allclose(X_l_check, zeros, atol=1e-8)
    
    # Handle overload if not disabled
    if not config.disable_overload_protection and torch.any(overload_mask):
        # For now, just track which ones overloaded
        # Full scaling loop would need per-sample handling
        T_values[overload_mask] = config.max_scaling_iterations
    
    return encodings, T_values


def batch_decode_e8(
    encodings: torch.Tensor,
    T_values: torch.Tensor,
    lattice: E8Lattice,
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Optimized batch decoding for E8 with vectorized operations.
    
    Args:
        encodings: Encoding vectors [batch_size, M, 8]
        T_values: Scaling counts [batch_size]
        lattice: E8Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        device: Device to perform computation on
        
    Returns:
        Decoded matrix [batch_size, 8]
    """
    if device is None:
        device = encodings.device
    
    encodings = encodings.to(device)
    T_values = T_values.to(device)
    batch_size = encodings.shape[0]
    M = encodings.shape[1]
    q = config.q
    
    # Ensure lattice is on correct device
    lattice.G = lattice.G.to(device)
    
    # Pre-allocate output
    X_hat = torch.zeros((batch_size, 8), dtype=torch.float32, device=device)
    
    # Process all M levels
    for i in range(M):
        b_i = encodings[:, i, :]  # [batch_size, 8]
        
        # Convert encoding coordinates to lattice point
        # Gb = G @ b_i for each batch sample
        G_expanded = lattice.G.unsqueeze(0).expand(batch_size, -1, -1)
        b_i_expanded = b_i.unsqueeze(2)  # [batch_size, 8, 1]
        Gb = torch.bmm(G_expanded, b_i_expanded).squeeze(2)  # [batch_size, 8]
        
        # Compute quantization error
        Gb_scaled = Gb / q
        Gb_quantized = batch_e8_quantize(Gb_scaled, device=device)
        x_i_hat = Gb - q * Gb_quantized
        
        # Accumulate with appropriate weight
        X_hat += (q ** i) * x_i_hat
    
    # Remove dither if applied
    if config.with_dither and dither is not None:
        dither = dither.to(device).flatten()
        X_hat = X_hat - dither.unsqueeze(0)
    
    # Apply scaling compensation
    scale_factors = config.beta * (2 ** (config.alpha * T_values.float()))
    X_hat = X_hat * scale_factors.unsqueeze(1)
    
    return X_hat


def batch_quantize_e8(
    X: torch.Tensor,
    lattice: E8Lattice,
    config: QuantizationConfig,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Complete batch quantization: encode and decode.
    
    Args:
        X: Input matrix [batch_size, 8]
        lattice: E8Lattice instance
        config: Quantization configuration
        dither: Optional dither vector
        device: Device to perform computation on
        
    Returns:
        Quantized matrix [batch_size, 8]
    """
    encodings, T_values = batch_encode_e8(X, lattice, config, dither, device)
    return batch_decode_e8(encodings, T_values, lattice, config, dither, device)
