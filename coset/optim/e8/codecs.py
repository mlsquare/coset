"""
E8 Lattice Codecs Module

This module provides optimized encode/decode functions specifically for E8 lattice
quantization, including both CPU and GPU implementations.
"""

import warnings
from typing import Tuple, Optional
import torch
from ...lattices import E8Lattice
from .config import E8Config


def e8_quantize(x: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    E8 nearest neighbor quantization.
    
    Find the closest point in the E8 lattice to the input vector.
    Optimized for single vector quantization.
    
    Args:
        x: Input vector of shape [8] or [batch_size, 8]
        device: Device to perform computation on (defaults to x's device)
        
    Returns:
        Quantized vector of same shape as input
    """
    if device is None:
        device = x.device
    
    x = x.to(device)
    original_shape = x.shape
    
    # Handle both single vector and batch cases
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Use batch implementation
    result = batch_e8_quantize(x, device)
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


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
    dist_0 = torch.sum((X - y_0) ** 2, dim=1)
    dist_1 = torch.sum((X - y_1) ** 2, dim=1)
    
    closer_mask = (dist_0 < dist_1).unsqueeze(1)
    Y = torch.where(closer_mask, y_0, y_1)
    
    return Y


def e8_encode(
    x: torch.Tensor, 
    config: E8Config,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, int]:
    """
    E8 hierarchical encoding.
    
    Encode a vector using hierarchical nested lattice quantization with M levels
    and handle overload by scaling the vector until quantization succeeds.
    
    Args:
        x: Input vector to be quantized (shape [8])
        config: E8 quantization configuration
        dither: Optional dither vector for randomized quantization
        device: Device to perform computation on (defaults to x's device)
        
    Returns:
        Tuple of (encoding_vectors, T) where:
        - encoding_vectors: Tensor of shape [M, 8] containing M encoding vectors
        - T: Number of scaling operations performed to handle overload
    """
    # Determine device
    if device is None:
        device = x.device
    
    # Ensure x is on the correct device
    x = x.to(device).flatten()
    if x.shape[0] != 8:
        raise ValueError(f"Input dimension {x.shape[0]} doesn't match E8 dimension 8")
    
    # Move dither to correct device
    if dither is not None:
        dither = dither.to(device)
    
    # Apply scaling and dithering
    x_scaled = x / config.beta
    if config.with_dither and dither is not None:
        x_scaled = x_scaled + dither.flatten()
    
    # Perform hierarchical encoding
    encoding_vectors, did_overload = _e8_encode_internal(x_scaled, config)
    t = 0
    
    # Handle overload with scaling
    if not config.disable_overload_protection:
        while did_overload and t < config.max_scaling_iterations:
            t += 1
            x_scaled = x_scaled / (2 ** config.alpha)
            encoding_vectors, did_overload = _e8_encode_internal(x_scaled, config)
        
        if did_overload:
            warnings.warn(
                f"Overload not resolved after {config.max_scaling_iterations} iterations. "
                "Consider increasing max_scaling_iterations or adjusting parameters."
            )
    
    return encoding_vectors, t


def _e8_encode_internal(
    x: torch.Tensor, 
    config: E8Config,
    check_overload: bool = True
) -> Tuple[torch.Tensor, bool]:
    """
    Internal E8 encoding function that performs hierarchical quantization.
    
    Args:
        x: Input vector (already scaled and dithered)
        config: E8 quantization configuration
        check_overload: Whether to check for overload (can be disabled for performance)
        
    Returns:
        Tuple of (encoding_vectors, overload_error)
    """
    x_l = x.clone()
    encoding_vectors = []
    
    for _ in range(config.M):
        # Quantize to E8 lattice
        quantized = e8_quantize(x_l, device=x.device)
        encoding_vectors.append(quantized)
        
        # Scale down for next level
        x_l = x_l / config.q
    
    encoding_vectors = torch.stack(encoding_vectors, dim=0)  # [M, 8]
    
    # Check for overload (simplified check)
    did_overload = False
    if check_overload and not config.disable_overload_protection:
        # Check if any encoding vector has values outside expected range
        max_val = torch.max(torch.abs(encoding_vectors))
        if max_val > config.q:
            did_overload = True
    
    return encoding_vectors, did_overload


def e8_decode(
    encoding_vectors: torch.Tensor,
    T: int,
    config: E8Config,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    E8 hierarchical decoding.
    
    Decode a vector from hierarchical encoding vectors and scaling factor.
    
    Args:
        encoding_vectors: Tensor of shape [M, 8] containing encoding vectors
        T: Number of scaling operations that were applied during encoding
        config: E8 quantization configuration
        device: Device to perform computation on (defaults to encoding_vectors' device)
        
    Returns:
        Decoded vector of shape [8]
    """
    if device is None:
        device = encoding_vectors.device
    
    encoding_vectors = encoding_vectors.to(device)
    
    if encoding_vectors.shape != (config.M, 8):
        raise ValueError(f"Expected encoding_vectors shape [{config.M}, 8], got {encoding_vectors.shape}")
    
    # Initialize output
    x_hat = torch.zeros(8, device=device)
    
    # Process each level
    for i in range(config.M):
        b_i = encoding_vectors[i]  # [8]
        
        # Convert encoding coordinates to lattice point
        # For E8, this is just the encoding vector itself (identity mapping)
        Gb = b_i
        
        # Compute quantization error
        Gb_scaled = Gb / config.q
        Gb_quantized = e8_quantize(Gb_scaled, device=device)
        
        # Compute x_i_hat = Gb - q * Gb_quantized
        x_i_hat = Gb - config.q * Gb_quantized
        
        # Accumulate with appropriate weight
        weight = config.q ** i
        x_hat += weight * x_i_hat
    
    # Apply scaling compensation
    scale_factor = config.beta * (2 ** T)
    x_hat *= scale_factor
    
    return x_hat


def batch_e8_encode(
    X: torch.Tensor,
    config: E8Config,
    dither: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch E8 encoding for multiple vectors.
    
    Args:
        X: Input tensor of shape [batch_size, 8]
        config: E8 quantization configuration
        dither: Optional dither tensor of shape [batch_size, 8]
        device: Device to perform computation on (defaults to X's device)
        
    Returns:
        Tuple of (encoding_vectors, T_values) where:
        - encoding_vectors: Tensor of shape [batch_size, M, 8]
        - T_values: Tensor of shape [batch_size] containing T values
    """
    if device is None:
        device = X.device
    
    X = X.to(device)
    batch_size = X.shape[0]
    
    # Apply scaling and dithering
    X_scaled = X / config.beta
    if config.with_dither and dither is not None:
        dither = dither.to(device)
        X_scaled = X_scaled + dither
    
    # Process each vector in the batch
    encoding_vectors_list = []
    T_values = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    for i in range(batch_size):
        x_i = X_scaled[i]
        dither_i = dither[i] if dither is not None else None
        
        # Encode single vector
        enc_vec, t = e8_encode(x_i, config, dither_i, device)
        encoding_vectors_list.append(enc_vec)
        T_values[i] = t
    
    encoding_vectors = torch.stack(encoding_vectors_list, dim=0)  # [batch_size, M, 8]
    
    return encoding_vectors, T_values


def batch_e8_decode(
    encoding_vectors: torch.Tensor,
    T_values: torch.Tensor,
    config: E8Config,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Batch E8 decoding for multiple vectors.
    
    Args:
        encoding_vectors: Tensor of shape [batch_size, M, 8]
        T_values: Tensor of shape [batch_size] containing T values
        config: E8 quantization configuration
        device: Device to perform computation on (defaults to encoding_vectors' device)
        
    Returns:
        Decoded tensor of shape [batch_size, 8]
    """
    if device is None:
        device = encoding_vectors.device
    
    encoding_vectors = encoding_vectors.to(device)
    T_values = T_values.to(device)
    batch_size = encoding_vectors.shape[0]
    
    # Process each vector in the batch
    decoded_list = []
    
    for i in range(batch_size):
        enc_vec = encoding_vectors[i]
        t = T_values[i].item()
        
        # Decode single vector
        decoded = e8_decode(enc_vec, t, config, device)
        decoded_list.append(decoded)
    
    return torch.stack(decoded_list, dim=0)  # [batch_size, 8]
