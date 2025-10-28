"""
Scalar Quantized Linear Layers

This module provides scalar quantized linear layers with straight-through
estimation for quantization-aware training.
"""

import torch
import torch.nn as nn
from typing import Optional
from .scalar.config import ScalarConfig
from .scalar.quantizers import scalar_quantize, batch_scalar_quantize


class ScalarStraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for scalar quantization with per-row scaling.
    
    This implements QAT with proper handling of weight norms for scalar quantization:
    
    For an n x m weight matrix:
    1. Compute L2 norm of each of the n original rows and store them.
    2. Reshape the matrix to have block_size structure (if block_size is specified)
    3. Scale each row such that the norm of each row has unit norm 
    4. Apply additional scaling factor if provided
    5. Quantize the normalized matrix using scalar quantization
    6. Re-scale back by the same n scaling factors
    
    This ensures:
    - Each row has unit norm before quantization (stable)
    - Only n scaling factors needed (not n*m/block_size factors)
    - STE is well-behaved with gradients flowing properly to FP32 weights
    - Uses optimized scalar quantization for better performance
    """
    
    @staticmethod
    def forward(ctx, input, config, scale_factor=None):
        """
        Forward pass: quantize the input with per-row scaling using scalar quantization.
        
        Args:
            ctx: Context object to store information for backward
            input: Input tensor to quantize (can be reshaped to 2D matrix)
            config: ScalarConfig instance
            scale_factor: Optional manual scale factor. If None, uses automatic scaling.
            
        Returns:
            Quantized tensor with same shape as input
        """
        # Store for backward
        ctx.original_shape = input.shape
        ctx.config = config
        device = input.device
        
        # Flatten to 2D: [num_rows, num_cols]
        if input.dim() == 1:
            input_2d = input.unsqueeze(0)  # [1, num_cols]
        else:
            # Flatten all but the last dimension to rows
            input_2d = input.view(-1, input.shape[-1])
        
        # Step 1: Compute L2 norm of each original row (before blocking)
        row_norms = torch.norm(input_2d, p=2, dim=1, keepdim=True)  # [num_rows, 1]
        # Avoid division by zero
        row_norms_safe = torch.clamp(row_norms, min=1e-8)
        ctx.row_norms = row_norms
        
        # Step 2: Rescale each row to unit norm and apply scaling factor
        if scale_factor is None:
            # Use automatic scaling logic
            # Scale up significantly to reach quantization threshold
            # Use a more conservative adaptive scaling that accounts for matrix size
            base_scale = 60.0  # Base scale factor that works well
            adaptive_factor = min(2.0, max(0.5, input_2d.shape[0] / 10000))  # Adaptive factor between 0.5-2.0
            scale_factor = base_scale * adaptive_factor
        else:
            # Use user-provided scale factor
            scale_factor = float(scale_factor)
        
        ctx.scale_factor = scale_factor  # Store for backward pass
        input_2d = (input_2d / row_norms_safe) * scale_factor
        
        # Step 3: Apply scalar quantization
        # Create a temporary config with the scale factor
        temp_config = ScalarConfig(
            q=config.q,
            M=config.M,
            mode=config.mode,
            block_size=config.block_size,
            per_row_scaling=False,  # We handle scaling manually
            scale_factor=None,  # We handle scaling manually
            asymmetric_method=config.asymmetric_method,
            with_dither=config.with_dither,
            with_tie_dither=config.with_tie_dither,
        )
        
        # Quantize
        quantized = scalar_quantize(input_2d, temp_config)
        
        # Step 4: Reshape back to original shape
        quantized = quantized.view(ctx.original_shape)
        
        # Step 5: Scale back by the original row norms
        quantized = quantized * row_norms_safe / ctx.scale_factor
        
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: straight-through estimator.
        
        Gradients pass through unchanged as if no quantization occurred.
        """
        return grad_output, None, None


class ScalarQLinear(nn.Module):
    """
    Scalar quantized linear layer with hierarchical nested-lattice quantization.
    
    This layer implements quantization-aware training (QAT) for scalar quantization
    with support for both symmetric and asymmetric quantization modes.
    
    Features:
    - Per-row L2 norm scaling for stable quantization
    - Block-based quantization with configurable block size
    - Vector weight detection (skips quantization for single input/output features)
    - Padding for dimensions not divisible by block_size
    - Manual scale factor override
    - Straight-through estimation for gradients
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: ScalarConfig,
        quantize_weights: bool = True,
        quantize_activations: bool = False,
        quantize_every: int = 1,
        bias: bool = True,
        scale_factor: Optional[float] = None,
        matrix_level_quantization: bool = True
    ):
        """
        Initialize the scalar quantized linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            config: ScalarConfig instance
            quantize_weights: Whether to quantize weights
            quantize_activations: Whether to quantize activations
            quantize_every: Frequency of quantization (1 = every step)
            bias: Whether to include bias term
            scale_factor: Optional manual scale factor for quantization. If None, uses automatic scaling.
            matrix_level_quantization: Whether to use matrix-level quantization (default: True)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_every = quantize_every
        self.scale_factor = scale_factor
        self.matrix_level_quantization = matrix_level_quantization
        
        # Create a modified config with matrix-level quantization enabled if requested
        if matrix_level_quantization:
            from dataclasses import replace
            self.config = replace(config, matrix_level_quantization=True)
        else:
            self.config = config
        
        # Handle edge cases
        self._is_vector_weight = (in_features == 1 or out_features == 1)
        self._input_padding = 0
        self._output_padding = 0
        
        # Calculate padding for dimensions not divisible by block_size
        if config.block_size is not None:
            if in_features % config.block_size != 0:
                self._input_padding = config.block_size - (in_features % config.block_size)
                print(f"Warning: Input features {in_features} not divisible by block_size {config.block_size}, padding with {self._input_padding} zeros")
            
            if out_features % config.block_size != 0:
                self._output_padding = config.block_size - (out_features % config.block_size)
                print(f"Warning: Output features {out_features} not divisible by block_size {config.block_size}, padding with {self._output_padding} zeros")
        
        # Calculate effective dimensions (after padding)
        effective_in_features = in_features + self._input_padding
        effective_out_features = out_features + self._output_padding
        
        # Initialize FP32 shadow weights
        self.weight = nn.Parameter(torch.randn(effective_out_features, effective_in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(effective_out_features)) if bias else None
        
        # If weight is a vector, disable quantization
        if self._is_vector_weight:
            print(f"Warning: Weight matrix is a vector (in_features={in_features}, out_features={out_features}), disabling quantization")
            self.quantize_weights = False
        
        # Quantization step counter
        self._step_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional scalar quantization.
        """
        # Add input padding if needed
        if self._input_padding > 0:
            padding = torch.zeros(x.size(0), self._input_padding, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        # Check if we should quantize this step
        should_quantize = (
            self.training and 
            self.quantize_every > 0 and 
            self._step_count % self.quantize_every == 0 and
            not self._is_vector_weight  # Don't quantize vector weights
        )
        
        if should_quantize:
            output = self._quantized_forward(x)
        else:
            output = self._standard_forward(x)
        
        # Remove output padding if needed
        if self._output_padding > 0:
            output = output[:, :self.out_features]
        
        return output
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without quantization."""
        # Get weight on the same device as x
        weight = self.weight.to(x.device)
        bias = self.bias.to(x.device) if self.bias is not None else None
        
        # Perform matrix multiplication
        output = torch.nn.functional.linear(x, weight, bias)
        
        # Increment step counter
        self._step_count += 1
        
        return output
    
    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with scalar quantization and straight-through estimation.
        """
        # Quantize weights if enabled (with STE)
        if self.quantize_weights:
            weight_quantized = ScalarStraightThroughQuantize.apply(
                self.weight, self.config, self.scale_factor
            )
        else:
            weight_quantized = self.weight
        
        # Ensure weight_quantized is on the same device as x
        weight_quantized = weight_quantized.to(x.device)
        
        # Quantize activations if enabled (with STE)
        if self.quantize_activations:
            x_quantized = ScalarStraightThroughQuantize.apply(
                x, self.config, self.scale_factor
            )
        else:
            x_quantized = x
        
        # Get bias on the same device
        bias = self.bias.to(x.device) if self.bias is not None else None
        
        # Perform matrix multiplication
        output = torch.nn.functional.linear(x_quantized, weight_quantized, bias)
        
        # Increment step counter
        self._step_count += 1
        
        return output
    
    def get_quantization_stats(self) -> dict:
        """Get statistics about quantization."""
        return {
            "step_count": self._step_count,
            "quantize_weights": self.quantize_weights,
            "quantize_activations": self.quantize_activations,
            "quantize_every": self.quantize_every,
            "quantization_type": "scalar",
            "is_vector_weight": self._is_vector_weight,
            "input_padding": self._input_padding,
            "output_padding": self._output_padding,
            "scale_factor": self.scale_factor,
            "matrix_level_quantization": self.matrix_level_quantization,
            "config": self.config.to_dict(),
        }
    
    def reset_step_count(self):
        """Reset the quantization step counter."""
        self._step_count = 0
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        vector_info = f", vector_weight={self._is_vector_weight}" if self._is_vector_weight else ""
        padding_info = ""
        if self._input_padding > 0 or self._output_padding > 0:
            padding_info = f", padding=({self._input_padding},{self._output_padding})"
        scale_info = f", scale_factor={self.scale_factor}" if self.scale_factor is not None else ""
        mode_info = f", mode={self.config.mode}"
        block_info = f", block_size={self.config.block_size}" if self.config.block_size is not None else ""
        matrix_info = f", matrix_level={self.matrix_level_quantization}"
        
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantization=scalar, "
            f"quantize_weights={self.quantize_weights}, "
            f"quantize_activations={self.quantize_activations}, "
            f"quantize_every={self.quantize_every}{vector_info}{padding_info}{scale_info}{mode_info}{block_info}{matrix_info}"
        )
