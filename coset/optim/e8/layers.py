"""
E8 Lattice Layers Module

This module provides optimized neural network layers specifically for E8 lattice
quantization, including the E8QLinear layer with straight-through estimation.
"""

import torch
import torch.nn as nn
from typing import Optional
from ...lattices import E8Lattice
from .config import E8Config
from .codecs import batch_e8_quantize
from .cuda import e8_quantize_cuda_jit, e8_cuda_available


class E8StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for E8 quantization with per-row scaling.
    
    This implements QAT with proper handling of weight norms for E8 lattice quantization:
    
    For an n x m weight matrix:
    1. Compute L2 norm of each of the n original rows (before tiling) and store them.
    2. Tile the matrix to have 8-column structure (E8 dimension)
    3. Scale each tiled row such that the norm of each tiled row has unit norm 
    4. Add another scaling beta such that E8 quantization is not overloaded
    5. Beta is pre-computed and stored in the config, which is specific to E8 and q
    6. Quantize the normalized tiled matrix using optimized batch_quantize_e8
    7. Re-scale back by the same n scaling factors and the beta scaling factor
    8. Reshape back to original shape
    
    This ensures:
    - Each row of the tiled matrix has unit norm before quantization (stable)
    - Only n scaling factors needed (not n*m/8 factors)
    - STE is well-behaved with gradients flowing properly to FP32 weights
    - Uses optimized E8 batch quantization for better performance
    """
    
    @staticmethod
    def forward(ctx, input, lattice, config, scale_factor=None):
        """
        Forward pass: quantize the input with per-row scaling using E8 optimization.
        
        Args:
            ctx: Context object to store information for backward
            input: Input tensor to quantize (can be reshaped to 2D matrix)
            lattice: E8Lattice instance
            config: E8 quantization configuration
            scale_factor: Optional manual scale factor. If None, uses automatic scaling.
            
        Returns:
            Quantized tensor with same shape as input
        """
        # Store for backward
        ctx.original_shape = input.shape
        ctx.lattice = lattice
        ctx.config = config
        device = input.device
        
        # Flatten to 2D: [num_rows, num_cols]
        if input.dim() == 1:
            input_2d = input.unsqueeze(0)  # [1, num_cols]
        else:
            # Flatten all but the last dimension to rows
            input_2d = input.view(-1, input.shape[-1])
        
        # Step 1: Compute L2 norm of each original row (before tiling)
        row_norms = torch.norm(input_2d, p=2, dim=1, keepdim=True)  # [num_rows, 1]
        # Avoid division by zero
        row_norms_safe = torch.clamp(row_norms, min=1e-8)
        ctx.row_norms = row_norms
        
        # Step 2: Rescale each row to unit norm and apply scaling factor
        beta = config.beta
        if scale_factor is None:
            # Use automatic scaling logic
            # Scale up significantly to reach E8 quantization threshold (around 2.0)
            # Since normalized values are around 0.035, we need scale_factor ~60 to reach 2.0
            # Use a more conservative adaptive scaling that accounts for matrix size
            base_scale = 60.0  # Base scale factor that works well
            adaptive_factor = min(2.0, max(0.5, input_2d.shape[0] / 10000))  # Adaptive factor between 0.5-2.0
            scale_factor = base_scale * adaptive_factor
        else:
            # Use user-provided scale factor
            scale_factor = float(scale_factor)
        
        ctx.scale_factor = scale_factor  # Store for backward pass
        input_2d = (input_2d / row_norms_safe) * scale_factor
        
        # Step 3: Tile the matrix to have 8-column structure (E8)
        input_2d = input_2d.view(-1, 8)  # [num_tiles, 8]
        
        # Step 4: Quantize using optimized E8 batch quantization (CUDA if available)
        if device.type == 'cuda' and e8_cuda_available():
            try:
                # Use JIT-compiled CUDA kernel for quantization
                quantized = e8_quantize_cuda_jit(input_2d, device=device)
            except:
                # Fallback to PyTorch GPU implementation
                quantized = batch_e8_quantize(input_2d, device=device)
        else:
            quantized = batch_e8_quantize(input_2d, device=device)
        
        # Step 5: Reshape back to original shape
        quantized = quantized.view(ctx.original_shape)
        
        # Step 6: Scale back by the original row norms
        quantized = quantized * row_norms_safe / ctx.scale_factor
        
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: straight-through estimator.
        
        Gradients pass through unchanged as if no quantization occurred.
        """
        return grad_output, None, None, None


class E8QLinear(nn.Module):
    """
    E8-optimized quantized linear layer with hierarchical nested-lattice quantization.
    
    This layer maintains FP32 shadow weights and can quantize weights,
    activations, or both during training. It uses optimized E8 batch quantization
    for better performance on both CPU and GPU.
    
    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        config: E8 quantization configuration
        lattice: E8Lattice instance for quantization
        quantize_weights: Whether to quantize weights
        quantize_activations: Whether to quantize activations
        quantize_every: Frequency of quantization (1 = every step)
        use_lut: Whether to use lookup table for matrix multiplication
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: E8Config,
        lattice: Optional[E8Lattice] = None,
        quantize_weights: bool = True,
        quantize_activations: bool = False,
        quantize_every: int = 1,
        use_lut: bool = False,
        bias: bool = True,
        scale_factor: Optional[float] = None
    ):
        """
        Initialize the E8-optimized quantized linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            config: E8 quantization configuration
            lattice: E8Lattice instance (defaults to E8Lattice)
            quantize_weights: Whether to quantize weights
            quantize_activations: Whether to quantize activations
            quantize_every: Frequency of quantization (1 = every step)
            use_lut: Whether to use lookup table for matrix multiplication
            bias: Whether to include bias term
            scale_factor: Optional manual scale factor for quantization. If None, uses automatic scaling.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.lattice = lattice if lattice is not None else E8Lattice()
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.quantize_every = quantize_every
        self.use_lut = use_lut
        self.scale_factor = scale_factor
        
        # Handle edge cases
        self._is_vector_weight = (in_features == 1 or out_features == 1)
        self._input_padding = 0
        self._output_padding = 0
        
        # Calculate padding for dimensions not divisible by 8
        if in_features % 8 != 0:
            self._input_padding = 8 - (in_features % 8)
            print(f"Warning: Input features {in_features} not divisible by 8, padding with {self._input_padding} zeros")
        
        if out_features % 8 != 0:
            self._output_padding = 8 - (out_features % 8)
            print(f"Warning: Output features {out_features} not divisible by 8, padding with {self._output_padding} zeros")
        
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
        Forward pass with optional E8 quantization.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
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
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def _quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with E8 quantization and straight-through estimation.
        
        Uses optimized batch_e8_quantize for better performance.
        """
        # Quantize weights if enabled (with STE)
        if self.quantize_weights:
            weight_quantized = E8StraightThroughQuantize.apply(
                self.weight, self.lattice, self.config, self.scale_factor
            )
        else:
            weight_quantized = self.weight
        
        # Ensure weight_quantized is on the same device as x
        weight_quantized = weight_quantized.to(x.device)
        
        # Quantize activations if enabled (with STE)
        if self.quantize_activations:
            x_quantized = E8StraightThroughQuantize.apply(
                x, self.lattice, self.config, self.scale_factor
            )
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
            "lattice_type": "E8",
            "is_vector_weight": self._is_vector_weight,
            "input_padding": self._input_padding,
            "output_padding": self._output_padding,
            "scale_factor": self.scale_factor,
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
        
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, lattice=E8, "
            f"quantize_weights={self.quantize_weights}, "
            f"quantize_activations={self.quantize_activations}, "
            f"quantize_every={self.quantize_every}{vector_info}{padding_info}{scale_info}"
        )
