"""
Vector Quantization Layers

This module provides vector quantization layers with hierarchical nested lattice quantization (HNLQ)
and learnable scale quantization (LSQ) for quantization-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

def ste_round(x):
    """Straight-through estimator for rounding."""
    return (x - x.detach()) + torch.round(x).detach()

def ste_clip(x, min_val, max_val):
    """Straight-through estimator for clipping."""
    return (x - x.detach()) + torch.clamp(x, min_val, max_val).detach()


def get_generators(lattice) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get generator matrices from a lattice instance.
    
    This is a generic helper that works with any Lattice instance.
    For specific lattices, import from their modules:
    - from coset.core.e8 import E8Lattice
    - from coset.core.d4 import D4Lattice (when available)
    
    Args:
        lattice: A Lattice instance (e.g., E8Lattice, D4Lattice)
        
    Returns:
        Tuple of (G, G_inv) where:
        - G: Generator matrix of shape [d, d]
        - G_inv: Inverse of G of shape [d, d]
        
    Example:
        >>> from coset.core.e8 import E8Lattice
        >>> from coset.core.vq_layers import get_generators
        >>> lattice = E8Lattice()
        >>> G, Ginv = get_generators(lattice)
    """
    return lattice.get_generators()


# Note: Lattice-specific quantization functions are in their respective modules:
# - from coset.core.e8 import e8_quantize
# - from coset.core.d4 import d4_quantize (when available)


# ---------- Straight-Through Estimator (STE) Functions ----------

class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-through estimator for quantization.
    Forward pass: quantizes the input
    Backward pass: passes gradients through unchanged
    """
    @staticmethod
    def forward(ctx, x, quantize_fn, q):
        """
        Forward pass: quantize the input.
        
        Args:
            x: Input tensor to quantize
            quantize_fn: Lattice-specific quantization function
            q: Quantization parameter
        """
        # Store for backward pass
        ctx.quantize_fn = quantize_fn
        ctx.q = q
        ctx.save_for_backward(x)
        
        # Quantize
        return quantize_fn(x, q=q)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: pass gradients through unchanged (straight-through).
        """
        # Get original input
        x, = ctx.saved_tensors
        
        # Straight-through: pass gradients through unchanged
        return grad_output, None, None


def ste_quantize(x, quantize_fn, q):
    """
    Straight-through quantization wrapper.
    
    Args:
        x: Input tensor
        quantize_fn: Lattice-specific quantization function
        q: Quantization parameter
        
    Returns:
        Quantized tensor with STE gradients
    """
    return StraightThroughQuantize.apply(x, quantize_fn, q)


# ---------- Activation Quantizer (LSQ-A style, per-tensor) ----------

class LSQActivation(nn.Module):
    """
    Learnable per-tensor activation quantizer with STE.
    
    Uses learnable clip (alpha) for symmetric uniform quantization in [-alpha, +alpha].
    """
    
    def __init__(self, bit_width: int = 8, init_alpha: float = 1.0):
        super().__init__()
        self.bit_width = bit_width
        self.qmax = 2 ** (bit_width - 1) - 1  # e.g., 127 for 8-bit
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, x):
        # symmetric uniform quantization in [-alpha, +alpha]
        alpha = torch.relu(self.alpha) + 1e-8
        x_clipped = ste_clip(x, -alpha, alpha)
        scale = alpha / self.qmax
        y = x_clipped / scale
        yq = ste_round(y)
        return yq * scale  # quantized activation (STE)

# ---------- HNLQ Linear with flexible tiling & bias (not quantized) ----------

@dataclass
class HNLQConfig:
    """Configuration for HNLQ Linear layer."""
    q: int = 4
    M: int = 2
    Delta0: float = 1.5
    eta: float = 0.1
    k: int = 1
    tiling: str = 'row'  # 'row' or 'block'
    block_size: int = 8
    quantize_activations: bool = False
    act_bit_width: int = 8
    act_init_alpha: float = 1.0
    init_method: str = 'normal'
    init_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.init_kwargs is None:
            self.init_kwargs = {}
        if self.tiling not in ['row', 'block']:
            raise ValueError(f"tiling must be 'row' or 'block', got {self.tiling}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


class HNLQLinear(nn.Module):
    """
    Hierarchical Nested Lattice Quantization Linear Layer.
    
    This layer implements quantization-aware training with hierarchical nested lattice quantization
    for the weights and optional activation quantization using LSQ.
    
    The layer supports:
    - Learnable scaling factors (beta) for weights
    - Flexible tiling (row-level or block-level scaling)
    - Optional activation quantization
    - Multiple lattice types (E8, D4, etc.)
    - Various weight initialization methods
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        G: Generator matrix for the lattice
        Ginv: Inverse of the generator matrix
        quantize_fn: Lattice-specific quantization function (e.g., e8_quantize)
        lattice_type: Type of lattice ('E8', 'D4', etc.) - for metadata only
        q: Quantization parameter (alphabet size)
        M: Number of hierarchical levels
        Delta0: Base quantization step size
        eta: Learning rate for scaling factors
        k: Number of scaling factors per row/block
        tiling: Tiling strategy ('row' or 'block')
        block_size: Size of quantization blocks
        quantize_activations: Whether to quantize activations
        act_bit_width: Bit width for activation quantization
        act_init_alpha: Initial alpha value for activation quantization
        init_method: Weight initialization method
        init_kwargs: Additional arguments for weight initialization
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        G: torch.Tensor,
        Ginv: torch.Tensor,
        quantize_fn,
        lattice_type: str = 'E8',
        q: int = 4,
        M: int = 2,
        Delta0: float = 1.5,
        eta: float = 0.1,
        k: int = 1,
        tiling: str = 'row',
        block_size: int = 8,
        quantize_activations: bool = False,
        act_bit_width: int = 8,
        act_init_alpha: float = 1.0,
        init_method: str = 'normal',
        init_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_fn = quantize_fn
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        self.Delta0 = Delta0
        self.eta = eta
        self.k = k
        self.tiling = tiling
        self.block_size = block_size
        self.quantize_activations = quantize_activations
        self.init_method = init_method
        self.init_kwargs = init_kwargs or {}
        
        # Store lattice matrices
        self.register_buffer('G', G)
        self.register_buffer('Ginv', Ginv)
        
        # Compute dimensions
        self.blocks_per_row = in_features // block_size
        if in_features % block_size != 0:
            raise ValueError(f"Input dimension {in_features} must be divisible by block_size {block_size}")
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._initialize_weights()
        
        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        # Initialize scaling factors
        if tiling == 'row':
            self.theta_beta = nn.Parameter(torch.ones(out_features, k))
        elif tiling == 'block':
            self.theta_beta = nn.Parameter(torch.ones(out_features, self.blocks_per_row, k))
        else:
            raise ValueError(f"Unknown tiling: {tiling}")
        
        # Initialize activation quantizer if requested
        if quantize_activations:
            self.actq = LSQActivation(bit_width=act_bit_width, init_alpha=act_init_alpha)
        else:
            self.actq = None
    
    def _initialize_weights(self):
        """Initialize weights using the specified method."""
        if self.init_method == 'normal':
            nn.init.normal_(self.weight, **self.init_kwargs)
        elif self.init_method == 'xavier':
            nn.init.xavier_normal_(self.weight, **self.init_kwargs)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_normal_(self.weight, **self.init_kwargs)
        elif self.init_method == 'zeros':
            nn.init.zeros_(self.weight)
        elif self.init_method == 'ones':
            nn.init.ones_(self.weight)
        elif self.init_method == 'load':
            # Weights will be loaded externally
            pass
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
    
    def load_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Load weights and bias from tensors.
        
        Args:
            weight: Weight tensor of shape [out_features, in_features]
            bias: Optional bias tensor of shape [out_features]
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape {weight.shape} doesn't match expected {(self.out_features, self.in_features)}")
        
        self.weight.data = weight.clone()
        
        if bias is not None:
            if self.bias is None:
                raise ValueError("Cannot load bias: layer was initialized with bias=False")
            if bias.shape != (self.out_features,):
                raise ValueError(f"Bias shape {bias.shape} doesn't match expected {(self.out_features,)}")
            self.bias.data = bias.clone()
    
    def load_from_linear(self, linear_layer: nn.Linear):
        """
        Load weights and bias from an existing nn.Linear layer.
        
        Args:
            linear_layer: nn.Linear layer to copy weights from
        """
        if linear_layer.in_features != self.in_features:
            raise ValueError(f"Input features mismatch: {linear_layer.in_features} vs {self.in_features}")
        if linear_layer.out_features != self.out_features:
            raise ValueError(f"Output features mismatch: {linear_layer.out_features} vs {self.out_features}")
        
        self.load_weights(linear_layer.weight.data, linear_layer.bias.data if linear_layer.bias is not None else None)
    
    def _quantize_weights(self):
        """Quantize weights using hierarchical nested lattice quantization."""
        W = self.weight  # [out_dim, in_dim]
        
        # Reshape to blocks
        W_blocks = W.view(self.out_features, self.blocks_per_row, self.block_size)  # [out_dim, blocks_per_row, block_size]
        
        # Apply scaling
        if self.tiling == 'row':
            # Row-level scaling: each row gets k scaling factors
            theta_beta = self.theta_beta  # [out_dim, k]
            # Broadcast to blocks: [out_dim, 1, k] -> [out_dim, blocks_per_row, k]
            theta_beta_expanded = theta_beta.unsqueeze(1).expand(-1, self.blocks_per_row, -1)
        else:  # block
            # Block-level scaling: each block gets k scaling factors
            theta_beta = self.theta_beta  # [out_dim, blocks_per_row, k]
            theta_beta_expanded = theta_beta
        
        # Apply scaling to each block
        W_scaled = W_blocks * theta_beta_expanded.mean(dim=-1, keepdim=True)  # [out_dim, blocks_per_row, block_size]
        
        # Reshape to blocks for quantization: [out_dim, blocks_per_row, block_size]
        # Then reshape to [total_blocks, block_size] for per-block quantization
        W_blocks_flat = W_scaled.reshape(-1, self.block_size)  # [out_dim * blocks_per_row, block_size]
        
        # Quantize using STE (Straight-Through Estimator)
        # This allows gradients to flow through quantization during training
        W_quantized_blocks = ste_quantize(W_blocks_flat, self.quantize_fn, self.q)
        
        # Reshape back to original weight shape
        W_quantized = W_quantized_blocks.reshape(self.out_features, self.in_features)
        
        return W_quantized
    
    def forward(self, x):
        """Forward pass with optional activation quantization."""
        # Quantize weights
        W_q = self._quantize_weights()
        
        # Linear transformation
        x = F.linear(x, W_q, self.bias)
        
        # Quantize activations if requested
        if self.quantize_activations and self.actq is not None:
            x = self.actq(x)
        
        return x
    
    def export_quantized(self):
        """
        Export quantized weights and metadata for inference.
        
        Returns:
            Dictionary containing quantized weights and configuration
        """
        with torch.no_grad():
            W_q = self._quantize_weights()
            
            metadata = {
                'lattice_type': self.lattice_type,
                'q': self.q,
                'M': self.M,
                'Delta0': self.Delta0,
                'tiling': self.tiling,
                'block_size': self.block_size,
                'quantize_activations': self.quantize_activations,
            }
            
            if self.quantize_activations and self.actq is not None:
                metadata.update({
                    'act_bit_width': self.actq.bit_width,
                    'act_alpha': self.actq.alpha.item(),
                })
            
            return {
                'weight': W_q,
                'bias': self.bias,
                'metadata': metadata
            }
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'lattice_type={self.lattice_type}, q={self.q}, M={self.M}, ' \
               f'tiling={self.tiling}, block_size={self.block_size}, ' \
               f'quantize_activations={self.quantize_activations}, bias={self.bias is not None}'
