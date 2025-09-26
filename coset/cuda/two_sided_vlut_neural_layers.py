"""
Neural Network Layers with Optimized Two-Sided vLUT Operations.

This module provides neural network layer implementations that use optimized
two-sided vLUT operations where both input and weight tensors are quantized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from two_sided_vlut_operations import (
    OptimizedTwoSidedVLUTOperations,
    TwoSidedVLUTConfig,
    create_optimized_two_sided_vlut_operations
)


class OptimizedTwoSidedVLUTLinear(nn.Module):
    """Optimized linear layer with two-sided vLUT operations."""
    
    def __init__(self, in_features: int, out_features: int, lattice, config, 
                 bias: bool = True, vlut_config: TwoSidedVLUTConfig = None):
        super(OptimizedTwoSidedVLUTLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lattice = lattice
        self.config = config
        self.vlut_config = vlut_config or TwoSidedVLUTConfig()
        
        # Initialize weight encodings (will be set during forward pass)
        self.register_buffer('weight_encodings', torch.zeros(out_features, in_features, lattice.d, dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize vLUT operations
        self.vlut_operations = create_optimized_two_sided_vlut_operations(
            lattice, config, self.vlut_config.use_cuda
        )
        
        # Performance tracking
        self.performance_stats = {
            'forward_calls': 0,
            'total_time': 0.0,
            'avg_time_per_call': 0.0
        }
    
    def forward(self, input_encodings: torch.Tensor) -> torch.Tensor:
        """Forward pass with two-sided vLUT operations."""
        import time
        start_time = time.time()
        
        # Ensure weight encodings are on the same device as input
        weight_encodings = self.weight_encodings.to(input_encodings.device)
        
        # Perform matrix multiplication using two-sided vLUT operations
        output = self.vlut_operations.matrix_multiply(input_encodings, weight_encodings)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Update performance stats
        time_taken = time.time() - start_time
        self.performance_stats['forward_calls'] += 1
        self.performance_stats['total_time'] += time_taken
        self.performance_stats['avg_time_per_call'] = (
            self.performance_stats['total_time'] / self.performance_stats['forward_calls']
        )
        
        return output
    
    def set_weight_encodings(self, weight_encodings: torch.Tensor):
        """Set the weight encodings for the layer."""
        if weight_encodings.shape != (self.out_features, self.in_features, self.lattice.d):
            raise ValueError(f"Expected weight encodings shape ({self.out_features}, {self.in_features}, {self.lattice.d}), "
                           f"got {weight_encodings.shape}")
        
        self.weight_encodings.data = weight_encodings.data
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return self.performance_stats.copy()


class OptimizedTwoSidedVLUTConv2d(nn.Module):
    """Optimized 2D convolution layer with two-sided vLUT operations."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1,
                 bias: bool = True, lattice=None, config=None, vlut_config: TwoSidedVLUTConfig = None):
        super(OptimizedTwoSidedVLUTConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.lattice = lattice
        self.config = config
        self.vlut_config = vlut_config or TwoSidedVLUTConfig()
        
        # Calculate output dimensions
        self.kernel_elements = self.kernel_size[0] * self.kernel_size[1]
        
        # Initialize weight encodings
        self.register_buffer('weight_encodings', 
                           torch.zeros(out_channels, in_channels // groups, self.kernel_elements, lattice.d, dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize vLUT operations
        self.vlut_operations = create_optimized_two_sided_vlut_operations(
            lattice, config, self.vlut_config.use_cuda
        )
        
        # Performance tracking
        self.performance_stats = {
            'forward_calls': 0,
            'total_time': 0.0,
            'avg_time_per_call': 0.0
        }
    
    def forward(self, input_encodings: torch.Tensor) -> torch.Tensor:
        """Forward pass with two-sided vLUT operations."""
        import time
        start_time = time.time()
        
        batch_size, in_channels, height, width = input_encodings.shape[:4]
        
        # Unfold input to get patches
        input_unfolded = F.unfold(input_encodings.view(batch_size, in_channels, height, width),
                                 kernel_size=self.kernel_size, stride=self.stride, 
                                 padding=self.padding, dilation=self.dilation)
        
        # Reshape for vLUT operations
        input_unfolded = input_unfolded.transpose(1, 2).contiguous()  # [batch_size, num_patches, in_channels * kernel_elements]
        
        # Reshape to match expected format for vLUT operations
        input_reshaped = input_unfolded.view(batch_size, -1, self.lattice.d)
        
        # Ensure weight encodings are on the same device
        weight_encodings = self.weight_encodings.to(input_encodings.device)
        weight_reshaped = weight_encodings.view(self.out_channels, -1, self.lattice.d)
        
        # Perform matrix multiplication using two-sided vLUT operations
        output = self.vlut_operations.matrix_multiply(input_reshaped, weight_reshaped)
        
        # Reshape output back to spatial dimensions
        output_height = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        output = output.view(batch_size, self.out_channels, output_height, output_width)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        # Update performance stats
        time_taken = time.time() - start_time
        self.performance_stats['forward_calls'] += 1
        self.performance_stats['total_time'] += time_taken
        self.performance_stats['avg_time_per_call'] = (
            self.performance_stats['total_time'] / self.performance_stats['forward_calls']
        )
        
        return output
    
    def set_weight_encodings(self, weight_encodings: torch.Tensor):
        """Set the weight encodings for the layer."""
        expected_shape = (self.out_channels, self.in_channels // self.groups, self.kernel_elements, self.lattice.d)
        if weight_encodings.shape != expected_shape:
            raise ValueError(f"Expected weight encodings shape {expected_shape}, got {weight_encodings.shape}")
        
        self.weight_encodings.data = weight_encodings.data
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return self.performance_stats.copy()


class OptimizedTwoSidedVLUTAttention(nn.Module):
    """Optimized attention mechanism with two-sided vLUT operations."""
    
    def __init__(self, embed_dim: int, num_heads: int, lattice, config,
                 dropout: float = 0.0, bias: bool = True, vlut_config: TwoSidedVLUTConfig = None):
        super(OptimizedTwoSidedVLUTAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lattice = lattice
        self.config = config
        self.vlut_config = vlut_config or TwoSidedVLUTConfig()
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Initialize weight encodings for Q, K, V projections
        self.register_buffer('q_weight_encodings', torch.zeros(embed_dim, embed_dim, lattice.d, dtype=torch.float32))
        self.register_buffer('k_weight_encodings', torch.zeros(embed_dim, embed_dim, lattice.d, dtype=torch.float32))
        self.register_buffer('v_weight_encodings', torch.zeros(embed_dim, embed_dim, lattice.d, dtype=torch.float32))
        self.register_buffer('out_weight_encodings', torch.zeros(embed_dim, embed_dim, lattice.d, dtype=torch.float32))
        
        if bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dim))
            self.k_bias = nn.Parameter(torch.zeros(embed_dim))
            self.v_bias = nn.Parameter(torch.zeros(embed_dim))
            self.out_bias = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('k_bias', None)
            self.register_parameter('v_bias', None)
            self.register_parameter('out_bias', None)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize vLUT operations
        self.vlut_operations = create_optimized_two_sided_vlut_operations(
            lattice, config, self.vlut_config.use_cuda
        )
        
        # Performance tracking
        self.performance_stats = {
            'forward_calls': 0,
            'total_time': 0.0,
            'avg_time_per_call': 0.0
        }
    
    def forward(self, input_encodings: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with two-sided vLUT operations."""
        import time
        start_time = time.time()
        
        batch_size, seq_len, embed_dim = input_encodings.shape[:3]
        
        # Ensure weight encodings are on the same device
        q_weight_encodings = self.q_weight_encodings.to(input_encodings.device)
        k_weight_encodings = self.k_weight_encodings.to(input_encodings.device)
        v_weight_encodings = self.v_weight_encodings.to(input_encodings.device)
        out_weight_encodings = self.out_weight_encodings.to(input_encodings.device)
        
        # Reshape input for vLUT operations
        input_reshaped = input_encodings.view(batch_size * seq_len, embed_dim, self.lattice.d)
        
        # Compute Q, K, V using two-sided vLUT operations
        q = self.vlut_operations.matrix_multiply(input_reshaped, q_weight_encodings)
        k = self.vlut_operations.matrix_multiply(input_reshaped, k_weight_encodings)
        v = self.vlut_operations.matrix_multiply(input_reshaped, v_weight_encodings)
        
        # Add biases
        if self.q_bias is not None:
            q = q + self.q_bias
        if self.k_bias is not None:
            k = k + self.k_bias
        if self.v_bias is not None:
            v = v + self.v_bias
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Apply output projection using two-sided vLUT operations
        output_reshaped = attn_output.view(batch_size * seq_len, embed_dim, self.lattice.d)
        output = self.vlut_operations.matrix_multiply(output_reshaped, out_weight_encodings)
        output = output.view(batch_size, seq_len, embed_dim)
        
        # Add output bias
        if self.out_bias is not None:
            output = output + self.out_bias
        
        # Update performance stats
        time_taken = time.time() - start_time
        self.performance_stats['forward_calls'] += 1
        self.performance_stats['total_time'] += time_taken
        self.performance_stats['avg_time_per_call'] = (
            self.performance_stats['total_time'] / self.performance_stats['forward_calls']
        )
        
        return output, attn_weights
    
    def set_weight_encodings(self, q_encodings: torch.Tensor, k_encodings: torch.Tensor,
                           v_encodings: torch.Tensor, out_encodings: torch.Tensor):
        """Set the weight encodings for all projections."""
        expected_shape = (self.embed_dim, self.embed_dim, self.lattice.d)
        
        if q_encodings.shape != expected_shape:
            raise ValueError(f"Expected Q weight encodings shape {expected_shape}, got {q_encodings.shape}")
        if k_encodings.shape != expected_shape:
            raise ValueError(f"Expected K weight encodings shape {expected_shape}, got {k_encodings.shape}")
        if v_encodings.shape != expected_shape:
            raise ValueError(f"Expected V weight encodings shape {expected_shape}, got {v_encodings.shape}")
        if out_encodings.shape != expected_shape:
            raise ValueError(f"Expected output weight encodings shape {expected_shape}, got {out_encodings.shape}")
        
        self.q_weight_encodings.data = q_encodings.data
        self.k_weight_encodings.data = k_encodings.data
        self.v_weight_encodings.data = v_encodings.data
        self.out_weight_encodings.data = out_encodings.data
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return self.performance_stats.copy()


# Utility functions for creating neural networks with two-sided vLUT operations
def create_two_sided_vlut_mlp(input_dim: int, hidden_dims: list, output_dim: int,
                             lattice, config, vlut_config: TwoSidedVLUTConfig = None) -> nn.Module:
    """Create a multi-layer perceptron with two-sided vLUT operations."""
    layers = []
    
    # Input layer
    layers.append(OptimizedTwoSidedVLUTLinear(input_dim, hidden_dims[0], lattice, config, vlut_config))
    layers.append(nn.ReLU())
    
    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(OptimizedTwoSidedVLUTLinear(hidden_dims[i], hidden_dims[i + 1], lattice, config, vlut_config))
        layers.append(nn.ReLU())
    
    # Output layer
    layers.append(OptimizedTwoSidedVLUTLinear(hidden_dims[-1], output_dim, lattice, config, vlut_config))
    
    return nn.Sequential(*layers)


def create_two_sided_vlut_cnn(input_channels: int, num_classes: int, lattice, config,
                             vlut_config: TwoSidedVLUTConfig = None) -> nn.Module:
    """Create a convolutional neural network with two-sided vLUT operations."""
    return nn.Sequential(
        OptimizedTwoSidedVLUTConv2d(input_channels, 32, 3, padding=1, lattice=lattice, config=config, vlut_config=vlut_config),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        OptimizedTwoSidedVLUTConv2d(32, 64, 3, padding=1, lattice=lattice, config=config, vlut_config=vlut_config),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        OptimizedTwoSidedVLUTConv2d(64, 128, 3, padding=1, lattice=lattice, config=config, vlut_config=vlut_config),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        
        nn.Flatten(),
        OptimizedTwoSidedVLUTLinear(128, num_classes, lattice=lattice, config=config, vlut_config=vlut_config)
    )
