"""
Quantized gradient hooks for distributed training

This module implements hooks for quantized gradient communication
during distributed training, enabling efficient all-reduce operations
in the quantized space.
"""

import torch
import torch.distributed as dist
from typing import Optional, Callable, Any
import time

from ..quantizers.config import LatticeConfig
from ..quantizers.radixq import QuantizedGradientCompressor


class QuantizedGradientHook:
    """
    Hook for quantized gradient communication in distributed training.
    
    This hook enables efficient gradient communication by:
    1. Quantizing gradients before communication
    2. Performing all-reduce operations in quantized space
    3. Dequantizing gradients after communication
    4. Supporting different quantization depths for different layers
    """
    
    def __init__(
        self,
        config: LatticeConfig,
        communication_depth: int = 1,
        compression_enabled: bool = True,
        timing_enabled: bool = False
    ):
        """
        Initialize quantized gradient hook.
        
        Args:
            config: Lattice quantization configuration
            communication_depth: Default depth for gradient communication
            compression_enabled: Whether to enable gradient compression
            timing_enabled: Whether to enable timing measurements
        """
        self.config = config
        self.communication_depth = communication_depth
        self.compression_enabled = compression_enabled
        self.timing_enabled = timing_enabled
        
        # Initialize gradient compressor
        self.compressor = QuantizedGradientCompressor(config)
        self.compressor.set_communication_depth(communication_depth)
        self.compressor.enable_compression(compression_enabled)
        
        # Timing statistics
        self.timing_stats = {
            'quantization_time': 0.0,
            'communication_time': 0.0,
            'dequantization_time': 0.0,
            'total_time': 0.0,
            'num_calls': 0
        }
    
    def __call__(self, state: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        """
        Hook function called during distributed training.
        
        Args:
            state: Hook state
            bucket: Gradient bucket containing gradients to communicate
            
        Returns:
            future: Future containing processed gradients
        """
        return self._quantized_allreduce(state, bucket)
    
    def _quantized_allreduce(
        self, 
        state: Any, 
        bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        """
        Perform quantized all-reduce operation.
        
        Args:
            state: Hook state
            bucket: Gradient bucket
            
        Returns:
            future: Future containing processed gradients
        """
        start_time = time.time() if self.timing_enabled else 0
        
        # Get gradients from bucket
        gradients = bucket.gradients()
        
        # Quantize gradients
        quantize_start = time.time() if self.timing_enabled else 0
        quantized_gradients = self.compressor.compress_gradients(gradients, self.communication_depth)
        if self.timing_enabled:
            self.timing_stats['quantization_time'] += time.time() - quantize_start
        
        # Perform all-reduce in quantized space
        comm_start = time.time() if self.timing_enabled else 0
        dist.all_reduce(quantized_gradients, op=dist.ReduceOp.SUM)
        if self.timing_enabled:
            self.timing_stats['communication_time'] += time.time() - comm_start
        
        # Dequantize gradients
        dequantize_start = time.time() if self.timing_enabled else 0
        processed_gradients = self.compressor.decompress_gradients(quantized_gradients, self.communication_depth)
        if self.timing_enabled:
            self.timing_stats['dequantization_time'] += time.time() - dequantize_start
        
        # Update timing statistics
        if self.timing_enabled:
            total_time = time.time() - start_time
            self.timing_stats['total_time'] += total_time
            self.timing_stats['num_calls'] += 1
        
        # Create future with processed gradients
        future = torch.futures.Future()
        future.set_result(processed_gradients)
        
        return future
    
    def set_communication_depth(self, depth: int):
        """Set communication depth for gradient quantization."""
        if depth <= 0 or depth > self.config.num_layers:
            raise ValueError(f"Depth {depth} must be between 1 and {self.config.num_layers}")
        self.communication_depth = depth
        self.compressor.set_communication_depth(depth)
    
    def enable_compression(self, enabled: bool = True):
        """Enable or disable gradient compression."""
        self.compression_enabled = enabled
        self.compressor.enable_compression(enabled)
    
    def enable_timing(self, enabled: bool = True):
        """Enable or disable timing measurements."""
        self.timing_enabled = enabled
        if not enabled:
            self.timing_stats = {
                'quantization_time': 0.0,
                'communication_time': 0.0,
                'dequantization_time': 0.0,
                'total_time': 0.0,
                'num_calls': 0
            }
    
    def get_timing_stats(self) -> dict:
        """Get timing statistics."""
        if not self.timing_enabled or self.timing_stats['num_calls'] == 0:
            return self.timing_stats
        
        # Compute averages
        num_calls = self.timing_stats['num_calls']
        avg_stats = {}
        for key, value in self.timing_stats.items():
            if key != 'num_calls':
                avg_stats[f'avg_{key}'] = value / num_calls
        
        return {**self.timing_stats, **avg_stats}
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        return self.compressor.get_compression_stats()
    
    def reset_timing_stats(self):
        """Reset timing statistics."""
        self.timing_stats = {
            'quantization_time': 0.0,
            'communication_time': 0.0,
            'dequantization_time': 0.0,
            'total_time': 0.0,
            'num_calls': 0
        }


class AdaptiveQuantizedGradientHook(QuantizedGradientHook):
    """
    Adaptive quantized gradient hook that automatically selects
    quantization depth based on gradient statistics.
    """
    
    def __init__(
        self,
        config: LatticeConfig,
        initial_depth: int = 1,
        compression_enabled: bool = True,
        timing_enabled: bool = False,
        adaptation_interval: int = 100
    ):
        """
        Initialize adaptive quantized gradient hook.
        
        Args:
            config: Lattice quantization configuration
            initial_depth: Initial communication depth
            compression_enabled: Whether to enable gradient compression
            timing_enabled: Whether to enable timing measurements
            adaptation_interval: Interval for depth adaptation
        """
        super().__init__(config, initial_depth, compression_enabled, timing_enabled)
        self.adaptation_interval = adaptation_interval
        self.call_count = 0
        self.gradient_stats = []
    
    def _quantized_allreduce(
        self, 
        state: Any, 
        bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        """
        Perform adaptive quantized all-reduce operation.
        
        Args:
            state: Hook state
            bucket: Gradient bucket
            
        Returns:
            future: Future containing processed gradients
        """
        # Get gradients from bucket
        gradients = bucket.gradients()
        
        # Collect gradient statistics for adaptation
        if self.call_count % self.adaptation_interval == 0:
            self._adapt_communication_depth(gradients)
        
        self.call_count += 1
        
        # Perform standard quantized all-reduce
        return super()._quantized_allreduce(state, bucket)
    
    def _adapt_communication_depth(self, gradients: torch.Tensor):
        """
        Adapt communication depth based on gradient statistics.
        
        Args:
            gradients: Gradient tensor
        """
        # Compute gradient statistics
        grad_std = torch.std(gradients).item()
        grad_mean = torch.mean(torch.abs(gradients)).item()
        
        # Store statistics
        self.gradient_stats.append((grad_std, grad_mean))
        
        # Keep only recent statistics
        if len(self.gradient_stats) > 10:
            self.gradient_stats.pop(0)
        
        # Adapt depth based on statistics
        if len(self.gradient_stats) >= 5:
            avg_std = sum(stat[0] for stat in self.gradient_stats) / len(self.gradient_stats)
            avg_mean = sum(stat[1] for stat in self.gradient_stats) / len(self.gradient_stats)
            
            # Simple heuristic for depth selection
            if avg_std > 1.0:
                new_depth = min(3, self.config.num_layers)
            elif avg_std > 0.1:
                new_depth = 2
            else:
                new_depth = 1
            
            if new_depth != self.communication_depth:
                self.set_communication_depth(new_depth)
                print(f"Adapted communication depth to {new_depth} (std: {avg_std:.4f}, mean: {avg_mean:.4f})")


def register_quantized_hook(
    model: torch.nn.Module,
    config: LatticeConfig,
    communication_depth: int = 1,
    compression_enabled: bool = True,
    adaptive: bool = False
) -> QuantizedGradientHook:
    """
    Register quantized gradient hook for distributed training.
    
    Args:
        model: PyTorch model
        config: Lattice quantization configuration
        communication_depth: Communication depth
        compression_enabled: Whether to enable compression
        adaptive: Whether to use adaptive depth selection
        
    Returns:
        hook: Registered quantized gradient hook
    """
    if adaptive:
        hook = AdaptiveQuantizedGradientHook(
            config, communication_depth, compression_enabled
        )
    else:
        hook = QuantizedGradientHook(
            config, communication_depth, compression_enabled
        )
    
    # Register hook
    model.register_comm_hook(None, hook)
    
    return hook
