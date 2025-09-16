"""
Quantized NCCL communicator for distributed training

This module implements efficient communication primitives for quantized
gradients using NCCL, enabling in-place operations in the quantized space.
"""

import torch
import torch.distributed as dist
from typing import Optional, List, Dict, Any
import time

from ..quantizers.config import LatticeConfig
from ..quantizers.radixq import QuantizedGradientCompressor


class QuantizedNCCLCommunicator:
    """
    NCCL communicator for quantized gradient operations.
    
    This class provides efficient communication primitives for quantized
    gradients, enabling in-place operations in the quantized space.
    """
    
    def __init__(
        self,
        config: LatticeConfig,
        world_size: int,
        rank: int,
        communication_depth: int = 1,
        compression_enabled: bool = True
    ):
        """
        Initialize quantized NCCL communicator.
        
        Args:
            config: Lattice quantization configuration
            world_size: Number of processes in distributed training
            rank: Rank of current process
            communication_depth: Default communication depth
            compression_enabled: Whether to enable gradient compression
        """
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.communication_depth = communication_depth
        self.compression_enabled = compression_enabled
        
        # Initialize gradient compressor
        self.compressor = QuantizedGradientCompressor(config)
        self.compressor.set_communication_depth(communication_depth)
        self.compressor.enable_compression(compression_enabled)
        
        # Communication statistics
        self.comm_stats = {
            'allreduce_calls': 0,
            'allgather_calls': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'total_communication_time': 0.0
        }
    
    def quantized_allreduce(
        self,
        tensor: torch.Tensor,
        depth: Optional[int] = None,
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """
        Perform quantized all-reduce operation.
        
        Args:
            tensor: Tensor to reduce
            depth: Quantization depth (uses default if None)
            op: Reduction operation
            
        Returns:
            reduced_tensor: Reduced tensor
        """
        if depth is None:
            depth = self.communication_depth
        
        start_time = time.time()
        
        # Quantize tensor
        quantized_tensor = self.compressor.compress_gradients(tensor, depth)
        
        # Perform all-reduce in quantized space
        dist.all_reduce(quantized_tensor, op=op)
        
        # Dequantize result
        result = self.compressor.decompress_gradients(quantized_tensor, depth)
        
        # Update statistics
        self._update_comm_stats(tensor, time.time() - start_time)
        self.comm_stats['allreduce_calls'] += 1
        
        return result
    
    def quantized_allgather(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        depth: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform quantized all-gather operation.
        
        Args:
            input_tensor: Input tensor to gather
            output_tensor: Output tensor to store gathered results
            depth: Quantization depth (uses default if None)
            
        Returns:
            output_tensor: Tensor containing gathered results
        """
        if depth is None:
            depth = self.communication_depth
        
        start_time = time.time()
        
        # Quantize input tensor
        quantized_input = self.compressor.compress_gradients(input_tensor, depth)
        
        # Create quantized output tensor
        quantized_output = torch.zeros_like(output_tensor, dtype=quantized_input.dtype)
        
        # Perform all-gather in quantized space
        dist.all_gather_into_tensor(quantized_output, quantized_input)
        
        # Dequantize result
        result = self.compressor.decompress_gradients(quantized_output, depth)
        output_tensor.copy_(result)
        
        # Update statistics
        self._update_comm_stats(input_tensor, time.time() - start_time)
        self.comm_stats['allgather_calls'] += 1
        
        return output_tensor
    
    def quantized_reduce_scatter(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        depth: Optional[int] = None,
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """
        Perform quantized reduce-scatter operation.
        
        Args:
            input_tensor: Input tensor to reduce and scatter
            output_tensor: Output tensor to store result
            depth: Quantization depth (uses default if None)
            op: Reduction operation
            
        Returns:
            output_tensor: Tensor containing reduced and scattered result
        """
        if depth is None:
            depth = self.communication_depth
        
        start_time = time.time()
        
        # Quantize input tensor
        quantized_input = self.compressor.compress_gradients(input_tensor, depth)
        
        # Create quantized output tensor
        quantized_output = torch.zeros_like(output_tensor, dtype=quantized_input.dtype)
        
        # Perform reduce-scatter in quantized space
        dist.reduce_scatter_tensor(quantized_output, quantized_input, op=op)
        
        # Dequantize result
        result = self.compressor.decompress_gradients(quantized_output, depth)
        output_tensor.copy_(result)
        
        # Update statistics
        self._update_comm_stats(input_tensor, time.time() - start_time)
        
        return output_tensor
    
    def quantized_broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        depth: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform quantized broadcast operation.
        
        Args:
            tensor: Tensor to broadcast
            src: Source rank
            depth: Quantization depth (uses default if None)
            
        Returns:
            tensor: Broadcasted tensor
        """
        if depth is None:
            depth = self.communication_depth
        
        start_time = time.time()
        
        # Quantize tensor
        quantized_tensor = self.compressor.compress_gradients(tensor, depth)
        
        # Perform broadcast in quantized space
        dist.broadcast(quantized_tensor, src=src)
        
        # Dequantize result
        result = self.compressor.decompress_gradients(quantized_tensor, depth)
        tensor.copy_(result)
        
        # Update statistics
        self._update_comm_stats(tensor, time.time() - start_time)
        
        return tensor
    
    def _update_comm_stats(self, tensor: torch.Tensor, comm_time: float):
        """Update communication statistics."""
        bytes_transferred = tensor.numel() * tensor.element_size()
        self.comm_stats['total_bytes_sent'] += bytes_transferred
        self.comm_stats['total_bytes_received'] += bytes_transferred
        self.comm_stats['total_communication_time'] += comm_time
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        stats = self.comm_stats.copy()
        
        # Add compression statistics
        compression_stats = self.compressor.get_compression_stats()
        stats.update(compression_stats)
        
        # Compute derived statistics
        if stats['total_communication_time'] > 0:
            stats['avg_communication_time'] = stats['total_communication_time'] / max(
                stats['allreduce_calls'] + stats['allgather_calls'], 1
            )
            stats['bandwidth_mbps'] = (
                stats['total_bytes_sent'] / stats['total_communication_time'] / 1e6
            )
        
        return stats
    
    def reset_communication_stats(self):
        """Reset communication statistics."""
        self.comm_stats = {
            'allreduce_calls': 0,
            'allgather_calls': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'total_communication_time': 0.0
        }
    
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


class QuantizedGradientAccumulator:
    """
    Gradient accumulator for quantized gradients.
    
    This class enables efficient accumulation of quantized gradients
    across multiple iterations before communication.
    """
    
    def __init__(
        self,
        config: LatticeConfig,
        accumulation_steps: int = 1,
        communication_depth: int = 1
    ):
        """
        Initialize quantized gradient accumulator.
        
        Args:
            config: Lattice quantization configuration
            accumulation_steps: Number of steps to accumulate before communication
            communication_depth: Communication depth
        """
        self.config = config
        self.accumulation_steps = accumulation_steps
        self.communication_depth = communication_depth
        
        # Initialize compressor
        self.compressor = QuantizedGradientCompressor(config)
        self.compressor.set_communication_depth(communication_depth)
        
        # Accumulation state
        self.accumulated_gradients = None
        self.step_count = 0
    
    def accumulate_gradients(self, gradients: torch.Tensor) -> bool:
        """
        Accumulate gradients.
        
        Args:
            gradients: Gradient tensor to accumulate
            
        Returns:
            should_communicate: Whether to perform communication
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = gradients.clone()
        else:
            self.accumulated_gradients += gradients
        
        self.step_count += 1
        
        return self.step_count >= self.accumulation_steps
    
    def get_accumulated_gradients(self) -> torch.Tensor:
        """Get accumulated gradients."""
        if self.accumulated_gradients is None:
            raise RuntimeError("No gradients accumulated")
        return self.accumulated_gradients
    
    def reset_accumulation(self):
        """Reset gradient accumulation."""
        self.accumulated_gradients = None
        self.step_count = 0
    
    def compress_accumulated_gradients(self) -> torch.Tensor:
        """Compress accumulated gradients for communication."""
        if self.accumulated_gradients is None:
            raise RuntimeError("No gradients accumulated")
        
        return self.compressor.compress_gradients(
            self.accumulated_gradients, self.communication_depth
        )
    
    def decompress_gradients(self, compressed_gradients: torch.Tensor) -> torch.Tensor:
        """Decompress gradients after communication."""
        return self.compressor.decompress_gradients(
            compressed_gradients, self.communication_depth
        )
