"""
Comprehensive tests for CoSet quantization operations
"""

import pytest
import torch
import numpy as np
from typing import Tuple

from coset import LatticeConfig, LatticeType, LatticeQuantizer, QuantizedLinear, QuantizedMLP
from coset.quantizers import RadixQEncoder
from coset.distributed import QuantizedGradientHook


class TestLatticeConfig:
    """Test lattice configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = LatticeConfig()
        assert config.type == LatticeType.HNLQ
        assert config.radix == 4
        assert config.num_layers == 3
        assert config.lattice_dim == 8
        assert len(config.scales) == 3
        assert len(config.zero_points) == 3
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = LatticeConfig(
            type=LatticeType.E8,
            radix=8,
            num_layers=5,
            lattice_dim=16,
            scales=[1.0, 2.0, 4.0, 8.0, 16.0],
            zero_points=[0, 0, 0, 0, 0]
        )
        assert config.type == LatticeType.E8
        assert config.radix == 8
        assert config.num_layers == 5
        assert config.lattice_dim == 16
        assert config.scales == [1.0, 2.0, 4.0, 8.0, 16.0]
        assert config.zero_points == [0, 0, 0, 0, 0]
    
    def test_invalid_config(self):
        """Test invalid configuration handling."""
        with pytest.raises(ValueError):
            LatticeConfig(radix=1)  # Invalid radix
        
        with pytest.raises(ValueError):
            LatticeConfig(num_layers=0)  # Invalid num_layers
        
        with pytest.raises(ValueError):
            LatticeConfig(lattice_dim=0)  # Invalid lattice_dim
        
        with pytest.raises(ValueError):
            LatticeConfig(scales=[1.0, 2.0])  # Mismatched scales length
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = LatticeConfig(
            type=LatticeType.A2,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        assert config_dict['type'] == 'a2'
        assert config_dict['radix'] == 4
        assert config_dict['num_layers'] == 3
        assert config_dict['lattice_dim'] == 8
        
        # Convert back from dict
        config_restored = LatticeConfig.from_dict(config_dict)
        assert config_restored.type == LatticeType.A2
        assert config_restored.radix == 4
        assert config_restored.num_layers == 3
        assert config_restored.lattice_dim == 8


class TestLatticeQuantizer:
    """Test lattice quantizer operations."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
    
    @pytest.fixture
    def quantizer(self, config):
        """Create test quantizer."""
        return LatticeQuantizer(config)
    
    def test_quantizer_creation(self, config):
        """Test quantizer creation."""
        quantizer = LatticeQuantizer(config)
        assert quantizer.config == config
        assert quantizer.lattice_dim == 8
        assert quantizer.num_layers == 3
        assert quantizer.radix == 4
    
    def test_single_level_quantization(self, quantizer):
        """Test single-level quantization."""
        input_tensor = torch.randn(32, 8)
        
        # Test quantization at specific depth
        quantized, indices = quantizer.quantize(input_tensor, depth=1)
        
        assert quantized.shape == input_tensor.shape
        assert indices.shape == input_tensor.shape
        assert indices.dtype == torch.long
        assert indices.min() >= 0
        assert indices.max() < 2 ** quantizer.lattice_dim
    
    def test_adaptive_quantization(self, quantizer):
        """Test adaptive quantization."""
        input_tensor = torch.randn(32, 8)
        
        # Test adaptive quantization
        quantized, indices = quantizer.quantize(input_tensor, depth=-1)
        
        assert quantized.shape == input_tensor.shape
        assert indices.shape == (32, 3)  # num_layers
        assert indices.dtype == torch.long
    
    def test_dequantization(self, quantizer):
        """Test dequantization."""
        input_tensor = torch.randn(32, 8)
        
        # Quantize and decode
        quantized, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
        reconstructed = quantizer.decode_from_depth(indices, source_depth=1)
        
        assert reconstructed.shape == input_tensor.shape
        assert reconstructed.dtype == torch.float32
        
        # Check that reconstruction is close to original
        error = torch.mean(torch.abs(input_tensor - reconstructed))
        assert error < 1.0  # Should be reasonably close
    
    def test_hierarchical_quantization(self, quantizer):
        """Test hierarchical quantization."""
        input_tensor = torch.randn(32, 8)
        
        # Test hierarchical quantization
        quantized = quantizer.quantize(input_tensor)
        quantized_depth, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
        
        assert quantized.shape == input_tensor.shape
        assert indices.shape == (32,)  # Single depth
        assert quantized_depth.shape == input_tensor.shape
    
    def test_packing_encoding(self, quantizer):
        """Test packing encoding/decoding."""
        input_tensor = torch.randn(32, 8)
        
        # Test packing encoding
        encoded = quantizer.packing_encode(input_tensor, packing_radix=4, depth=2)
        decoded = quantizer.packing_decode(encoded, packing_radix=4, depth=2)
        
        assert encoded.shape == input_tensor.shape
        assert encoded.dtype == torch.int32
        assert decoded.shape == input_tensor.shape
        assert decoded.dtype == torch.float32
    
    def test_lookup_table_operations(self, quantizer):
        """Test lookup table operations."""
        # Create test indices
        x_indices = torch.randint(0, 256, (32,))
        y_indices = torch.randint(0, 256, (32,))
        
        # Test lookup dot product
        dot_products = quantizer.lookup_dot_product(x_indices, y_indices)
        
        assert dot_products.shape == (32,)
        assert dot_products.dtype == torch.float32
    
    def test_quantized_vector_operations(self, quantizer):
        """Test quantized vector operations."""
        x_indices = torch.randint(0, 256, (32,))
        y_indices = torch.randint(0, 256, (32,))
        
        # Test quantized addition
        sum_indices = quantizer.quantized_add(x_indices, y_indices)
        assert sum_indices.shape == x_indices.shape
        
        # Test quantized reduction
        indices = torch.randint(0, 256, (32, 8))
        reduced_indices = quantizer.quantized_reduce(indices, dim=1)
        assert reduced_indices.shape == (32, 1)
    
    def test_quantization_stats(self, quantizer):
        """Test quantization statistics."""
        stats = quantizer.get_quantization_stats()
        
        assert 'scales' in stats
        assert 'zero_points' in stats
        assert 'hierarchy_weights' in stats
        assert 'num_codewords' in stats
        
        assert stats['scales'].shape == (3,)
        assert stats['zero_points'].shape == (3,)
        assert stats['hierarchy_weights'].shape == (3,)
        assert stats['num_codewords'] == 256  # 2^8


class TestQuantizedLinear:
    """Test quantized linear layer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
    
    @pytest.fixture
    def layer(self, config):
        """Create test quantized linear layer."""
        return QuantizedLinear(
            in_features=512,
            out_features=256,
            config=config,
            bias=True,
            use_ste=True,
            use_lookup_tables=True
        )
    
    def test_layer_creation(self, config):
        """Test layer creation."""
        layer = QuantizedLinear(512, 256, config)
        
        assert layer.in_features == 512
        assert layer.out_features == 256
        assert layer.config == config
        assert layer.weight.shape == (256, 512)
        assert layer.bias.shape == (256,)
    
    def test_forward_pass(self, layer):
        """Test forward pass."""
        input_tensor = torch.randn(32, 512)
        
        # Test forward pass
        output = layer(input_tensor)
        
        assert output.shape == (32, 256)
        assert output.dtype == torch.float32
    
    def test_forward_pass_with_depth(self, layer):
        """Test forward pass with specific depth."""
        input_tensor = torch.randn(32, 512)
        
        # Test forward pass with specific depth
        output = layer(input_tensor, depth=1)
        
        assert output.shape == (32, 256)
        assert output.dtype == torch.float32
    
    def test_quantized_weights(self, layer):
        """Test quantized weight retrieval."""
        quantized_weights = layer.get_quantized_weights(depth=1)
        
        assert quantized_weights.shape == layer.weight.shape
        assert quantized_weights.dtype == torch.float32
    
    def test_quantization_stats(self, layer):
        """Test quantization statistics."""
        stats = layer.get_quantization_stats()
        
        assert 'weight_shape' in stats
        assert 'bias_shape' in stats
        assert 'quantization_error' in stats
        assert 'compression_stats' in stats
        
        assert stats['weight_shape'] == (256, 512)
        assert stats['bias_shape'] == (256,)
        assert stats['quantization_error'] >= 0.0
    
    def test_gradient_compression(self, layer):
        """Test gradient compression."""
        # Create mock gradients
        layer.weight.grad = torch.randn(256, 512)
        
        # Test gradient compression
        quantized_gradients = layer.get_quantized_gradients(depth=1)
        
        assert quantized_gradients is not None
        assert quantized_gradients.shape == layer.weight.grad.shape
        assert quantized_gradients.dtype == torch.int32
    
    def test_gradient_decompression(self, layer):
        """Test gradient decompression."""
        # Create mock gradients
        original_gradients = torch.randn(256, 512)
        layer.weight.grad = original_gradients.clone()
        
        # Compress and decompress gradients
        quantized_gradients = layer.get_quantized_gradients(depth=1)
        layer.set_quantized_gradients(quantized_gradients, depth=1)
        
        # Check that gradients are approximately preserved
        error = torch.mean(torch.abs(original_gradients - layer.weight.grad))
        assert error < 1.0  # Should be reasonably close


class TestRadixQEncoder:
    """Test radix-q encoder."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
    
    @pytest.fixture
    def encoder(self, config):
        """Create test encoder."""
        return RadixQEncoder(config)
    
    def test_encoder_creation(self, config):
        """Test encoder creation."""
        encoder = RadixQEncoder(config)
        assert encoder.config == config
        assert encoder.radix == 4
        assert encoder.num_layers == 3
        assert encoder.lattice_dim == 8
    
    def test_encoding_decoding(self, encoder):
        """Test encoding and decoding."""
        input_tensor = torch.randint(0, 16, (32, 8))
        
        # Test encoding
        encoded = encoder.encode(input_tensor, depth=2)
        
        assert encoded.shape == input_tensor.shape
        assert encoded.dtype == torch.int32
        
        # Test decoding
        decoded = encoder.decode(encoded, depth=2)
        
        assert decoded.shape == input_tensor.shape
        assert decoded.dtype == torch.long
    
    def test_gradient_encoding(self, encoder):
        """Test gradient encoding/decoding."""
        gradients = torch.randn(1000, 8)
        
        # Test gradient encoding
        encoded_gradients = encoder.encode_gradients(gradients, depth=1)
        
        assert encoded_gradients.shape == gradients.shape
        assert encoded_gradients.dtype == torch.int32
        
        # Test gradient decoding
        decoded_gradients = encoder.decode_gradients(encoded_gradients, depth=1)
        
        assert decoded_gradients.shape == gradients.shape
        assert decoded_gradients.dtype == torch.long
    
    def test_compression_ratio(self, encoder):
        """Test compression ratio calculation."""
        ratio = encoder.compute_compression_ratio(depth=2)
        
        assert ratio > 0.0
        assert isinstance(ratio, float)
    
    def test_max_encoded_value(self, encoder):
        """Test maximum encoded value calculation."""
        max_value = encoder.get_max_encoded_value(depth=2)
        
        assert max_value > 0
        assert isinstance(max_value, int)


class TestQuantizedGradientHook:
    """Test quantized gradient hook."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
    
    @pytest.fixture
    def hook(self, config):
        """Create test hook."""
        return QuantizedGradientHook(
            config=config,
            communication_depth=1,
            compression_enabled=True,
            timing_enabled=True
        )
    
    def test_hook_creation(self, config):
        """Test hook creation."""
        hook = QuantizedGradientHook(config, timing_enabled=True)
        
        assert hook.config == config
        assert hook.communication_depth == 1
        assert hook.compression_enabled == True
        assert hook.timing_enabled == True
    
    def test_compression_stats(self, hook):
        """Test compression statistics."""
        stats = hook.get_compression_stats()
        
        assert 'compression_ratio' in stats
        assert 'max_encoded_value' in stats
        assert 'communication_depth' in stats
        assert 'compression_enabled' in stats
        
        assert stats['communication_depth'] == 1
        assert stats['compression_enabled'] == True
    
    def test_timing_stats(self, hook):
        """Test timing statistics."""
        stats = hook.get_timing_stats()
        
        assert 'quantization_time' in stats
        assert 'communication_time' in stats
        assert 'dequantization_time' in stats
        assert 'total_time' in stats
        assert 'num_calls' in stats
        
        assert stats['num_calls'] == 0  # No calls yet
    
    def test_depth_adaptation(self, hook):
        """Test communication depth adaptation."""
        # Test setting communication depth
        hook.set_communication_depth(2)
        assert hook.communication_depth == 2
        
        # Test invalid depth
        with pytest.raises(ValueError):
            hook.set_communication_depth(0)
        
        with pytest.raises(ValueError):
            hook.set_communication_depth(10)
    
    def test_compression_toggle(self, hook):
        """Test compression enable/disable."""
        # Test disabling compression
        hook.enable_compression(False)
        assert hook.compression_enabled == False
        
        # Test enabling compression
        hook.enable_compression(True)
        assert hook.compression_enabled == True
    
    def test_timing_toggle(self, hook):
        """Test timing enable/disable."""
        # Test disabling timing
        hook.enable_timing(False)
        assert hook.timing_enabled == False
        
        # Test enabling timing
        hook.enable_timing(True)
        assert hook.timing_enabled == True


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_quantization(self):
        """Test end-to-end quantization workflow."""
        # Create configuration
        config = LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
        
        # Create quantizer
        quantizer = LatticeQuantizer(config)
        
        # Create input
        input_tensor = torch.randn(32, 8)
        
        # Quantize
        quantized = quantizer.quantize(input_tensor)
        
        # Test decode from depth
        quantized_depth, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
        reconstructed = quantizer.decode_from_depth(indices, source_depth=1)
        
        # Check reconstruction quality
        error = torch.mean(torch.abs(input_tensor - reconstructed))
        assert error < 1.0
    
    def test_mlp_training(self):
        """Test MLP training with quantization."""
        # Create configuration
        config = LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
        
        # Create MLP
        mlp = QuantizedMLP(
            input_dim=784,
            hidden_dims=[512, 256],
            output_dim=10,
            config=config
        )
        
        # Create test data
        input_tensor = torch.randn(32, 784)
        target = torch.randint(0, 10, (32,))
        
        # Forward pass
        output = mlp(input_tensor)
        
        # Check output
        assert output.shape == (32, 10)
        assert output.dtype == torch.float32
        
        # Test loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        assert loss.item() > 0.0
        assert loss.requires_grad
    
    def test_distributed_training_simulation(self):
        """Test distributed training simulation."""
        # Create configuration
        config = LatticeConfig(
            type=LatticeType.HNLQ,
            radix=4,
            num_layers=3,
            lattice_dim=8
        )
        
        # Create hook
        hook = QuantizedGradientHook(config, communication_depth=1)
        
        # Create mock gradients
        gradients = torch.randn(1000, 8)
        
        # Simulate gradient communication
        quantized_gradients = hook.compressor.compress_gradients(gradients, depth=1)
        reconstructed_gradients = hook.compressor.decompress_gradients(quantized_gradients, depth=1)
        
        # Check compression
        original_size = gradients.numel() * gradients.element_size()
        compressed_size = quantized_gradients.numel() * quantized_gradients.element_size()
        compression_ratio = original_size / compressed_size
        
        assert compression_ratio > 1.0  # Should achieve compression
        
        # Check reconstruction quality
        error = torch.mean(torch.abs(gradients - reconstructed_gradients))
        assert error < 1.0  # Should be reasonably close


if __name__ == "__main__":
    pytest.main([__file__])
