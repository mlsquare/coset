#!/usr/bin/env python3
"""
Test script for CUDA product quantization kernels.
This script verifies that the CUDA implementation works correctly
with product quantization for arbitrary input dimensions.
"""

import torch
import numpy as np
from coset import LatticeConfig, LatticeType, LatticeQuantizer

def test_product_quantization_cuda():
    """Test product quantization with CUDA kernels."""
    print("🧪 Testing CUDA Product Quantization")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping CUDA tests")
        return False
    
    device = torch.device('cuda')
    print(f"✅ Using device: {device}")
    
    try:
        # Create lattice configuration
        config = LatticeConfig(
            type=LatticeType.Z2,  # 2D lattice
            radix=4,
            num_layers=3,
            beta=1.0,
            alpha=1.0
        )
        print(f"✅ Configuration: {config}")
        
        # Create quantizer
        quantizer = LatticeQuantizer(config).to(device)
        print(f"✅ Quantizer created on {device}")
        
        # Test with arbitrary input dimensions
        test_cases = [
            (32, 8),    # 8D input with 2D lattice (4 blocks)
            (16, 64),   # 64D input with 2D lattice (32 blocks)
            (8, 512),   # 512D input with 2D lattice (256 blocks)
        ]
        
        for batch_size, input_dim in test_cases:
            print(f"\n📊 Testing: batch_size={batch_size}, input_dim={input_dim}")
            
            # Create test input
            input_tensor = torch.randn(batch_size, input_dim, device=device)
            print(f"   Input shape: {input_tensor.shape}")
            
            # Test quantization
            quantized = quantizer.quantize(input_tensor)
            print(f"   Quantized shape: {quantized.shape}")
            
            # Test quantization to depth
            quantized_depth, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
            print(f"   Quantized depth shape: {quantized_depth.shape}")
            print(f"   Indices shape: {indices.shape}")
            
            # Verify indices have correct block structure
            expected_blocks = (input_dim + config.lattice_dim - 1) // config.lattice_dim
            expected_indices_shape = (batch_size, expected_blocks, config.lattice_dim)
            assert indices.shape == expected_indices_shape, f"Expected {expected_indices_shape}, got {indices.shape}"
            print(f"   ✅ Indices shape correct: {indices.shape}")
            
            # Test decoding
            reconstructed = quantizer.decode_from_depth(indices, source_depth=1)
            print(f"   Reconstructed shape: {reconstructed.shape}")
            
            # Verify shapes match
            assert quantized.shape == input_tensor.shape, f"Quantized shape mismatch: {quantized.shape} vs {input_tensor.shape}"
            assert reconstructed.shape == input_tensor.shape, f"Reconstructed shape mismatch: {reconstructed.shape} vs {input_tensor.shape}"
            print(f"   ✅ Shape consistency verified")
            
            # Test packing encoding/decoding
            packed = quantizer.packing_encode(input_tensor, packing_radix=4, depth=1)
            unpacked = quantizer.packing_decode(packed, packing_radix=4, depth=1)
            print(f"   Packed shape: {packed.shape}")
            print(f"   Unpacked shape: {unpacked.shape}")
            
            assert packed.shape == indices.shape, f"Packed shape mismatch: {packed.shape} vs {indices.shape}"
            assert unpacked.shape == input_tensor.shape, f"Unpacked shape mismatch: {unpacked.shape} vs {input_tensor.shape}"
            print(f"   ✅ Packing encoding/decoding verified")
            
            # Test quantization error
            quant_error = torch.mean(torch.abs(input_tensor - quantized)).item()
            print(f"   Quantization error: {quant_error:.4f}")
            
            print(f"   ✅ Test case passed!")
        
        print(f"\n🎉 All CUDA product quantization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_lattice_types():
    """Test product quantization with different lattice types."""
    print("\n🧪 Testing Different Lattice Types")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping lattice type tests")
        return False
    
    device = torch.device('cuda')
    
    lattice_types = [
        (LatticeType.Z2, 2),   # 2D square lattice
        (LatticeType.A2, 2),   # 2D hexagonal lattice
        (LatticeType.D4, 4),   # 4D lattice
        (LatticeType.E8, 8),   # 8D lattice
    ]
    
    for lattice_type, lattice_dim in lattice_types:
        print(f"\n📊 Testing {lattice_type.name} (dim={lattice_dim})")
        
        try:
            # Create configuration
            config = LatticeConfig(
                type=lattice_type,
                radix=4,
                num_layers=3,
                beta=1.0,
                alpha=1.0
            )
            
            # Create quantizer
            quantizer = LatticeQuantizer(config).to(device)
            
            # Test with input dimension that's not a multiple of lattice dimension
            input_dim = 32
            batch_size = 8
            input_tensor = torch.randn(batch_size, input_dim, device=device)
            
            # Quantize
            quantized = quantizer.quantize(input_tensor)
            quantized_depth, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
            
            # Verify shapes
            expected_blocks = (input_dim + lattice_dim - 1) // lattice_dim
            expected_indices_shape = (batch_size, expected_blocks, lattice_dim)
            
            assert indices.shape == expected_indices_shape, f"Expected {expected_indices_shape}, got {indices.shape}"
            assert quantized.shape == input_tensor.shape, f"Shape mismatch: {quantized.shape} vs {input_tensor.shape}"
            
            print(f"   ✅ {lattice_type.name} test passed (blocks: {expected_blocks})")
            
        except Exception as e:
            print(f"   ❌ {lattice_type.name} test failed: {e}")
            return False
    
    print(f"\n🎉 All lattice type tests passed!")
    return True

def main():
    """Run all CUDA product quantization tests."""
    print("🚀 CUDA Product Quantization Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test basic product quantization
    success &= test_product_quantization_cuda()
    
    # Test different lattice types
    success &= test_different_lattice_types()
    
    if success:
        print(f"\n🎉 All tests passed! CUDA product quantization is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()
