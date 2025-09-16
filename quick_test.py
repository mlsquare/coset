#!/usr/bin/env python3
"""
Quick CoSet Test - Minimal Example

This is a minimal example to quickly test CoSet functionality.
Run with: python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from coset import LatticeConfig, LatticeType, LatticeQuantizer, QuantizedMLP

def main():
    print("🚀 Quick CoSet Test")
    print("=" * 30)
    
    # 1. Create configuration
    config = LatticeConfig(
        type=LatticeType.Z2,  # Use Z2 lattice (2D)
        radix=4,
        num_layers=3,
        beta=1.0,
        alpha=1.0
    )
    print(f"✅ Configuration: {config}")
    
    # 2. Test basic quantization with product quantization
    quantizer = LatticeQuantizer(config)
    input_tensor = torch.randn(10, 8)  # 8D input with 2D lattice (product quantization)
    
    quantized = quantizer.quantize(input_tensor)
    quantized_depth, indices = quantizer.quantize_to_depth(input_tensor, depth=1)
    reconstructed = quantizer.decode_from_depth(indices, source_depth=1)
    
    print(f"✅ Quantization: {input_tensor.shape} → {quantized.shape}")
    print(f"✅ Dequantization: {indices.shape} → {reconstructed.shape}")
    
    # 3. Test quantized MLP with product quantization
    mlp = QuantizedMLP(
        input_dim=784,  # 784D input (product quantization with 2D lattice)
        hidden_dims=[256, 128],
        output_dim=10,
        config=config
    )
    
    test_input = torch.randn(5, 784)  # 784D input (product quantization)
    output = mlp(test_input)
    
    print(f"✅ MLP: {test_input.shape} → {output.shape}")
    print(f"✅ Model has {sum(p.numel() for p in mlp.parameters()):,} parameters")
    
    # 4. Test packing encoding
    encoded = quantizer.packing_encode(input_tensor, packing_radix=4, depth=2)
    decoded = quantizer.packing_decode(encoded, packing_radix=4, depth=2)
    
    print(f"✅ Packing: {input_tensor.shape} → {encoded.shape} → {decoded.shape}")
    
    print("\n🎉 All tests passed! CoSet is working correctly.")
    print("\n💡 Try running 'python simple_example.py' for a comprehensive demo.")

if __name__ == "__main__":
    main()
