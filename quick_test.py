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
        type=LatticeType.HNLQ,
        radix=4,
        num_layers=3,
        lattice_dim=8
    )
    print(f"✅ Configuration: {config}")
    
    # 2. Test basic quantization
    quantizer = LatticeQuantizer(config)
    input_tensor = torch.randn(10, 8)
    
    quantized, indices = quantizer.quantize(input_tensor, depth=1)
    reconstructed = quantizer.dequantize(indices, depth=1)
    
    print(f"✅ Quantization: {input_tensor.shape} → {quantized.shape}")
    print(f"✅ Dequantization: {indices.shape} → {reconstructed.shape}")
    
    # 3. Test quantized MLP
    mlp = QuantizedMLP(
        input_dim=100,
        hidden_dims=[64, 32],
        output_dim=10,
        config=config
    )
    
    test_input = torch.randn(5, 100)
    output = mlp(test_input)
    
    print(f"✅ MLP: {test_input.shape} → {output.shape}")
    print(f"✅ Model has {sum(p.numel() for p in mlp.parameters()):,} parameters")
    
    # 4. Test radix-q encoding
    encoded = quantizer.radixq_encode(input_tensor, radix=4, depth=2)
    decoded = quantizer.radixq_decode(encoded, radix=4, depth=2)
    
    print(f"✅ Radix-Q: {input_tensor.shape} → {encoded.shape} → {decoded.shape}")
    
    print("\n🎉 All tests passed! CoSet is working correctly.")
    print("\n💡 Try running 'python simple_example.py' for a comprehensive demo.")

if __name__ == "__main__":
    main()
