#!/usr/bin/env python3
"""
Test E8 HNLQ Complete Encode/Decode Cycle

This test verifies that our CUDA E8 HNLQ encoder and decoder work together
to perfectly reconstruct quantized vectors, matching the coset library behavior.
"""

import sys
import os
import torch
import numpy as np

# Add the parent directory to the path to import coset
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coset
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig
from coset.quant.functional import encode, decode

def test_e8_encode_decode_cycle():
    """Test complete E8 HNLQ encode/decode cycle with quantized vectors."""
    
    print("🔬 Testing E8 HNLQ Complete Encode/Decode Cycle")
    print("=" * 60)
    
    # Load CUDA kernels
    try:
        import torch.utils.cpp_extension
        encoder_module = torch.utils.cpp_extension.load(
            name="e8_hnlq_encoder",
            sources=["e8_hnlq_encoder_kernel.cu"],
            verbose=False
        )
        decoder_module = torch.utils.cpp_extension.load(
            name="e8_hnlq_decoder", 
            sources=["e8_hnlq_decoder_kernel.cu"],
            verbose=False
        )
        print("✅ CUDA kernels loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        return False
    
    # Test configuration
    Q = 3
    M = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create lattice and configuration
    lattice = E8Lattice()
    config = QuantizationConfig(q=Q, M=M, beta=1.0, alpha=1.0, with_dither=False)
    
    # Test vectors - use quantized vectors for perfect reconstruction
    test_vectors = [
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device),
        torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], device=device),
        torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device),
        torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0, 1.5, -1.5, 0.0], device=device),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device),
    ]
    
    # Move to device
    test_vectors = [v.to(device) for v in test_vectors]
    
    # Get transformation matrices
    T_to_lat = torch.eye(8, device=device)  # Identity for now
    G = lattice.G.to(device)
    G_inv = lattice.G_inv.to(device)
    
    print(f"📊 Testing with Q={Q}, M={M}")
    print(f"🔧 Device: {device}")
    print()
    
    all_passed = True
    
    for i, test_vector in enumerate(test_vectors):
        print(f"🧪 Test {i+1}: {test_vector.cpu().numpy()}")
        
        # CPU Reference: Encode and decode
        try:
            # CPU encode
            cpu_encoding, cpu_T = encode(test_vector, lattice, config)
            
            # CPU decode  
            cpu_reconstructed = decode(cpu_encoding, lattice, config, cpu_T)
            
            # CUDA encode
            cuda_indices = encoder_module.cuda_e8_hnlq_encode(
                test_vector.unsqueeze(0),  # Add batch dimension
                Q, M, T_to_lat, G_inv
            )
            
            # CUDA decode
            cuda_reconstructed = decoder_module.cuda_e8_hnlq_decode(
                cuda_indices, Q, M, T_to_lat, G
            )
            cuda_reconstructed = cuda_reconstructed.squeeze(0)  # Remove batch dimension
            
            # Calculate reconstruction errors
            cpu_error = torch.norm(cpu_reconstructed - test_vector).item()
            cuda_error = torch.norm(cuda_reconstructed - test_vector).item()
            
            # Check if reconstruction is perfect (within numerical precision)
            cpu_perfect = cpu_error < 1e-6
            cuda_perfect = cuda_error < 1e-6
            
            print(f"  📈 CPU reconstruction error: {cpu_error:.2e}")
            print(f"  📈 CUDA reconstruction error: {cuda_error:.2e}")
            print(f"  ✅ CPU perfect reconstruction: {cpu_perfect}")
            print(f"  ✅ CUDA perfect reconstruction: {cuda_perfect}")
            
            if not (cpu_perfect and cuda_perfect):
                print(f"  ❌ Reconstruction not perfect!")
                all_passed = False
            else:
                print(f"  🎉 Perfect reconstruction achieved!")
                
        except Exception as e:
            print(f"  ❌ Error during encode/decode: {e}")
            all_passed = False
        
        print()
    
    # Summary
    if all_passed:
        print("🎉 All tests PASSED: Perfect encode/decode cycle achieved!")
    else:
        print("❌ Some tests FAILED: Encode/decode cycle needs debugging")
    
    return all_passed

def test_with_simulated_vectors():
    """Test with vectors that are guaranteed to be quantized."""
    
    print("\n" + "=" * 60)
    print("🎯 Testing with Simulated Quantized Vectors")
    print("=" * 60)
    
    # Load CUDA kernels
    try:
        import torch.utils.cpp_extension
        encoder_module = torch.utils.cpp_extension.load(
            name="e8_hnlq_encoder",
            sources=["e8_hnlq_encoder_kernel.cu"],
            verbose=False
        )
        decoder_module = torch.utils.cpp_extension.load(
            name="e8_hnlq_decoder", 
            sources=["e8_hnlq_decoder_kernel.cu"],
            verbose=False
        )
        print("✅ CUDA kernels loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        return False
    
    # Test configuration
    Q = 3
    M = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create lattice and configuration
    lattice = E8Lattice()
    config = QuantizationConfig(q=Q, M=M, beta=1.0, alpha=1.0, with_dither=False)
    
    # Generate quantized test vectors using coset's quantize function
    print("🔬 Generating quantized test vectors...")
    
    # Create some random vectors and quantize them
    np.random.seed(42)  # For reproducibility
    quantized_vectors = []
    
    for i in range(5):
        # Generate random vector
        random_vec = torch.randn(8, device=device) * 2.0  # Scale for better quantization
        
        # Quantize it using coset
        quantized_vec = coset.quant.functional.quantize(random_vec, lattice, config)
        quantized_vectors.append(quantized_vec)
        
        print(f"  Vector {i+1}: {quantized_vec.cpu().numpy()}")
    
    print()
    
    # Get transformation matrices
    T_to_lat = torch.eye(8, device=device)  # Identity for now
    G = lattice.G.to(device)
    G_inv = lattice.G_inv.to(device)
    
    all_passed = True
    
    for i, quantized_vector in enumerate(quantized_vectors):
        print(f"🧪 Test {i+1} with quantized vector:")
        print(f"  Input: {quantized_vector.cpu().numpy()}")
        
        try:
            # CPU Reference: Encode and decode
            cpu_encoding, cpu_T = encode(quantized_vector, lattice, config)
            cpu_reconstructed = decode(cpu_encoding, lattice, config, cpu_T)
            
            # CUDA encode
            cuda_indices = encoder_module.cuda_e8_hnlq_encode(
                quantized_vector.unsqueeze(0),  # Add batch dimension
                Q, M, T_to_lat, G_inv
            )
            
            # CUDA decode
            cuda_reconstructed = decoder_module.cuda_e8_hnlq_decode(
                cuda_indices, Q, M, T_to_lat, G
            )
            cuda_reconstructed = cuda_reconstructed.squeeze(0)  # Remove batch dimension
            
            # Calculate reconstruction errors
            cpu_error = torch.norm(cpu_reconstructed - quantized_vector).item()
            cuda_error = torch.norm(cuda_reconstructed - quantized_vector).item()
            
            print(f"  📈 CPU reconstruction error: {cpu_error:.2e}")
            print(f"  📈 CUDA reconstruction error: {cuda_error:.2e}")
            print(f"  📈 CPU reconstructed: {cpu_reconstructed.cpu().numpy()}")
            print(f"  📈 CUDA reconstructed: {cuda_reconstructed.cpu().numpy()}")
            
            # Check if reconstruction is perfect (within numerical precision)
            cpu_perfect = cpu_error < 1e-6
            cuda_perfect = cuda_error < 1e-6
            
            if not (cpu_perfect and cuda_perfect):
                print(f"  ❌ Reconstruction not perfect!")
                all_passed = False
            else:
                print(f"  🎉 Perfect reconstruction achieved!")
                
        except Exception as e:
            print(f"  ❌ Error during encode/decode: {e}")
            all_passed = False
        
        print()
    
    # Summary
    if all_passed:
        print("🎉 All quantized vector tests PASSED!")
    else:
        print("❌ Some quantized vector tests FAILED!")
    
    return all_passed

if __name__ == "__main__":
    print("🚀 E8 HNLQ Complete Encode/Decode Cycle Test")
    print("=" * 60)
    
    # Test 1: Basic encode/decode cycle
    test1_passed = test_e8_encode_decode_cycle()
    
    # Test 2: With simulated quantized vectors
    test2_passed = test_with_simulated_vectors()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ E8 HNLQ encoder and decoder work perfectly together")
        print("✅ Perfect reconstruction achieved for quantized vectors")
        print("🚀 Ready for integration with vLUT system!")
    else:
        print("❌ SOME TESTS FAILED!")
        if not test1_passed:
            print("  - Basic encode/decode cycle needs debugging")
        if not test2_passed:
            print("  - Quantized vector reconstruction needs debugging")
    
    print("\n🔧 Next steps:")
    print("  1. Integrate with vLUT system")
    print("  2. Add performance benchmarking")
    print("  3. Test with larger batches and matrices")
