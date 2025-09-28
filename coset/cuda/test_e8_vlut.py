#!/usr/bin/env python3
"""
Test vLUT with Working Vectors

This test uses only vectors that we know work correctly with the CUDA encoder.
"""

import sys
import os
import torch
import numpy as np
import time

# Add the parent directory to the path to import coset
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import coset
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig

def test_vlut_with_working_vectors():
    """Test vLUT with vectors that work correctly."""
    
    print("🚀 vLUT Test with Working Vectors")
    print("=" * 50)
    
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
        vlut_module = torch.utils.cpp_extension.load(
            name="e8_vlut",
            sources=["e8_vlut_kernel.cu"],
            verbose=False
        )
        print("✅ All CUDA kernels loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        return False
    
    # Test configuration
    Q = 3
    M_levels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create lattice and configuration
    lattice = E8Lattice()
    config = QuantizationConfig(q=Q, M=M_levels, beta=1.0, alpha=1.0, with_dither=False)
    
    # Get transformation matrices
    T_to_lat = torch.eye(8, device=device)
    G = lattice.G.to(device)
    G_inv = lattice.G_inv.to(device)
    vlut_table = torch.zeros(1000, device=device)
    
    print(f"📊 Configuration: Q={Q}, M={M_levels}")
    print(f"🔧 Device: {device}")
    print()
    
    # Use only vectors that we know work correctly
    working_vectors = [
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device),
        torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], device=device),
        torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device),
    ]
    
    # Create a simple 4x8 matrix from working vectors
    A = torch.stack(working_vectors)
    B = torch.ones(8, 2, device=device)  # 8x2 matrix
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"A:\n{A.cpu().numpy()}")
    print(f"B:\n{B.cpu().numpy()}")
    
    # PyTorch reference
    C_ref = torch.matmul(A, B)
    print(f"PyTorch result C: {C_ref.cpu().numpy()}")
    
    # Encode matrix A manually (we know these vectors work)
    print("\n🔍 Encoding matrix A with working vectors...")
    A_encoded = torch.zeros(4, 1, dtype=torch.int32, device=device)  # 4 rows, 1 block (8 columns)
    
    for i, vector in enumerate(working_vectors):
        print(f"  Encoding row {i}: {vector.cpu().numpy()}")
        
        # Encode using CUDA encoder
        cuda_encoded = encoder_module.cuda_e8_hnlq_encode(
            vector.unsqueeze(0), Q, M_levels, T_to_lat, G_inv
        )
        A_encoded[i, 0] = cuda_encoded[0]
        print(f"  Encoded index: {cuda_encoded[0].item()}")
        
        # Verify by decoding
        decoded = decoder_module.cuda_e8_hnlq_decode(
            cuda_encoded, Q, M_levels, T_to_lat, G
        )
        error = torch.norm(decoded.squeeze(0) - vector).item()
        print(f"  Reconstruction error: {error:.2e}")
    
    print(f"A_encoded shape: {A_encoded.shape}")
    print(f"A_encoded:\n{A_encoded.cpu().numpy()}")
    
    # Test vLUT
    print("\n🔍 Testing vLUT matrix multiplication...")
    try:
        C_vlut = vlut_module.cuda_e8_vlut_onesided_matmul(
            A_encoded, B, Q, M_levels, T_to_lat, G_inv, G, vlut_table
        )
        print(f"vLUT result C: {C_vlut.cpu().numpy()}")
        
        # Compare results
        error = torch.norm(C_vlut - C_ref).item()
        max_error = torch.max(torch.abs(C_vlut - C_ref)).item()
        
        print(f"📈 Reconstruction error: {error:.2e}")
        print(f"📈 Max element error: {max_error:.2e}")
        
        if error < 1e-6:
            print("✅ vLUT test with working vectors passed!")
            return True
        else:
            print("❌ vLUT test with working vectors failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in vLUT: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_larger_matrix_with_working_vectors():
    """Test with a larger matrix using repeated working vectors."""
    
    print("\n" + "=" * 50)
    print("🧪 Testing Larger Matrix with Working Vectors")
    print("=" * 50)
    
    # Load CUDA kernels
    try:
        import torch.utils.cpp_extension
        encoder_module = torch.utils.cpp_extension.load(
            name="e8_hnlq_encoder",
            sources=["e8_hnlq_encoder_kernel.cu"],
            verbose=False
        )
        vlut_module = torch.utils.cpp_extension.load(
            name="e8_vlut",
            sources=["e8_vlut_kernel.cu"],
            verbose=False
        )
        print("✅ CUDA kernels loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        return False
    
    # Test configuration
    Q = 3
    M_levels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create lattice and configuration
    lattice = E8Lattice()
    config = QuantizationConfig(q=Q, M=M_levels, beta=1.0, alpha=1.0, with_dither=False)
    
    # Get transformation matrices
    T_to_lat = torch.eye(8, device=device)
    G = lattice.G.to(device)
    G_inv = lattice.G_inv.to(device)
    vlut_table = torch.zeros(1000, device=device)
    
    # Create a larger matrix by repeating working vectors
    working_vectors = [
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device),
        torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], device=device),
    ]
    
    # Create 8x16 matrix (8 rows, 2 blocks of 8 columns each)
    A_rows = []
    for i in range(8):
        A_rows.append(working_vectors[i % 2])  # Alternate between the two working vectors
    
    A = torch.stack(A_rows)  # 8x8 matrix
    
    # Extend to 8x16 by duplicating
    A = torch.cat([A, A], dim=1)  # 8x16 matrix
    
    B = torch.ones(16, 4, device=device)  # 16x4 matrix
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    
    # PyTorch reference
    C_ref = torch.matmul(A, B)
    print(f"PyTorch result C shape: {C_ref.shape}")
    print(f"PyTorch result C:\n{C_ref.cpu().numpy()}")
    
    # Encode matrix A
    print("\n🔍 Encoding larger matrix A...")
    A_encoded = torch.zeros(8, 2, dtype=torch.int32, device=device)  # 8 rows, 2 blocks
    
    for i in range(8):
        for block in range(2):
            vector = A[i, block*8:(block+1)*8]
            
            # Encode using CUDA encoder
            cuda_encoded = encoder_module.cuda_e8_hnlq_encode(
                vector.unsqueeze(0), Q, M_levels, T_to_lat, G_inv
            )
            A_encoded[i, block] = cuda_encoded[0]
    
    print(f"A_encoded shape: {A_encoded.shape}")
    
    # Test vLUT
    print("\n🔍 Testing vLUT with larger matrix...")
    try:
        start_time = time.time()
        C_vlut = vlut_module.cuda_e8_vlut_onesided_matmul(
            A_encoded, B, Q, M_levels, T_to_lat, G_inv, G, vlut_table
        )
        vlut_time = time.time() - start_time
        
        print(f"vLUT result C shape: {C_vlut.shape}")
        print(f"vLUT result C:\n{C_vlut.cpu().numpy()}")
        
        # Compare results
        error = torch.norm(C_vlut - C_ref).item()
        max_error = torch.max(torch.abs(C_vlut - C_ref)).item()
        
        print(f"📈 Reconstruction error: {error:.2e}")
        print(f"📈 Max element error: {max_error:.2e}")
        print(f"⏱️  vLUT time: {vlut_time:.4f}s")
        
        if error < 1e-6:
            print("✅ vLUT test with larger matrix passed!")
            return True
        else:
            print("❌ vLUT test with larger matrix failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in vLUT: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 vLUT Test Suite with Working Vectors")
    print("=" * 50)
    
    # Test 1: Simple matrix with working vectors
    test1_passed = test_vlut_with_working_vectors()
    
    # Test 2: Larger matrix with working vectors
    test2_passed = test_larger_matrix_with_working_vectors()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ vLUT works correctly with vectors that encode properly")
        print("🚀 Ready to debug encoding issues for complex vectors")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Need to debug vLUT implementation")
    
    print("\n🔧 Next steps:")
    print("  1. Fix CUDA encoder for complex vectors")
    print("  2. Implement true vLUT table lookup")
    print("  3. Scale to production use")
