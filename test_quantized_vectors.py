"""
Test script for lattice quantization using simulated quantized data from sim.py.

This script uses the LatticeVectorSimulator to generate vectors that are already
in the quantized space, then tests the encoder/decoder to ensure zero reconstruction error.
"""

import torch
import numpy as np
from coset.lattices import Z2Lattice, D4Lattice, E8Lattice
from coset.quant import QuantizationConfig, encode, decode, quantize
from coset.quant.sim import LatticeVectorSimulator, create_simulator


def test_encoder_decoder_with_simulated_data():
    """Test encoder/decoder with simulated quantized vectors."""
    print("🚀 Testing Encoder/Decoder with Simulated Quantized Data")
    print("=" * 70)
    
    # Test different lattice types and configurations
    test_configs = [
        ("Z2", "Z2", 3, 2),
        ("D4", "D4", 3, 2), 
        ("E8", "E8", 3, 2),
        ("D4", "D4", 4, 2),
        ("E8", "E8", 4, 3)
    ]
    
    for lattice_name, lattice_type, q, M in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing {lattice_name} Lattice: q={q}, M={M}")
        print(f"{'='*60}")
        
        # Create simulator
        simulator = create_simulator(lattice_type, q, M, device="cpu")
        
        # Generate simulated quantized vectors
        batch_size = 20
        print(f"Generating {batch_size} simulated quantized vectors...")
        simulated_vectors = simulator.generate_vectors(batch_size)
        
        print(f"Vector shape: {simulated_vectors.shape}")
        print(f"Vector range: [{torch.min(simulated_vectors):.4f}, {torch.max(simulated_vectors):.4f}]")
        print(f"Vector mean: {torch.mean(simulated_vectors):.4f}")
        print(f"Vector std: {torch.std(simulated_vectors):.4f}")
        
        # Test encoder/decoder on each simulated vector
        zero_error_count = 0
        total_error = 0.0
        max_error = 0.0
        
        print(f"\nTesting encoder/decoder on {batch_size} vectors:")
        for i in range(batch_size):
            x = simulated_vectors[i]
            
            # Encode the simulated vector
            try:
                b, T = encode(x, simulator.lattice, simulator.config)
                
                # Decode back
                x_reconstructed = decode(b, simulator.lattice, simulator.config, T)
                
                # Calculate reconstruction error
                error = torch.norm(x - x_reconstructed)
                total_error += error.item()
                max_error = max(max_error, error.item())
                
                if error < 1e-6:  # Essentially zero
                    zero_error_count += 1
                
                if i < 5:  # Show details for first 5 vectors
                    print(f"  Vector {i+1}: error = {error:.8f}, T = {T}")
                    print(f"    Original:    {x}")
                    print(f"    Reconstructed: {x_reconstructed}")
                    print(f"    Encoding shape: {b.shape}")
                
            except Exception as e:
                print(f"  Vector {i+1}: ERROR - {e}")
                continue
        
        # Calculate statistics
        avg_error = total_error / batch_size
        zero_error_rate = zero_error_count / batch_size
        
        print(f"\n  Results for {lattice_name} (q={q}, M={M}):")
        print(f"    Average reconstruction error: {avg_error:.8f}")
        print(f"    Maximum reconstruction error: {max_error:.8f}")
        print(f"    Zero error rate: {zero_error_rate:.2%}")
        print(f"    Expected: ~100% zero error for simulated quantized vectors")
        
        if zero_error_rate >= 0.95:  # 95% or higher zero error rate
            print(f"    ✅ PASS: High zero error rate achieved")
        else:
            print(f"    ❌ FAIL: Low zero error rate")


def test_quantize_wrapper_with_simulated_data():
    """Test the quantize wrapper function with simulated data."""
    print(f"\n{'='*70}")
    print("Testing quantize() wrapper function with simulated data")
    print(f"{'='*70}")
    
    # Test with D4 lattice
    simulator = create_simulator("D4", q=4, M=2, device="cpu")
    
    # Generate simulated vectors
    batch_size = 15
    print(f"Generating {batch_size} simulated vectors for quantize wrapper test...")
    simulated_vectors = simulator.generate_vectors(batch_size)
    
    print(f"Testing quantize() wrapper consistency:")
    consistent_count = 0
    
    for i in range(batch_size):
        x = simulated_vectors[i]
        
        try:
            # Use quantize wrapper to quantize the simulated vector
            x_quantized = quantize(x, simulator.lattice, simulator.config)
            
            # Re-quantize to check consistency
            x_re_quantized = quantize(x_quantized, simulator.lattice, simulator.config)
            
            # Check if re-quantization gives same result (should be identical)
            consistency_error = torch.norm(x_quantized - x_re_quantized)
            
            if consistency_error < 1e-6:
                consistent_count += 1
                status = "✅"
            else:
                status = "❌"
            
            if i < 5:  # Show details for first 5 vectors
                print(f"  Vector {i+1}: {status} consistency_error = {consistency_error:.8f}")
                print(f"    Original:      {x}")
                print(f"    Quantized:     {x_quantized}")
                print(f"    Re-quantized:  {x_re_quantized}")
            
        except Exception as e:
            print(f"  Vector {i+1}: ERROR - {e}")
            continue
    
    consistency_rate = consistent_count / batch_size
    print(f"\n  Quantize wrapper consistency results:")
    print(f"    Consistent vectors: {consistent_count}/{batch_size}")
    print(f"    Consistency rate: {consistency_rate:.2%}")
    print(f"    Expected: ~100% consistency for quantized vectors")
    
    if consistency_rate >= 0.95:
        print(f"    ✅ PASS: High consistency rate achieved")
    else:
        print(f"    ❌ FAIL: Low consistency rate")


def test_different_parameters():
    """Test with different quantization parameters."""
    print(f"\n{'='*70}")
    print("Testing with different quantization parameters")
    print(f"{'='*70}")
    
    # Test different parameter combinations
    param_configs = [
        ("E8", 3, 2),
        ("E8", 4, 2),
        ("E8", 3, 3),
        ("D4", 3, 2),
        ("D4", 4, 2),
        ("D4", 5, 2),
        ("Z2", 3, 2),
        ("Z2", 4, 3)
    ]
    
    for lattice_type, q, M in param_configs:
        print(f"\nTesting {lattice_type} with q={q}, M={M}:")
        
        try:
            # Create simulator
            simulator = create_simulator(lattice_type, q, M, device="cpu")
            
            # Generate a small batch for quick testing
            batch_size = 10
            simulated_vectors = simulator.generate_vectors(batch_size)
            
            # Test a few vectors
            zero_error_count = 0
            for i in range(min(5, batch_size)):
                x = simulated_vectors[i]
                
                # Test encode/decode
                b, T = encode(x, simulator.lattice, simulator.config)
                x_reconstructed = decode(b, simulator.lattice, simulator.config, T)
                
                error = torch.norm(x - x_reconstructed)
                if error < 1e-6:
                    zero_error_count += 1
            
            success_rate = zero_error_count / min(5, batch_size)
            print(f"  Success rate: {success_rate:.2%} ({zero_error_count}/{min(5, batch_size)})")
            
        except Exception as e:
            print(f"  ERROR: {e}")


def test_validation_with_simulator():
    """Test the simulator's built-in validation."""
    print(f"\n{'='*70}")
    print("Testing simulator's built-in validation")
    print(f"{'='*70}")
    
    # Test with E8 lattice
    simulator = create_simulator("E8", q=3, M=2, device="cpu")
    
    # Generate vectors
    batch_size = 50
    print(f"Generating {batch_size} vectors and running validation...")
    simulated_vectors = simulator.generate_vectors(batch_size)
    
    # Use simulator's validation method
    validation_results = simulator.validate_reconstruction(simulated_vectors)
    
    print(f"Validation results:")
    print(f"  Max error: {validation_results['max_error']:.8f}")
    print(f"  Mean error: {validation_results['mean_error']:.8f}")
    print(f"  Std error: {validation_results['std_error']:.8f}")
    print(f"  Exact reconstructions: {validation_results['exact_reconstructions']}/{batch_size}")
    print(f"  Exact rate: {validation_results['exact_rate']:.2%}")
    print(f"  Tolerance: {validation_results['tolerance']}")
    
    if validation_results['exact_rate'] >= 0.95:
        print(f"  ✅ PASS: High exact reconstruction rate")
    else:
        print(f"  ❌ FAIL: Low exact reconstruction rate")


if __name__ == "__main__":
    print("🧪 Testing Lattice Quantization with Simulated Quantized Data")
    print("Using LatticeVectorSimulator from sim.py")
    print("=" * 70)
    
    # Test 1: Encoder/decoder with simulated data
    test_encoder_decoder_with_simulated_data()
    
    # Test 2: Quantize wrapper with simulated data
    test_quantize_wrapper_with_simulated_data()
    
    # Test 3: Different parameters
    test_different_parameters()
    
    # Test 4: Simulator's built-in validation
    test_validation_with_simulator()
    
    print(f"\n{'='*70}")
    print("✅ All tests completed!")
    print("Expected results:")
    print("- Simulated quantized vectors should have ~100% zero reconstruction error")
    print("- Quantize wrapper should be consistent (re-quantization gives same result)")
    print("- Different lattice types and parameters should work correctly")
