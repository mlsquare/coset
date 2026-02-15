#!/usr/bin/env python3
"""
E8 Lattice Codecs Verification Script

This script verifies two key properties of the E8 lattice codecs:

1. Idempotency: For a valid lattice point, encode_coords followed by 
   decode_coords should return the same point.

2. Full Pipeline: For an arbitrary vector, projection finds the closest 
   lattice point, encode_coords converts it to M-bit radix-q integers,
   and decode_coords reconstructs the lattice point. The reconstructed
   point should match the projected point.
"""

import sys
import os
import torch
import numpy as np

# Add the parent directory to the path to import coset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from coset.core.e8.lattice import E8Lattice
    from coset.core.base import LatticeConfig
    print("✓ Successfully imported coset core modules")
except ImportError as e:
    print(f"✗ Failed to import coset core modules: {e}")
    sys.exit(1)


def test_encode_decode_idempotency(lattice, q_values, num_tests=10, tolerance=1e-5):
    """
    Test that encode_coords followed by decode_coords is idempotent
    for valid lattice points.
    
    Note: Since encode_coords uses modulo q, idempotency only holds when
    the lattice point's coordinates in the generator basis are in [0, q-1].
    We generate lattice points from integer vectors z in [0, q-1]^8.
    
    Args:
        lattice: E8Lattice instance
        q_values: List of q values to test
        num_tests: Number of random test cases per q value
        tolerance: Floating point tolerance for comparisons
    
    Returns:
        Tuple of (passed_count, total_count, failures)
    """
    print("\n" + "="*70)
    print("TEST 1: Encode/Decode Idempotency")
    print("="*70)
    print(f"Testing {num_tests} random lattice points for each q value...")
    print("Note: Using lattice points generated from coordinates in [0, q-1]")
    print("      since encode_coords applies modulo q.")
    
    passed_count = 0
    total_count = 0
    failures = []
    
    for q in q_values:
        print(f"\nTesting with q={q}:")
        for i in range(num_tests):
            # Generate integer vector z in [0, q-1]^8
            # This ensures the lattice point can be encoded/decoded without information loss
            z = torch.randint(0, q, (8,), dtype=torch.float32)
            lattice_point = torch.matmul(lattice.G, z)
            
            # Encode and decode
            encoded = lattice.encode_coords(lattice_point, q)
            decoded = lattice.decode_coords(encoded, q)
            
            # Check if decoded equals original
            is_close = torch.allclose(lattice_point, decoded, atol=tolerance, rtol=tolerance)
            total_count += 1
            
            if is_close:
                passed_count += 1
                if i < 3:  # Show first 3 examples
                    print(f"  ✓ Test {i+1}: PASSED")
            else:
                failures.append({
                    'q': q,
                    'test': i+1,
                    'original': lattice_point,
                    'encoded': encoded,
                    'decoded': decoded,
                    'error': torch.norm(lattice_point - decoded).item()
                })
                print(f"  ✗ Test {i+1}: FAILED (error: {torch.norm(lattice_point - decoded).item():.2e})")
        
        print(f"  Summary for q={q}: {num_tests - len([f for f in failures if f['q'] == q])}/{num_tests} passed")
    
    return passed_count, total_count, failures


def test_full_pipeline(lattice, q_values, num_tests=10, tolerance=1e-5):
    """
    Test the full pipeline: arbitrary vector -> projection -> encode -> decode.
    
    Note: Since encode_coords uses modulo q, exact reconstruction only works
    when the projected point's generator coordinates are in [0, q-1]. For other
    cases, the decoded point will be a different (but valid) lattice point that
    is equivalent modulo q. We verify:
    1. The decoded point is a valid lattice point
    2. The encoded coordinates match what we'd get from the decoded point
    3. For small projected points, exact reconstruction should work
    
    Args:
        lattice: E8Lattice instance
        q_values: List of q values to test
        num_tests: Number of random test cases per q value
        tolerance: Floating point tolerance for comparisons
    
    Returns:
        Tuple of (passed_count, total_count, failures)
    """
    print("\n" + "="*70)
    print("TEST 2: Full Pipeline (Projection -> Encode -> Decode)")
    print("="*70)
    print(f"Testing {num_tests} random vectors for each q value...")
    print("Note: Using smaller scale vectors to increase chance of exact reconstruction.")
    
    passed_count = 0
    total_count = 0
    failures = []
    
    for q in q_values:
        print(f"\nTesting with q={q}:")
        for i in range(num_tests):
            # Generate arbitrary 8D vector with smaller scale
            # Smaller scale increases chance that projected point's generator
            # coordinates are in [0, q-1], allowing exact reconstruction
            scale = 2.0 + (i % 3) * 1.0  # Vary scale: 2, 3, 4
            arbitrary_vector = torch.randn(8, dtype=torch.float32) * scale
            
            # Step 1: Project to closest lattice point
            projected_point = lattice.projection(arbitrary_vector)
            
            # Step 2: Encode to radix-q coordinates
            encoded_coords = lattice.encode_coords(projected_point, q)
            
            # Step 3: Decode back to lattice point
            reconstructed_point = lattice.decode_coords(encoded_coords, q)
            
            # Verify the decoded point encodes to the same coordinates (consistency check)
            encoded_again = lattice.encode_coords(reconstructed_point, q)
            coords_match = torch.allclose(encoded_coords, encoded_again, atol=1e-6)
            
            # Check if reconstructed equals projected (exact reconstruction)
            is_exact = torch.allclose(projected_point, reconstructed_point, 
                                     atol=tolerance, rtol=tolerance)
            
            # If not exact, verify they are equivalent modulo q in the generator basis
            # i.e., G_inv @ (reconstructed - projected) should be multiples of q
            if not is_exact:
                diff = reconstructed_point - projected_point
                G_inv_diff = torch.matmul(lattice.G_inv, diff)
                # Check if all components are multiples of q (within tolerance)
                remainder = G_inv_diff % q
                # Allow remainder to be either close to 0 or close to q (wraparound)
                remainder_normalized = torch.minimum(remainder, q - remainder)
                is_modulo_q_equivalent = torch.allclose(remainder_normalized, 
                                                       torch.zeros_like(remainder_normalized),
                                                       atol=1e-5)
            else:
                is_modulo_q_equivalent = True
            
            total_count += 1
            
            # Test passes if either exact match OR modulo-q equivalent
            if is_exact:
                passed_count += 1
                if i < 3:  # Show first 3 examples
                    error_norm = torch.norm(arbitrary_vector - projected_point).item()
                    print(f"  ✓ Test {i+1}: PASSED (exact reconstruction, projection error: {error_norm:.4f})")
            elif is_modulo_q_equivalent and coords_match:
                passed_count += 1
                if i < 3:  # Show first 3 examples
                    error_norm = torch.norm(arbitrary_vector - projected_point).item()
                    print(f"  ✓ Test {i+1}: PASSED (modulo-q equivalent, projection error: {error_norm:.4f})")
            else:
                error = torch.norm(projected_point - reconstructed_point).item()
                failures.append({
                    'q': q,
                    'test': i+1,
                    'input': arbitrary_vector,
                    'projected': projected_point,
                    'encoded': encoded_coords,
                    'reconstructed': reconstructed_point,
                    'error': error,
                    'coords_match': coords_match,
                    'is_modulo_q_equivalent': is_modulo_q_equivalent if not is_exact else True
                })
                status_parts = []
                if not coords_match:
                    status_parts.append("coords don't match")
                if not is_modulo_q_equivalent:
                    status_parts.append("not modulo-q equivalent")
                status = ", ".join(status_parts) if status_parts else "unknown issue"
                print(f"  ✗ Test {i+1}: FAILED (reconstruction error: {error:.2e}, {status})")
        
        print(f"  Summary for q={q}: {num_tests - len([f for f in failures if f['q'] == q])}/{num_tests} passed")
    
    return passed_count, total_count, failures


def main():
    """Main verification function."""
    print("E8 Lattice Codecs Verification")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    lattice = E8Lattice(device=device)
    print(f"Initialized {lattice.name} lattice (dimension: {lattice.d})")
    
    # Test parameters
    q_values = [4, 8, 16, 32]
    num_tests_per_q = 10
    tolerance = 1e-5
    
    print(f"\nTest parameters:")
    print(f"  q values: {q_values}")
    print(f"  Tests per q: {num_tests_per_q}")
    print(f"  Tolerance: {tolerance}")
    
    # Run tests
    test1_passed, test1_total, test1_failures = test_encode_decode_idempotency(
        lattice, q_values, num_tests_per_q, tolerance
    )
    
    test2_passed, test2_total, test2_failures = test_full_pipeline(
        lattice, q_values, num_tests_per_q, tolerance
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Idempotency): {test1_passed}/{test1_total} passed")
    print(f"Test 2 (Full Pipeline): {test2_passed}/{test2_total} passed")
    print(f"Total: {test1_passed + test2_passed}/{test1_total + test2_total} passed")
    
    if test1_failures:
        print(f"\nTest 1 Failures ({len(test1_failures)}):")
        for failure in test1_failures[:5]:  # Show first 5 failures
            print(f"  q={failure['q']}, test={failure['test']}, error={failure['error']:.2e}")
    
    if test2_failures:
        print(f"\nTest 2 Failures ({len(test2_failures)}):")
        for failure in test2_failures[:5]:  # Show first 5 failures
            print(f"  q={failure['q']}, test={failure['test']}, error={failure['error']:.2e}")
    
    if test1_passed == test1_total and test2_passed == test2_total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ Some tests failed ({test1_total + test2_total - test1_passed - test2_passed} failures)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

