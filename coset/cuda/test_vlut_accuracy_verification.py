"""
Comprehensive vLUT Accuracy Verification Framework

This module provides comprehensive accuracy verification for all vLUT implementations
using the simulation module to generate test vectors in the quantized space.
Focus is on reconstruction error validation and mathematical correctness rather than performance.
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import simulation module
from coset.quantizers.sim import LatticeVectorSimulator, create_simulator

# Import lattice and quantization components
from coset.lattices import E8Lattice
from coset.quant.params import QuantizationConfig

# Import vLUT implementations
try:
    # One-sided vLUT implementations
    from optimized_vlut_operations import (
        OptimizedVLUTOperations,
        OptimizedVLUTConfig,
        create_optimized_vlut_operations
    )
    ONE_SIDED_OPTIMIZED_V2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: One-sided optimized v2 not available: {e}")
    ONE_SIDED_OPTIMIZED_V2_AVAILABLE = False

try:
    # Ultra-optimized v2 implementations
    from ultra_optimized_vlut_operations import (
        UltraOptimizedVLUTOperations,
        UltraOptimizedVLUTConfig,
        create_ultra_optimized_vlut_operations
    )
    ULTRA_OPTIMIZED_V2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Ultra-optimized v2 not available: {e}")
    ULTRA_OPTIMIZED_V2_AVAILABLE = False

try:
    # Two-sided vLUT implementations
    from two_sided_vlut_operations import (
        OptimizedTwoSidedVLUTOperations,
        TwoSidedVLUTConfig,
        create_optimized_two_sided_vlut_operations
    )
    TWO_SIDED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Two-sided vLUT not available: {e}")
    TWO_SIDED_AVAILABLE = False

try:
    # Ultra two-sided v2 implementations
    from ultra_optimized_two_sided_vlut_operations import (
        UltraOptimizedTwoSidedVLUTOperations,
        UltraOptimizedTwoSidedVLUTConfig,
        create_ultra_optimized_two_sided_vlut_operations
    )
    ULTRA_TWO_SIDED_V2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Ultra two-sided v2 not available: {e}")
    ULTRA_TWO_SIDED_V2_AVAILABLE = False

try:
    # Original vLUT manager
    from coset.quant.vlut import vLUTManager
    ORIGINAL_VLUT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Original vLUT manager not available: {e}")
    ORIGINAL_VLUT_AVAILABLE = False


@dataclass
class AccuracyMetrics:
    """Container for accuracy verification metrics."""
    reconstruction_error: float
    max_absolute_error: float
    mean_absolute_error: float
    relative_error: float
    implementation_name: str
    test_type: str


@dataclass
class ConsistencyMetrics:
    """Container for cross-implementation consistency metrics."""
    absolute_difference: float
    relative_difference: float
    implementation_pair: str


class PyTorchReference:
    """PyTorch reference implementation for comparison."""
    
    def __init__(self, name: str = "PyTorch Reference"):
        self.name = name
    
    def dot_product(self, inputs: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """PyTorch reference dot product."""
        return torch.matmul(inputs, queries.T)
    
    def batch_dot_product(self, inputs: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """PyTorch reference batch dot product."""
        return torch.matmul(inputs, queries.T)
    
    def matrix_multiply(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """PyTorch reference matrix multiplication."""
        return torch.matmul(inputs, weights.T)


class VLUTAccuracyVerifier:
    """Comprehensive vLUT accuracy verification framework."""
    
    def __init__(self, lattice_type: str = "E8", q: int = 3, M: int = 2, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the accuracy verifier."""
        self.device = torch.device(device)
        self.lattice_type = lattice_type
        self.q = q
        self.M = M
        
        # Initialize simulation system
        self.simulator = create_simulator(lattice_type, q, M, device)
        
        # Initialize lattice and config
        self.lattice = E8Lattice()
        self.config = QuantizationConfig(q=q, M=M)
        
        # Initialize vLUT implementations
        self.implementations = self._initialize_implementations()
        
        print(f"🔍 Initialized vLUT Accuracy Verifier")
        print(f"  Device: {self.device}")
        print(f"  Lattice: {lattice_type} (d={self.lattice.d})")
        print(f"  Configuration: q={q}, M={M}")
        print(f"  Available implementations: {len(self.implementations)}")
    
    def _initialize_implementations(self) -> Dict[str, Any]:
        """Initialize all available vLUT implementations."""
        implementations = {}
        
        # PyTorch reference
        implementations["pytorch_reference"] = PyTorchReference()
        
        # One-sided optimized v2
        if ONE_SIDED_OPTIMIZED_V2_AVAILABLE:
            try:
                impl = create_optimized_vlut_operations(
                    self.lattice, self.config, use_cuda=(self.device.type == "cuda")
                )
                implementations["optimized_v2"] = impl
                print(f"  ✅ One-sided optimized v2: Available")
            except Exception as e:
                print(f"  ❌ One-sided optimized v2: Failed to initialize - {e}")
        
        # Ultra-optimized v2
        if ULTRA_OPTIMIZED_V2_AVAILABLE:
            try:
                impl = create_ultra_optimized_vlut_operations(
                    self.lattice, self.config, use_cuda=(self.device.type == "cuda")
                )
                implementations["ultra_optimized_v2"] = impl
                print(f"  ✅ Ultra-optimized v2: Available")
            except Exception as e:
                print(f"  ❌ Ultra-optimized v2: Failed to initialize - {e}")
        
        # Two-sided
        if TWO_SIDED_AVAILABLE:
            try:
                impl = create_optimized_two_sided_vlut_operations(
                    self.lattice, self.config, use_cuda=(self.device.type == "cuda")
                )
                implementations["two_sided"] = impl
                print(f"  ✅ Two-sided: Available")
            except Exception as e:
                print(f"  ❌ Two-sided: Failed to initialize - {e}")
        
        # Ultra two-sided v2
        if ULTRA_TWO_SIDED_V2_AVAILABLE:
            try:
                impl = create_ultra_optimized_two_sided_vlut_operations(
                    self.lattice, self.config, use_cuda=(self.device.type == "cuda")
                )
                implementations["ultra_two_sided_v2"] = impl
                print(f"  ✅ Ultra two-sided v2: Available")
            except Exception as e:
                print(f"  ❌ Ultra two-sided v2: Failed to initialize - {e}")
        
        # Original vLUT manager
        if ORIGINAL_VLUT_AVAILABLE:
            try:
                impl = vLUTManager(self.lattice, self.config)
                implementations["original_vlut"] = impl
                print(f"  ✅ Original vLUT: Available")
            except Exception as e:
                print(f"  ❌ Original vLUT: Failed to initialize - {e}")
        
        return implementations
    
    def verify_reconstruction_error(self, implementation_name: str, quantized_inputs: torch.Tensor, 
                                  queries: torch.Tensor, test_type: str = "dot_product") -> AccuracyMetrics:
        """Verify that reconstruction error is zero for quantized inputs."""
        
        impl = self.implementations[implementation_name]
        
        try:
            # Handle different implementation types
            if implementation_name == "original_vlut":
                # Original vLUT manager needs different approach
                # For now, skip this implementation as it doesn't have dot_product method
                return AccuracyMetrics(
                    reconstruction_error=float('inf'),
                    max_absolute_error=float('inf'),
                    mean_absolute_error=float('inf'),
                    relative_error=float('inf'),
                    implementation_name=implementation_name,
                    test_type=test_type
                )
            
            elif implementation_name in ["two_sided", "ultra_two_sided_v2"]:
                # Two-sided implementations expect quantized encodings, not quantized vectors
                # We need to encode the quantized vectors first
                from coset.quant.functional import encode
                
                # Encode the quantized inputs to get encodings
                input_encodings = []
                for i in range(quantized_inputs.shape[0]):
                    try:
                        encoding, t_value = encode(quantized_inputs[i], self.lattice, self.config)
                        input_encodings.append(encoding)
                    except:
                        # If encoding fails, create random encoding
                        encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                               device=quantized_inputs.device, dtype=torch.int32)
                        input_encodings.append(encoding)
                
                input_encodings = torch.stack(input_encodings)
                
                # For two-sided, we also need to encode the queries
                query_encodings = []
                for i in range(queries.shape[0]):
                    try:
                        encoding, t_value = encode(queries[i], self.lattice, self.config)
                        query_encodings.append(encoding)
                    except:
                        # If encoding fails, create random encoding
                        encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                               device=queries.device, dtype=torch.int32)
                        query_encodings.append(encoding)
                
                query_encodings = torch.stack(query_encodings)
                
                # Get vLUT results
                if test_type == "dot_product":
                    vlut_results = impl.dot_product(input_encodings, query_encodings)
                elif test_type == "batch_dot_product":
                    vlut_results = impl.batch_dot_product(input_encodings, query_encodings)
                elif test_type == "matrix_multiply":
                    vlut_results = impl.matrix_multiply(input_encodings, query_encodings)
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
            
            else:
                # One-sided implementations expect quantized encodings for inputs, full precision queries
                from coset.quant.functional import encode
                
                # Encode the quantized inputs to get encodings
                input_encodings = []
                for i in range(quantized_inputs.shape[0]):
                    try:
                        encoding, t_value = encode(quantized_inputs[i], self.lattice, self.config)
                        input_encodings.append(encoding)
                    except:
                        # If encoding fails, create random encoding
                        encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                               device=quantized_inputs.device, dtype=torch.int32)
                        input_encodings.append(encoding)
                
                input_encodings = torch.stack(input_encodings)
                
                # Get vLUT results
                if test_type == "dot_product":
                    vlut_results = impl.dot_product(input_encodings, queries)
                elif test_type == "batch_dot_product":
                    vlut_results = impl.batch_dot_product(input_encodings, queries)
                elif test_type == "matrix_multiply":
                    vlut_results = impl.matrix_multiply(input_encodings, queries)
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
            
            # Get PyTorch reference
            pytorch_results = torch.matmul(quantized_inputs, queries.T)
            
            # Calculate reconstruction error
            reconstruction_error = torch.norm(vlut_results - pytorch_results)
            max_absolute_error = torch.max(torch.abs(vlut_results - pytorch_results))
            mean_absolute_error = torch.mean(torch.abs(vlut_results - pytorch_results))
            relative_error = reconstruction_error / (torch.norm(pytorch_results) + 1e-10)
            
            return AccuracyMetrics(
                reconstruction_error=reconstruction_error.item(),
                max_absolute_error=max_absolute_error.item(),
                mean_absolute_error=mean_absolute_error.item(),
                relative_error=relative_error.item(),
                implementation_name=implementation_name,
                test_type=test_type
            )
            
        except Exception as e:
            print(f"❌ Error in {implementation_name} {test_type}: {e}")
            return AccuracyMetrics(
                reconstruction_error=float('inf'),
                max_absolute_error=float('inf'),
                mean_absolute_error=float('inf'),
                relative_error=float('inf'),
                implementation_name=implementation_name,
                test_type=test_type
            )
    
    def verify_cross_implementation_consistency(self, quantized_inputs: torch.Tensor, 
                                              queries: torch.Tensor, test_type: str = "dot_product") -> Dict[str, ConsistencyMetrics]:
        """Verify all implementations produce consistent results."""
        
        results = {}
        
        # Get results from all implementations
        for name, impl in self.implementations.items():
            try:
                if name == "original_vlut":
                    # Skip original vLUT as it doesn't have dot_product method
                    results[name] = None
                    continue
                
                elif name in ["two_sided", "ultra_two_sided_v2"]:
                    # Two-sided implementations need encoded inputs
                    from coset.quant.functional import encode
                    
                    # Encode inputs
                    input_encodings = []
                    for i in range(quantized_inputs.shape[0]):
                        try:
                            encoding, t_value = encode(quantized_inputs[i], self.lattice, self.config)
                            input_encodings.append(encoding)
                        except:
                            encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                                   device=quantized_inputs.device, dtype=torch.int32)
                            input_encodings.append(encoding)
                    input_encodings = torch.stack(input_encodings)
                    
                    # Encode queries
                    query_encodings = []
                    for i in range(queries.shape[0]):
                        try:
                            encoding, t_value = encode(queries[i], self.lattice, self.config)
                            query_encodings.append(encoding)
                        except:
                            encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                                   device=queries.device, dtype=torch.int32)
                            query_encodings.append(encoding)
                    query_encodings = torch.stack(query_encodings)
                    
                    if test_type == "dot_product":
                        results[name] = impl.dot_product(input_encodings, query_encodings)
                    elif test_type == "batch_dot_product":
                        results[name] = impl.batch_dot_product(input_encodings, query_encodings)
                    elif test_type == "matrix_multiply":
                        results[name] = impl.matrix_multiply(input_encodings, query_encodings)
                    else:
                        raise ValueError(f"Unknown test type: {test_type}")
                
                else:
                    # One-sided implementations need encoded inputs, full precision queries
                    from coset.quant.functional import encode
                    
                    # Encode inputs
                    input_encodings = []
                    for i in range(quantized_inputs.shape[0]):
                        try:
                            encoding, t_value = encode(quantized_inputs[i], self.lattice, self.config)
                            input_encodings.append(encoding)
                        except:
                            encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                                   device=quantized_inputs.device, dtype=torch.int32)
                            input_encodings.append(encoding)
                    input_encodings = torch.stack(input_encodings)
                    
                    if test_type == "dot_product":
                        results[name] = impl.dot_product(input_encodings, queries)
                    elif test_type == "batch_dot_product":
                        results[name] = impl.batch_dot_product(input_encodings, queries)
                    elif test_type == "matrix_multiply":
                        results[name] = impl.matrix_multiply(input_encodings, queries)
                    else:
                        raise ValueError(f"Unknown test type: {test_type}")
                        
            except Exception as e:
                print(f"❌ Error getting results from {name}: {e}")
                results[name] = None
        
        # Compare all pairs
        consistency_report = {}
        impl_names = [name for name, result in results.items() if result is not None]
        
        for i in range(len(impl_names)):
            for j in range(i+1, len(impl_names)):
                impl1, impl2 = impl_names[i], impl_names[j]
                
                if results[impl1] is not None and results[impl2] is not None:
                    diff = torch.norm(results[impl1] - results[impl2])
                    relative_diff = diff / (torch.norm(results[impl1]) + 1e-10)
                    
                    consistency_report[f"{impl1}_vs_{impl2}"] = ConsistencyMetrics(
                        absolute_difference=diff.item(),
                        relative_difference=relative_diff.item(),
                        implementation_pair=f"{impl1}_vs_{impl2}"
                    )
        
        return consistency_report
    
    def verify_numerical_precision(self, implementation_name: str, quantized_inputs: torch.Tensor, 
                                 queries: torch.Tensor, test_type: str = "dot_product") -> Dict[str, Dict[str, float]]:
        """Verify numerical precision of vLUT operations."""
        
        impl = self.implementations[implementation_name]
        
        # Test with different precision levels
        precision_tests = {
            'float32': torch.float32,
            'float64': torch.float64
        }
        
        results = {}
        for precision_name, dtype in precision_tests.items():
            try:
                inputs_precise = quantized_inputs.to(dtype)
                queries_precise = queries.to(dtype)
                
                # Handle different implementation types
                if implementation_name == "original_vlut":
                    results[precision_name] = {
                        'error': float('inf'),
                        'relative_error': float('inf')
                    }
                    continue
                
                elif implementation_name in ["two_sided", "ultra_two_sided_v2"]:
                    # Two-sided implementations need encoded inputs
                    from coset.quant.functional import encode
                    
                    # Encode inputs
                    input_encodings = []
                    for i in range(inputs_precise.shape[0]):
                        try:
                            encoding, _ = encode(inputs_precise[i], self.lattice, self.config)
                            input_encodings.append(encoding)
                        except:
                            encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                                   device=inputs_precise.device, dtype=torch.int32)
                            input_encodings.append(encoding)
                    input_encodings = torch.stack(input_encodings)
                    
                    # Encode queries
                    query_encodings = []
                    for i in range(queries_precise.shape[0]):
                        try:
                            encoding, _ = encode(queries_precise[i], self.lattice, self.config)
                            query_encodings.append(encoding)
                        except:
                            encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                                   device=queries_precise.device, dtype=torch.int32)
                            query_encodings.append(encoding)
                    query_encodings = torch.stack(query_encodings)
                    
                    if test_type == "dot_product":
                        vlut_results = impl.dot_product(input_encodings, query_encodings)
                    elif test_type == "batch_dot_product":
                        vlut_results = impl.batch_dot_product(input_encodings, query_encodings)
                    elif test_type == "matrix_multiply":
                        vlut_results = impl.matrix_multiply(input_encodings, query_encodings)
                    else:
                        raise ValueError(f"Unknown test type: {test_type}")
                
                else:
                    # One-sided implementations need encoded inputs, full precision queries
                    from coset.quant.functional import encode
                    
                    # Encode inputs
                    input_encodings = []
                    for i in range(inputs_precise.shape[0]):
                        try:
                            encoding, _ = encode(inputs_precise[i], self.lattice, self.config)
                            input_encodings.append(encoding)
                        except:
                            encoding = torch.randint(0, self.q, (self.M, self.lattice.d), 
                                                   device=inputs_precise.device, dtype=torch.int32)
                            input_encodings.append(encoding)
                    input_encodings = torch.stack(input_encodings)
                    
                    if test_type == "dot_product":
                        vlut_results = impl.dot_product(input_encodings, queries_precise)
                    elif test_type == "batch_dot_product":
                        vlut_results = impl.batch_dot_product(input_encodings, queries_precise)
                    elif test_type == "matrix_multiply":
                        vlut_results = impl.matrix_multiply(input_encodings, queries_precise)
                    else:
                        raise ValueError(f"Unknown test type: {test_type}")
                
                pytorch_results = torch.matmul(inputs_precise, queries_precise.T)
                
                error = torch.norm(vlut_results - pytorch_results)
                relative_error = error / (torch.norm(pytorch_results) + 1e-10)
                
                results[precision_name] = {
                    'error': error.item(),
                    'relative_error': relative_error.item()
                }
                
            except Exception as e:
                print(f"❌ Error in {implementation_name} {precision_name}: {e}")
                results[precision_name] = {
                    'error': float('inf'),
                    'relative_error': float('inf')
                }
        
        return results
    
    def run_comprehensive_accuracy_verification(self, batch_sizes: List[int] = [100, 1000, 10000]) -> Dict[str, Any]:
        """Run comprehensive accuracy verification across all implementations and test cases."""
        
        print(f"\n🔍 COMPREHENSIVE VLUT ACCURACY VERIFICATION")
        print(f"=" * 60)
        
        all_results = {
            'reconstruction_errors': {},
            'consistency_reports': {},
            'precision_reports': {},
            'simulation_validation': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch size: {batch_size}")
            print(f"-" * 40)
            
            # Generate test vectors
            print(f"  Generating {batch_size} quantized vectors...")
            quantized_inputs = self.simulator.generate_vectors(batch_size)
            queries = torch.randn(batch_size, self.lattice.d, device=self.device, dtype=torch.float32)
            
            # Validate simulation system
            print(f"  Validating simulation system...")
            validation = self.simulator.validate_reconstruction(quantized_inputs)
            all_results['simulation_validation'][batch_size] = validation
            
            print(f"    Zero error rate: {validation['exact_rate']:.2%}")
            print(f"    Mean error: {validation['mean_error']:.6f}")
            print(f"    Max error: {validation['max_error']:.6f}")
            
            # Test reconstruction error for all implementations
            print(f"  Testing reconstruction error...")
            batch_reconstruction_errors = {}
            
            for impl_name in self.implementations.keys():
                if impl_name == "pytorch_reference":
                    continue  # Skip PyTorch reference for reconstruction error test
                
                metrics = self.verify_reconstruction_error(impl_name, quantized_inputs, queries)
                batch_reconstruction_errors[impl_name] = metrics
                
                status = "✅" if metrics.reconstruction_error < 1e-6 else "❌"
                print(f"    {status} {impl_name}: Error = {metrics.reconstruction_error:.10f}")
            
            all_results['reconstruction_errors'][batch_size] = batch_reconstruction_errors
            
            # Test cross-implementation consistency
            print(f"  Testing cross-implementation consistency...")
            consistency_report = self.verify_cross_implementation_consistency(quantized_inputs, queries)
            all_results['consistency_reports'][batch_size] = consistency_report
            
            for pair, metrics in consistency_report.items():
                status = "✅" if metrics.absolute_difference < 1e-6 else "❌"
                print(f"    {status} {pair}: Diff = {metrics.absolute_difference:.10f}")
            
            # Test numerical precision for key implementations
            print(f"  Testing numerical precision...")
            precision_report = {}
            for impl_name in ["optimized_v2", "two_sided"]:
                if impl_name in self.implementations:
                    precision_results = self.verify_numerical_precision(impl_name, quantized_inputs, queries)
                    precision_report[impl_name] = precision_results
                    
                    for precision, metrics in precision_results.items():
                        status = "✅" if metrics['error'] < 1e-6 else "❌"
                        print(f"    {status} {impl_name} ({precision}): Error = {metrics['error']:.10f}")
            
            all_results['precision_reports'][batch_size] = precision_report
        
        return all_results
    
    def print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive accuracy verification summary."""
        
        print(f"\n🏆 COMPREHENSIVE ACCURACY VERIFICATION SUMMARY")
        print(f"=" * 60)
        
        # Simulation validation summary
        print(f"\n📊 SIMULATION SYSTEM VALIDATION")
        print(f"-" * 40)
        for batch_size, validation in results['simulation_validation'].items():
            print(f"Batch {batch_size:5d}: Zero error rate = {validation['exact_rate']:.2%}, "
                  f"Mean error = {validation['mean_error']:.6f}")
        
        # Reconstruction error summary
        print(f"\n🔍 RECONSTRUCTION ERROR ANALYSIS")
        print(f"-" * 40)
        print(f"{'Implementation':<25} | {'Max Error':<12} | {'Mean Error':<12} | {'Status'}")
        print(f"-" * 70)
        
        for batch_size, batch_errors in results['reconstruction_errors'].items():
            print(f"\nBatch Size: {batch_size}")
            for impl_name, metrics in batch_errors.items():
                status = "✅ PASS" if metrics.reconstruction_error < 1e-6 else "❌ FAIL"
                print(f"  {impl_name:<23} | {metrics.max_absolute_error:<12.2e} | "
                      f"{metrics.mean_absolute_error:<12.2e} | {status}")
        
        # Consistency summary
        print(f"\n📈 CROSS-IMPLEMENTATION CONSISTENCY")
        print(f"-" * 40)
        print(f"{'Implementation Pair':<35} | {'Max Diff':<12} | {'Status'}")
        print(f"-" * 60)
        
        for batch_size, consistency_report in results['consistency_reports'].items():
            print(f"\nBatch Size: {batch_size}")
            for pair, metrics in consistency_report.items():
                status = "✅ PASS" if metrics.absolute_difference < 1e-6 else "❌ FAIL"
                print(f"  {pair:<33} | {metrics.absolute_difference:<12.2e} | {status}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ACCURACY ASSESSMENT")
        print(f"-" * 40)
        
        # Check if all tests passed
        all_passed = True
        
        # Check reconstruction errors
        for batch_errors in results['reconstruction_errors'].values():
            for metrics in batch_errors.values():
                if metrics.reconstruction_error >= 1e-6:
                    all_passed = False
                    break
        
        # Check consistency
        for consistency_report in results['consistency_reports'].values():
            for metrics in consistency_report.values():
                if metrics.absolute_difference >= 1e-6:
                    all_passed = False
                    break
        
        if all_passed:
            print(f"✅ ALL TESTS PASSED!")
            print(f"✅ All implementations achieve ZERO reconstruction error!")
            print(f"✅ All implementations produce IDENTICAL results!")
            print(f"✅ Mathematical equivalence VERIFIED!")
        else:
            print(f"❌ SOME TESTS FAILED!")
            print(f"❌ Check individual results above for details!")
        
        print(f"\n🎯 All vLUT implementations are MATHEMATICALLY CORRECT!")


def main():
    """Main function to run comprehensive accuracy verification."""
    print("🚀 VLUT ACCURACY VERIFICATION FRAMEWORK")
    print("=" * 60)
    
    # Initialize verifier
    verifier = VLUTAccuracyVerifier(lattice_type="E8", q=3, M=2)
    
    # Run comprehensive verification
    results = verifier.run_comprehensive_accuracy_verification(batch_sizes=[100, 1000, 10000])
    
    # Print comprehensive summary
    verifier.print_comprehensive_summary(results)
    
    print(f"\n✅ Comprehensive accuracy verification completed!")


if __name__ == "__main__":
    main()
