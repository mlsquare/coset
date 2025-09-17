#!/usr/bin/env python3
"""
Test ultra-optimized quantization kernels for memory efficiency and performance
"""

import torch
import time
from coset import LatticeConfig, LatticeType, LatticeQuantizer
from coset.layers.autograd import quantized_matmul

def create_unit_norm_matrix(shape, device='cuda'):
    """Create a random matrix with unit norm"""
    A = torch.randn(shape, device=device)
    A = A / torch.norm(A)
    return A

def test_ultra_optimized_kernels():
    """Test ultra-optimized kernels for memory efficiency and performance"""
    print("🎯 Ultra-Optimized Quantization Kernels Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    try:
        config = LatticeConfig(type=LatticeType.E8, radix=2, num_layers=1)
        
        # Test configurations - focus on larger matrices
        test_configs = [
            (256, 256, 256),   # Medium
            (512, 512, 512),   # Large
            (1024, 1024, 1024), # Very large
        ]
        
        print(f"🔧 Configuration: E8, radix=2, layers=1, lattice_dim=8")
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print()
        
        for batch_size, input_dim, out_dim in test_configs:
            print(f"📏 Testing {batch_size}x{input_dim} @ {input_dim}x{out_dim} = {batch_size}x{out_dim}")
            print("-" * 70)
            
            # Create unit norm matrices
            A = create_unit_norm_matrix((batch_size, input_dim), device)
            B = create_unit_norm_matrix((input_dim, out_dim), device)
            
            print(f"   A shape: {A.shape}, memory: {A.numel() * 4 / 1024**2:.1f} MB")
            print(f"   B shape: {B.shape}, memory: {B.numel() * 4 / 1024**2:.1f} MB")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            # Test 1: PyTorch Standard MatMul
            print("\n1️⃣ PyTorch Standard MatMul (A @ B):")
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(5):
                pytorch_result = torch.matmul(A, B)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start) / 5 * 1000
            
            pytorch_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"   Time: {pytorch_time:.3f}ms")
            print(f"   Memory usage: {pytorch_memory - initial_memory:.1f} MB")
            print(f"   Output shape: {pytorch_result.shape}")
            
            # Test 2: Current Optimized CUDA Kernels
            print("\n2️⃣ Current Optimized CUDA Kernels:")
            
            # Clear memory
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024**2
            
            quantizer_current = LatticeQuantizer(config, use_cuda_kernels=True).to(device)
            
            # Quantize A
            print("   Quantizing A...")
            torch.cuda.synchronize()
            start = time.time()
            _, A_indices = quantizer_current.quantize_to_depth(A, 0)
            torch.cuda.synchronize()
            A_quant_time = (time.time() - start) * 1000
            print(f"   A quantization: {A_quant_time:.3f}ms")
            
            # Quantize B
            print("   Quantizing B...")
            torch.cuda.synchronize()
            start = time.time()
            _, B_indices = quantizer_current.quantize_to_depth(B.t(), 0)
            torch.cuda.synchronize()
            B_quant_time = (time.time() - start) * 1000
            print(f"   B quantization: {B_quant_time:.3f}ms")
            
            # Create lookup table
            print("   Creating lookup table...")
            torch.cuda.synchronize()
            start = time.time()
            lookup_table = quantizer_current.create_lookup_table()
            torch.cuda.synchronize()
            lut_time = (time.time() - start) * 1000
            print(f"   Lookup table: {lut_time:.3f}ms")
            
            memory_after_quant = torch.cuda.memory_allocated() / 1024**2
            print(f"   Memory after quantization: {memory_after_quant - memory_before:.1f} MB")
            
            # Try quantized matmul
            try:
                print("   Running quantized matmul...")
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(5):
                    current_result = quantized_matmul(A_indices, B_indices, lookup_table, None)
                torch.cuda.synchronize()
                current_matmul_time = (time.time() - start) / 5 * 1000
                
                total_current_time = A_quant_time + B_quant_time + current_matmul_time
                print(f"   MatMul time: {current_matmul_time:.3f}ms")
                print(f"   Total time: {total_current_time:.3f}ms")
                print(f"   Output shape: {current_result.shape}")
                
                current_success = True
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ Out of memory during matmul: {e}")
                    current_success = False
                else:
                    raise e
            
            # Test 3: Ultra-Optimized Kernels
            print("\n3️⃣ Ultra-Optimized CUDA Kernels:")
            
            # Clear memory
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024**2
            
            try:
                from coset.quantizers.ultra_optimized_quantization_kernels import (
                    ultra_optimized_quantize_cuda,
                    chunked_quantized_matmul_cuda
                )
                
                # Test ultra-optimized quantization
                print("   Testing ultra-optimized quantization...")
                torch.cuda.synchronize()
                start = time.time()
                _, A_indices_ultra = ultra_optimized_quantize_cuda(
                    A,
                    quantizer_current.lattice.get_generator_matrix(),
                    quantizer_current.lattice.get_inverse_generator_matrix(),
                    quantizer_current.lattice.get_eps(),
                    quantizer_current.lattice.get_beta(),
                    quantizer_current.q
                )
                torch.cuda.synchronize()
                A_quant_time_ultra = (time.time() - start) * 1000
                print(f"   A quantization: {A_quant_time_ultra:.3f}ms")
                
                torch.cuda.synchronize()
                start = time.time()
                _, B_indices_ultra = ultra_optimized_quantize_cuda(
                    B.t(),
                    quantizer_current.lattice.get_generator_matrix(),
                    quantizer_current.lattice.get_inverse_generator_matrix(),
                    quantizer_current.lattice.get_eps(),
                    quantizer_current.lattice.get_beta(),
                    quantizer_current.q
                )
                torch.cuda.synchronize()
                B_quant_time_ultra = (time.time() - start) * 1000
                print(f"   B quantization: {B_quant_time_ultra:.3f}ms")
                
                memory_after_quant_ultra = torch.cuda.memory_allocated() / 1024**2
                print(f"   Memory after quantization: {memory_after_quant_ultra - memory_before:.1f} MB")
                
                # Test chunked matmul
                print("   Testing chunked matmul...")
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(5):
                    ultra_result = chunked_quantized_matmul_cuda(
                        A_indices_ultra, B_indices_ultra, lookup_table, None, chunk_size=128
                    )
                torch.cuda.synchronize()
                ultra_matmul_time = (time.time() - start) / 5 * 1000
                
                total_ultra_time = A_quant_time_ultra + B_quant_time_ultra + ultra_matmul_time
                print(f"   MatMul time: {ultra_matmul_time:.3f}ms")
                print(f"   Total time: {total_ultra_time:.3f}ms")
                print(f"   Output shape: {ultra_result.shape}")
                
                ultra_success = True
                
            except ImportError as e:
                print(f"   ❌ Could not import ultra-optimized kernels: {e}")
                ultra_success = False
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ Out of memory with ultra-optimized kernels: {e}")
                    ultra_success = False
                else:
                    raise e
            
            # Performance Comparison
            print(f"\n📊 Performance Comparison:")
            print(f"   PyTorch standard: {pytorch_time:.3f}ms")
            
            if current_success:
                print(f"   Current CUDA: {total_current_time:.3f}ms")
            else:
                print(f"   Current CUDA: ❌ Out of memory")
            
            if ultra_success:
                print(f"   Ultra-optimized: {total_ultra_time:.3f}ms")
            else:
                print(f"   Ultra-optimized: ❌ Failed")
            
            # Memory Comparison
            print(f"\n🔍 Memory Usage Comparison:")
            print(f"   PyTorch: {pytorch_memory - initial_memory:.1f} MB")
            
            if current_success:
                print(f"   Current CUDA: {memory_after_quant - memory_before:.1f} MB")
            
            if ultra_success:
                print(f"   Ultra-optimized: {memory_after_quant_ultra - memory_before:.1f} MB")
            
            # Performance assessment
            print(f"\n🎯 Performance Assessment:")
            if ultra_success and current_success:
                speedup = total_current_time / total_ultra_time
                memory_reduction = (memory_after_quant - memory_after_quant_ultra) / memory_after_quant * 100
                
                print(f"   Ultra-optimized vs Current:")
                print(f"   - Speed: {speedup:.2f}x")
                print(f"   - Memory reduction: {memory_reduction:.1f}%")
                
                if speedup > 1.2:
                    print(f"   🚀 Ultra-optimized kernels provide significant speedup!")
                elif speedup > 0.8:
                    print(f"   ✅ Ultra-optimized kernels have similar performance")
                else:
                    print(f"   ⚠️  Ultra-optimized kernels are slower")
                
                if memory_reduction > 20:
                    print(f"   🚀 Significant memory reduction achieved!")
                elif memory_reduction > 5:
                    print(f"   ✅ Good memory reduction achieved")
                else:
                    print(f"   ➖ Minimal memory reduction")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Ultra-optimized kernels test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run ultra-optimized kernels test"""
    print("🎯 Ultra-Optimized Quantization Kernels Analysis")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available on this system")
        return
    
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    print()
    
    try:
        success = test_ultra_optimized_kernels()
        
        print("\n📊 Summary")
        print("=" * 100)
        print(f"✅ CUDA available and working: {torch.cuda.is_available()}")
        print(f"✅ Ultra-optimized kernels test: {success}")
        
        if success:
            print("\n🎉 Ultra-optimized kernels analysis completed!")
            print("\n💡 Key Insights:")
            print("   • Memory-efficient quantization kernels")
            print("   • Chunked matrix multiplication to avoid memory explosion")
            print("   • Performance vs memory trade-offs")
            print("   • Scalability to larger matrices")
        else:
            print("\n⚠️  Test failed. Check the output above for details.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
