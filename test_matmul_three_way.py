#!/usr/bin/env python3
"""
Three-way comparison: CoSet matmul with CUDA kernels vs without vs PyTorch matmul
"""

import torch
import time
from coset import LatticeConfig, LatticeType, LatticeQuantizer
from coset.layers.autograd import quantized_matmul

def test_three_way_matmul():
    """Test three-way matmul comparison"""
    print("🚀 Three-Way MatMul Comparison Test")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    try:
        config = LatticeConfig(type=LatticeType.E8, radix=2, num_layers=1)
        
        # Test configurations
        test_configs = [
            (2, 16, 8),    # Small
            (4, 32, 16),   # Medium
            (8, 64, 32),   # Large
            (16, 128, 64), # Very large
        ]
        
        print(f"🔧 Configuration: E8, radix=2, layers=1, lattice_dim=8")
        print()
        
        for batch_size, input_dim, out_dim in test_configs:
            print(f"📏 Testing {batch_size}x{input_dim} -> {out_dim}")
            print("-" * 60)
            
            # Create test data
            x = torch.randn(batch_size, input_dim, device=device)
            weight = torch.randn(out_dim, input_dim, device=device)
            bias = torch.randn(out_dim, device=device)
            
            # Test 1: PyTorch Standard MatMul
            print("1️⃣ PyTorch Standard MatMul:")
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                pytorch_output = torch.matmul(x, weight.t()) + bias
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start) / 100 * 1000
            
            print(f"   Time: {pytorch_time:.3f}ms")
            print(f"   Output shape: {pytorch_output.shape}")
            
            # Test 2: CoSet MatMul with CUDA kernels
            print("\n2️⃣ CoSet MatMul with CUDA Kernels:")
            
            # Create quantizer with CUDA kernels enabled
            quantizer_cuda = LatticeQuantizer(config, use_cuda_kernels=True).to(device)
            
            # Quantize to get indices
            _, input_indices = quantizer_cuda.quantize_to_depth(x, 0)
            _, weight_indices = quantizer_cuda.quantize_to_depth(weight, 0)
            lookup_table = quantizer_cuda.create_lookup_table()
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                coset_cuda_output = quantized_matmul(input_indices, weight_indices, lookup_table, bias)
            torch.cuda.synchronize()
            coset_cuda_time = (time.time() - start) / 100 * 1000
            
            print(f"   Time: {coset_cuda_time:.3f}ms")
            print(f"   Output shape: {coset_cuda_output.shape}")
            print(f"   CUDA kernels used: {quantizer_cuda.use_cuda_kernels}")
            
            # Test 3: CoSet MatMul without CUDA kernels
            print("\n3️⃣ CoSet MatMul without CUDA Kernels:")
            
            # Create quantizer with CUDA kernels disabled
            quantizer_no_cuda = LatticeQuantizer(config, use_cuda_kernels=False).to(device)
            
            # Quantize to get indices
            _, input_indices = quantizer_no_cuda.quantize_to_depth(x, 0)
            _, weight_indices = quantizer_no_cuda.quantize_to_depth(weight, 0)
            lookup_table = quantizer_no_cuda.create_lookup_table()
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                coset_no_cuda_output = quantized_matmul(input_indices, weight_indices, lookup_table, bias)
            torch.cuda.synchronize()
            coset_no_cuda_time = (time.time() - start) / 100 * 1000
            
            print(f"   Time: {coset_no_cuda_time:.3f}ms")
            print(f"   Output shape: {coset_no_cuda_output.shape}")
            print(f"   CUDA kernels used: {quantizer_no_cuda.use_cuda_kernels}")
            
            # Performance comparison
            print(f"\n📊 Performance Comparison:")
            print(f"   PyTorch standard: {pytorch_time:.3f}ms")
            print(f"   CoSet with CUDA:  {coset_cuda_time:.3f}ms")
            print(f"   CoSet no CUDA:    {coset_no_cuda_time:.3f}ms")
            
            # Calculate speedups
            pytorch_vs_cuda = coset_cuda_time / pytorch_time
            pytorch_vs_no_cuda = coset_no_cuda_time / pytorch_time
            cuda_vs_no_cuda = coset_cuda_time / coset_no_cuda_time
            
            print(f"\n📈 Speedup Analysis:")
            print(f"   PyTorch vs CoSet CUDA: {pytorch_vs_cuda:.1f}x slower")
            print(f"   PyTorch vs CoSet no CUDA: {pytorch_vs_no_cuda:.1f}x slower")
            print(f"   CoSet CUDA vs no CUDA: {cuda_vs_no_cuda:.1f}x")
            
            # Performance assessment
            print(f"\n🎯 Performance Assessment:")
            if pytorch_vs_cuda < 10:
                print(f"   🚀 CoSet with CUDA: Excellent performance!")
            elif pytorch_vs_cuda < 100:
                print(f"   ✅ CoSet with CUDA: Good performance")
            elif pytorch_vs_cuda < 1000:
                print(f"   ⚠️  CoSet with CUDA: Acceptable performance")
            else:
                print(f"   ❌ CoSet with CUDA: Poor performance")
            
            if pytorch_vs_no_cuda < 10:
                print(f"   🚀 CoSet without CUDA: Excellent performance!")
            elif pytorch_vs_no_cuda < 100:
                print(f"   ✅ CoSet without CUDA: Good performance")
            elif pytorch_vs_no_cuda < 1000:
                print(f"   ⚠️  CoSet without CUDA: Acceptable performance")
            else:
                print(f"   ❌ CoSet without CUDA: Poor performance")
            
            if cuda_vs_no_cuda > 1.2:
                print(f"   ✅ CUDA kernels provide speedup")
            elif cuda_vs_no_cuda < 0.8:
                print(f"   ⚠️  CUDA kernels are slower (fallback better)")
            else:
                print(f"   ➖ CUDA kernels have minimal impact")
            
            # Verify numerical accuracy
            mse_cuda = torch.mean((coset_cuda_output - coset_no_cuda_output) ** 2).item()
            mse_pytorch_cuda = torch.mean((pytorch_output - coset_cuda_output) ** 2).item()
            mse_pytorch_no_cuda = torch.mean((pytorch_output - coset_no_cuda_output) ** 2).item()
            
            print(f"\n🔍 Numerical Accuracy:")
            print(f"   MSE (CoSet CUDA vs no CUDA): {mse_cuda:.6f}")
            print(f"   MSE (PyTorch vs CoSet CUDA): {mse_pytorch_cuda:.6f}")
            print(f"   MSE (PyTorch vs CoSet no CUDA): {mse_pytorch_no_cuda:.6f}")
            
            if mse_cuda < 1e-6:
                print(f"   ✅ CoSet CUDA vs no CUDA: Perfect accuracy")
            else:
                print(f"   ⚠️  CoSet CUDA vs no CUDA: Accuracy degraded")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Three-way matmul test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantization_overhead():
    """Test quantization overhead separately"""
    print("\n🔍 Quantization Overhead Analysis")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    try:
        config = LatticeConfig(type=LatticeType.E8, radix=2, num_layers=1)
        
        # Test configurations
        test_configs = [
            (2, 16),    # Small
            (4, 32),    # Medium
            (8, 64),    # Large
        ]
        
        for batch_size, input_dim in test_configs:
            print(f"📏 Testing {batch_size}x{input_dim}")
            print("-" * 40)
            
            # Create test data
            x = torch.randn(batch_size, input_dim, device=device)
            weight = torch.randn(32, input_dim, device=device)
            
            # Test quantization overhead
            quantizer_cuda = LatticeQuantizer(config, use_cuda_kernels=True).to(device)
            quantizer_no_cuda = LatticeQuantizer(config, use_cuda_kernels=False).to(device)
            
            # Input quantization
            print("1️⃣ Input Quantization:")
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(50):
                _, input_indices_cuda = quantizer_cuda.quantize_to_depth(x, 0)
            torch.cuda.synchronize()
            input_quant_cuda_time = (time.time() - start) / 50 * 1000
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(50):
                _, input_indices_no_cuda = quantizer_no_cuda.quantize_to_depth(x, 0)
            torch.cuda.synchronize()
            input_quant_no_cuda_time = (time.time() - start) / 50 * 1000
            
            print(f"   With CUDA kernels: {input_quant_cuda_time:.3f}ms")
            print(f"   Without CUDA kernels: {input_quant_no_cuda_time:.3f}ms")
            
            # Weight quantization (one-time cost)
            print("\n2️⃣ Weight Quantization:")
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _, weight_indices_cuda = quantizer_cuda.quantize_to_depth(weight, 0)
            torch.cuda.synchronize()
            weight_quant_cuda_time = (time.time() - start) / 10 * 1000
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _, weight_indices_no_cuda = quantizer_no_cuda.quantize_to_depth(weight, 0)
            torch.cuda.synchronize()
            weight_quant_no_cuda_time = (time.time() - start) / 10 * 1000
            
            print(f"   With CUDA kernels: {weight_quant_cuda_time:.3f}ms")
            print(f"   Without CUDA kernels: {weight_quant_no_cuda_time:.3f}ms")
            
            # Lookup table creation
            print("\n3️⃣ Lookup Table Creation:")
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(50):
                lookup_table = quantizer_cuda.create_lookup_table()
            torch.cuda.synchronize()
            lut_time = (time.time() - start) / 50 * 1000
            
            print(f"   Time: {lut_time:.3f}ms")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization overhead test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run three-way matmul comparison"""
    print("🎯 Three-Way MatMul Performance Analysis")
    print("=" * 100)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available on this system")
        return
    
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    print()
    
    # Run tests
    try:
        # 1. Three-way matmul comparison
        matmul_success = test_three_way_matmul()
        
        # 2. Quantization overhead analysis
        overhead_success = test_quantization_overhead()
        
        # Summary
        print("\n📊 Summary")
        print("=" * 100)
        print(f"✅ CUDA available and working: {torch.cuda.is_available()}")
        print(f"✅ Three-way matmul test: {matmul_success}")
        print(f"✅ Quantization overhead test: {overhead_success}")
        
        if all([matmul_success, overhead_success]):
            print("\n🎉 Three-way matmul comparison completed!")
            print("\n💡 Key Insights:")
            print("   • Compare PyTorch vs CoSet with/without CUDA kernels")
            print("   • Analyze quantization overhead separately")
            print("   • Determine if CUDA kernels provide benefit")
            print("   • Guide decision on proper CUDA kernel implementation")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
