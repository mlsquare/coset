"""
MNIST QAT CPU vs GPU Performance Comparison with E8 Lattice Quantization

This example demonstrates Quantization-Aware Training (QAT) using E8 lattice 
quantization on MNIST data with batch size 128, comparing CPU and GPU performance.

This is an E8-optimized example located in examples/optim/ to showcase the 
optimized E8 implementation from coset.optim.e8.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from coset.optim.e8 import (
    E8Config, 
    E8QLinear, 
    batch_e8_quantize,
    e8_cuda_available
)
from coset.lattices import E8Lattice
from coset.quant.functional import encode, decode, quantize


class QuantizedMNISTQAT(nn.Module):
    """Quantized MNIST model using E8 lattice quantization with QAT."""
    
    def __init__(self, config: E8Config, device='cpu', use_cuda_kernel=False):
        super().__init__()
        self.config = config
        self.device = device
        self.use_cuda_kernel = use_cuda_kernel and device == 'cuda' and e8_cuda_available()
        self.lattice = E8Lattice(device=torch.device(device))
        
        # Quantized layers using E8QLinear for optimized E8 QAT
        self.fc1 = E8QLinear(
            784, 128, config,
            lattice=self.lattice,
            quantize_weights=True,
            quantize_every=1
        )
        self.fc2 = E8QLinear(
            128, 64, config,
            lattice=self.lattice,
            quantize_weights=True,
            quantize_every=1
        )
        
        # Standard output layer (not quantized)
        self.fc3 = nn.Linear(64, 10)
        
        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_implementation_type(self):
        """Get the implementation type for this model."""
        if self.use_cuda_kernel:
            return "CUDA-Accelerated"
        elif self.device == 'cuda':
            return "GPU (PyTorch)"
        else:
            return "CPU"


class StandardMNIST(nn.Module):
    """Standard MNIST model for comparison."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cpu'):
    """Train a model and return training history with timing."""
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_accuracies.append(test_acc)
        
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'epoch_times': epoch_times
    }


def benchmark_forward_pass(model, data_loader, device, num_batches=20):
    """Benchmark forward pass performance."""
    model = model.to(device)
    model.eval()
    
    total_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            data = data.to(device)
            
            # Warmup
            if batch_idx == 0:
                _ = model(data)
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            # Time the forward pass
            start_time = time.perf_counter()
            _ = model(data)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += data.shape[0]
    
    avg_time_per_sample = total_time / total_samples * 1000  # ms
    return avg_time_per_sample


def main():
    """Main function to run CPU vs GPU vs CUDA-accelerated QAT comparison."""
    print("MNIST QAT Performance Comparison")
    print("CPU vs GPU (PyTorch) vs CUDA-Accelerated")
    print("E8 Lattice Quantization with Batch Size 128")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use subset of data for faster execution
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Create smaller training subset
    train_size = min(10000, len(train_dataset))  # Use 10,000 samples
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    # Data loaders with batch size 128
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: 128")
    
    # E8 Quantization configuration
    config = E8Config(
        q=4,
        M=2,
        beta=0.3,  # Use optimal E8 beta
        alpha=1.0,
        disable_overload_protection=True
    )
    
    print(f"\nQuantization Config: {config}")
    print(f"Compression ratio: {config.get_compression_ratio():.2f}x")
    
    # Create models
    cpu_quantized_model = QuantizedMNISTQAT(config, device='cpu', use_cuda_kernel=False)
    gpu_quantized_model = QuantizedMNISTQAT(config, device='cuda', use_cuda_kernel=False) if torch.cuda.is_available() else None
    cuda_quantized_model = QuantizedMNISTQAT(config, device='cuda', use_cuda_kernel=True) if torch.cuda.is_available() else None
    standard_model = StandardMNIST()
    
    print(f"\nModels created:")
    print(f"CPU Quantized Model: {sum(p.numel() for p in cpu_quantized_model.parameters()):,} parameters")
    if gpu_quantized_model:
        print(f"GPU Quantized Model: {sum(p.numel() for p in gpu_quantized_model.parameters()):,} parameters")
    if cuda_quantized_model:
        print(f"CUDA Quantized Model: {sum(p.numel() for p in cuda_quantized_model.parameters()):,} parameters")
    print(f"Standard Model: {sum(p.numel() for p in standard_model.parameters()):,} parameters")
    
    # Check CUDA availability
    cuda_available = e8_cuda_available()
    print(f"\nCUDA Kernel Availability: {'Yes' if cuda_available else 'No'}")
    if cuda_quantized_model:
        print(f"CUDA Model Implementation: {cuda_quantized_model.get_implementation_type()}")
    
    # Benchmark forward pass performance
    print("\n" + "="*70)
    print("FORWARD PASS PERFORMANCE BENCHMARK")
    print("="*70)
    
    print("Benchmarking CPU Quantized Model...")
    cpu_forward_time = benchmark_forward_pass(cpu_quantized_model, test_loader, 'cpu', num_batches=50)
    print(f"CPU Forward Pass: {cpu_forward_time:.3f} ms/sample")
    
    gpu_forward_time = None
    cuda_forward_time = None
    
    if gpu_quantized_model and torch.cuda.is_available():
        print("Benchmarking GPU Quantized Model (PyTorch)...")
        gpu_forward_time = benchmark_forward_pass(gpu_quantized_model, test_loader, 'cuda', num_batches=50)
        print(f"GPU Forward Pass: {gpu_forward_time:.3f} ms/sample")
        
        gpu_speedup = cpu_forward_time / gpu_forward_time
        print(f"GPU Speedup: {gpu_speedup:.2f}x faster than CPU")
    
    if cuda_quantized_model and torch.cuda.is_available():
        print("Benchmarking CUDA-Accelerated Model...")
        cuda_forward_time = benchmark_forward_pass(cuda_quantized_model, test_loader, 'cuda', num_batches=50)
        print(f"CUDA Forward Pass: {cuda_forward_time:.3f} ms/sample")
        
        if gpu_forward_time:
            cuda_speedup = gpu_forward_time / cuda_forward_time
            print(f"CUDA Speedup: {cuda_speedup:.2f}x faster than GPU PyTorch")
        
        cuda_cpu_speedup = cpu_forward_time / cuda_forward_time
        print(f"CUDA vs CPU Speedup: {cuda_cpu_speedup:.2f}x faster than CPU")
    
    # Train models
    print("\n" + "="*70)
    print("QUANTIZATION-AWARE TRAINING")
    print("="*70)
    
    print("\nTraining Standard Model (CPU)...")
    standard_history = train_model(standard_model, train_loader, test_loader, epochs=5, device='cpu')
    
    print("\nTraining Quantized Model (CPU)...")
    cpu_history = train_model(cpu_quantized_model, train_loader, test_loader, epochs=5, device='cpu')
    
    gpu_history = None
    cuda_history = None
    
    if gpu_quantized_model and torch.cuda.is_available():
        print("\nTraining Quantized Model (GPU - PyTorch)...")
        gpu_history = train_model(gpu_quantized_model, train_loader, test_loader, epochs=5, device='cuda')
    else:
        print("\nGPU training skipped (GPU not available)")
    
    if cuda_quantized_model and torch.cuda.is_available():
        print("\nTraining Quantized Model (CUDA-Accelerated)...")
        cuda_history = train_model(cuda_quantized_model, train_loader, test_loader, epochs=5, device='cuda')
    else:
        print("\nCUDA training skipped (CUDA not available)")
    
    # Results comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"Standard Model - Final Test Accuracy: {standard_history['test_accuracies'][-1]:.2f}%")
    print(f"CPU Quantized Model - Final Test Accuracy: {cpu_history['test_accuracies'][-1]:.2f}%")
    if gpu_history:
        print(f"GPU Quantized Model (PyTorch) - Final Test Accuracy: {gpu_history['test_accuracies'][-1]:.2f}%")
    if cuda_history:
        print(f"CUDA Quantized Model - Final Test Accuracy: {cuda_history['test_accuracies'][-1]:.2f}%")
    
    # Accuracy drop analysis
    cpu_accuracy_drop = standard_history['test_accuracies'][-1] - cpu_history['test_accuracies'][-1]
    print(f"\nAccuracy Drop (CPU Quantized vs Standard): {cpu_accuracy_drop:.2f}%")
    
    if gpu_history:
        gpu_accuracy_drop = standard_history['test_accuracies'][-1] - gpu_history['test_accuracies'][-1]
        print(f"Accuracy Drop (GPU Quantized vs Standard): {gpu_accuracy_drop:.2f}%")
    
    if cuda_history:
        cuda_accuracy_drop = standard_history['test_accuracies'][-1] - cuda_history['test_accuracies'][-1]
        print(f"Accuracy Drop (CUDA Quantized vs Standard): {cuda_accuracy_drop:.2f}%")
    
    # Training time comparison
    print(f"\nTraining Time Comparison (per epoch):")
    for epoch in range(len(standard_history['epoch_times'])):
        print(f"  Epoch {epoch}: Standard {standard_history['epoch_times'][epoch]:.2f}s, CPU Quantized {cpu_history['epoch_times'][epoch]:.2f}s", end="")
        if gpu_history:
            print(f", GPU Quantized {gpu_history['epoch_times'][epoch]:.2f}s", end="")
        if cuda_history:
            print(f", CUDA Quantized {cuda_history['epoch_times'][epoch]:.2f}s", end="")
        print()
    
    # Performance summary
    print(f"\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Batch Size: 128")
    print(f"CPU Forward Pass: {cpu_forward_time:.3f} ms/sample")
    if gpu_forward_time:
        print(f"GPU Forward Pass (PyTorch): {gpu_forward_time:.3f} ms/sample")
        print(f"GPU Speedup vs CPU: {cpu_forward_time/gpu_forward_time:.2f}x")
    if cuda_forward_time:
        print(f"CUDA Forward Pass: {cuda_forward_time:.3f} ms/sample")
        print(f"CUDA Speedup vs CPU: {cpu_forward_time/cuda_forward_time:.2f}x")
        if gpu_forward_time:
            print(f"CUDA Speedup vs GPU: {gpu_forward_time/cuda_forward_time:.2f}x")
    
    print(f"\nFinal Accuracies:")
    print(f"  Standard Model: {standard_history['test_accuracies'][-1]:.2f}%")
    print(f"  CPU Quantized: {cpu_history['test_accuracies'][-1]:.2f}%")
    if gpu_history:
        print(f"  GPU Quantized (PyTorch): {gpu_history['test_accuracies'][-1]:.2f}%")
    if cuda_history:
        print(f"  CUDA Quantized: {cuda_history['test_accuracies'][-1]:.2f}%")
    
    # Quantization statistics
    print(f"\nQuantization Statistics (CPU Model):")
    for name, module in cpu_quantized_model.named_modules():
        if isinstance(module, E8QLinear):
            stats = module.get_quantization_stats()
            print(f"  {name}: {stats['step_count']} quantization steps")
    
    print("\nQAT Benchmark Complete!")
    print("\n" + "="*70)
    print("IMPLEMENTATION SUMMARY")
    print("="*70)
    print("✓ CPU: PyTorch CPU implementation")
    print("✓ GPU (PyTorch): PyTorch GPU implementation with vectorized operations")
    if cuda_available:
        print("✓ CUDA-Accelerated: Custom CUDA kernels (with known accuracy issues)")
    else:
        print("✗ CUDA-Accelerated: Not available (CUDA kernels not compiled)")
    print("="*70)


if __name__ == "__main__":
    main()
