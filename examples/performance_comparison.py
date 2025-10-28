#!/usr/bin/env python3
"""
Performance comparison between baseline and HNLQ models with different batch sizes and data fractions.
Tests scalability and performance characteristics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import argparse

# Import coset core modules
try:
    from coset.core import HNLQLinear
    from coset.core.base import LatticeConfig
    from coset.core.e8 import E8Lattice, create_e8_hnlq_linear
    print("✓ Successfully imported coset core modules")
except ImportError as e:
    print(f"✗ Failed to import coset core modules: {e}")
    exit(1)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class BaselineMLP(nn.Module):
    """Baseline MLP without quantization."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
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

class HNLQMLP(nn.Module):
    """MLP with HNLQ quantization."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10, 
                 lattice_type="E8", q=4, M=2):
        super().__init__()
        
        self.config = LatticeConfig(
            lattice_type=lattice_type,
            q=q,
            M=M,
            beta=1.0,
            alpha=1.0,
            disable_scaling=True,
            disable_overload_protection=True,
            with_tie_dither=True,
            with_dither=False,
            max_scaling_iterations=10
        )
        
        # For E8 lattice, input dimension must be divisible by 8
        self.input_padding = 0
        if lattice_type == "E8" and input_size % 8 != 0:
            self.input_padding = 8 - (input_size % 8)
            input_size += self.input_padding
        
        if lattice_type == "E8":
            self.fc1 = create_e8_hnlq_linear(
                in_dim=input_size,
                out_dim=hidden_size,
                device=device,
                q=q,
                M=M,
                Delta0=1.5
            )
            self.fc2 = create_e8_hnlq_linear(
                in_dim=hidden_size,
                out_dim=hidden_size // 2,
                device=device,
                q=q,
                M=M,
                Delta0=1.5
            )
        else:
            raise NotImplementedError(f"Lattice type {lattice_type} not yet implemented")
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        if self.input_padding > 0:
            x = torch.cat([x, torch.zeros(x.size(0), self.input_padding, device=x.device)], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_mnist_data(batch_size=64, data_fraction=0.1):
    """Load MNIST dataset with specified fraction."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_size = int(len(train_dataset) * data_fraction)
    test_size = int(len(test_dataset) * data_fraction)
    
    train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
    test_subset = torch.utils.data.Subset(test_dataset, range(test_size))
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=3, lr=0.001, model_name="Model"):
    """Train a model and return performance metrics."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name} for {epochs} epochs...")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    total_train_time = 0
    final_train_acc = 0
    final_test_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:  # Less frequent updates for larger datasets
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        final_train_acc = train_acc
        final_test_acc = test_acc
        
        print(f'  Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    avg_time_per_epoch = total_train_time / epochs
    print(f'  Average time per epoch: {avg_time_per_epoch:.2f}s')
    
    return {
        'total_time': total_train_time,
        'avg_time_per_epoch': avg_time_per_epoch,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'parameters': sum(p.numel() for p in model.parameters())
    }

def run_comparison(batch_sizes=[64, 128, 256], data_fractions=[0.1, 0.5], epochs=3):
    """Run comprehensive performance comparison."""
    print("=" * 80)
    print("PERFORMANCE COMPARISON: Baseline vs HNLQ")
    print("=" * 80)
    
    results = {}
    
    for data_fraction in data_fractions:
        print(f"\n{'='*60}")
        print(f"DATA FRACTION: {data_fraction*100}%")
        print(f"{'='*60}")
        
        for batch_size in batch_sizes:
            print(f"\n{'-'*40}")
            print(f"BATCH SIZE: {batch_size}")
            print(f"{'-'*40}")
            
            # Load data
            train_loader, test_loader = load_mnist_data(
                batch_size=batch_size, 
                data_fraction=data_fraction
            )
            
            # Test Baseline Model
            baseline_model = BaselineMLP()
            baseline_results = train_model(
                baseline_model, train_loader, test_loader, 
                epochs=epochs, model_name="Baseline"
            )
            
            # Test HNLQ Model
            hnlq_model = HNLQMLP()
            hnlq_results = train_model(
                hnlq_model, train_loader, test_loader, 
                epochs=epochs, model_name="HNLQ"
            )
            
            # Store results
            key = f"data_{data_fraction}_batch_{batch_size}"
            results[key] = {
                'baseline': baseline_results,
                'hnlq': hnlq_results,
                'speedup_ratio': hnlq_results['avg_time_per_epoch'] / baseline_results['avg_time_per_epoch'],
                'accuracy_ratio': hnlq_results['final_test_acc'] / baseline_results['final_test_acc']
            }
            
            # Print comparison
            print(f"\n📊 COMPARISON (Data: {data_fraction*100}%, Batch: {batch_size}):")
            print(f"  Baseline: {baseline_results['avg_time_per_epoch']:.2f}s/epoch, {baseline_results['final_test_acc']:.2f}% acc")
            print(f"  HNLQ:     {hnlq_results['avg_time_per_epoch']:.2f}s/epoch, {hnlq_results['final_test_acc']:.2f}% acc")
            print(f"  Speedup:  {results[key]['speedup_ratio']:.2f}x slower")
            print(f"  Accuracy: {results[key]['accuracy_ratio']:.2f}x of baseline")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"{'Data%':<8} {'Batch':<6} {'Baseline':<20} {'HNLQ':<20} {'Speedup':<8} {'Acc Ratio':<10}")
    print(f"{'-'*80}")
    
    for key, result in results.items():
        data_pct = float(key.split('_')[1]) * 100
        batch_size = int(key.split('_')[3])
        baseline = result['baseline']
        hnlq = result['hnlq']
        
        print(f"{data_pct:>6.0f}% {batch_size:>4}   "
              f"{baseline['avg_time_per_epoch']:>6.2f}s, {baseline['final_test_acc']:>6.2f}%   "
              f"{hnlq['avg_time_per_epoch']:>6.2f}s, {hnlq['final_test_acc']:>6.2f}%   "
              f"{result['speedup_ratio']:>6.2f}x   {result['accuracy_ratio']:>8.2f}x")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Performance comparison between baseline and HNLQ models')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[64, 128, 256],
                        help='Batch sizes to test')
    parser.add_argument('--data-fractions', nargs='+', type=float, default=[0.1, 0.5],
                        help='Data fractions to test')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train')
    
    args = parser.parse_args()
    
    run_comparison(
        batch_sizes=args.batch_sizes,
        data_fractions=args.data_fractions,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
