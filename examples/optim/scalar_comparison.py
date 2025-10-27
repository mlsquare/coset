#!/usr/bin/env python3
"""
Scalar Quantization Comparison Example

This example demonstrates different scalar quantization modes on MNIST:
- Standard (no quantization)
- Scalar 4-bit symmetric
- Scalar 4-bit asymmetric
- Scalar 8-bit symmetric
- E8 quantization (for reference)

Shows accuracy and performance trade-offs between different quantization methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from typing import Dict, List, Tuple

# Import COSET modules
from coset.optim import (
    ScalarConfig, get_scalar_config,
    SCALAR_INT4_SYM, SCALAR_INT8_SYM, SCALAR_INT4_ASYM, SCALAR_INT8_ASYM,
    ScalarQLinear, ScalarStraightThroughQuantize,
    E8Config, E8QLinear, E8StraightThroughQuantize
)


class StandardMLP(nn.Module):
    """Standard MLP without quantization."""
    
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class QuantizedMLP(nn.Module):
    """Quantized MLP using ScalarQLinear."""
    
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, 
                 config: ScalarConfig = None, scale_factor=None):
        super().__init__()
        self.config = config
        
        # Ensure hidden dimensions are divisible by 8 for E8 compatibility
        if hidden_size % 8 != 0:
            hidden_size = ((hidden_size + 7) // 8) * 8
        
        self.fc1 = ScalarQLinear(
            input_size, hidden_size, config,
            quantize_weights=True,
            quantize_every=1,
            scale_factor=scale_factor
        )
        self.fc2 = ScalarQLinear(
            hidden_size, hidden_size // 2, config,
            quantize_weights=True,
            quantize_every=1,
            scale_factor=scale_factor
        )
        # Final layer - use standard linear (single output, no quantization needed)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class E8QuantizedMLP(nn.Module):
    """E8 quantized MLP for comparison."""
    
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, 
                 config: E8Config = None, scale_factor=None):
        super().__init__()
        self.config = config
        
        # Ensure hidden dimensions are divisible by 8 for E8
        if hidden_size % 8 != 0:
            hidden_size = ((hidden_size + 7) // 8) * 8
        
        self.fc1 = E8QLinear(
            input_size, hidden_size, config,
            quantize_weights=True,
            quantize_every=1,
            scale_factor=scale_factor
        )
        self.fc2 = E8QLinear(
            hidden_size, hidden_size // 2, config,
            quantize_weights=True,
            quantize_every=1,
            scale_factor=scale_factor
        )
        # Final layer - use standard linear (single output, no quantization needed)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device='cpu'):
    """Train a model and return training history."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100.0 * correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_accuracies


def benchmark_model(model, test_loader, device='cpu', num_runs=5):
    """Benchmark model inference speed."""
    model = model.to(device)
    model.eval()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for data, _ in test_loader:
                data = data.to(device)
                _ = model(data)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def main():
    """Main function to run scalar quantization comparison."""
    print("=== Scalar Quantization Comparison on MNIST ===\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Define model configurations
    configs = {
        "Standard": {
            "model_class": StandardMLP,
            "config": None,
            "scale_factor": None,
            "description": "No quantization (baseline)"
        },
        "Scalar 4-bit Symmetric": {
            "model_class": QuantizedMLP,
            "config": SCALAR_INT4_SYM,
            "scale_factor": None,
            "description": "4-bit symmetric scalar quantization"
        },
        "Scalar 4-bit Asymmetric": {
            "model_class": QuantizedMLP,
            "config": SCALAR_INT4_ASYM,
            "scale_factor": None,
            "description": "4-bit asymmetric scalar quantization"
        },
        "Scalar 8-bit Symmetric": {
            "model_class": QuantizedMLP,
            "config": SCALAR_INT8_SYM,
            "scale_factor": None,
            "description": "8-bit symmetric scalar quantization"
        },
        "Scalar 8-bit Asymmetric": {
            "model_class": QuantizedMLP,
            "config": SCALAR_INT8_ASYM,
            "scale_factor": None,
            "description": "8-bit asymmetric scalar quantization"
        },
        "E8 Quantization": {
            "model_class": E8QuantizedMLP,
            "config": E8Config(q=4, M=2, beta=0.3),
            "scale_factor": None,
            "description": "E8 lattice quantization (reference)"
        }
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Create model
        if config["model_class"] == StandardMLP:
            model = config["model_class"]()
        else:
            model = config["model_class"](config=config["config"], scale_factor=config["scale_factor"])
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        start_time = time.time()
        train_losses, val_accuracies = train_model(model, train_loader, val_loader, epochs=5, device=device)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        final_accuracy = 100.0 * correct / total
        
        # Benchmark inference speed
        inference_time, inference_std = benchmark_model(model, test_loader, device=device)
        
        # Store results
        results[name] = {
            "final_accuracy": final_accuracy,
            "training_time": training_time,
            "inference_time": inference_time,
            "inference_std": inference_std,
            "config": config["config"],
            "description": config["description"]
        }
        
        print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Inference Time: {inference_time:.2f}s ± {inference_std:.2f}s")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("QUANTIZATION COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Accuracy':<10} {'Train Time':<12} {'Inference Time':<15} {'Description'}")
    print(f"{'-'*80}")
    
    for name, result in results.items():
        print(f"{name:<25} {result['final_accuracy']:<10.2f} {result['training_time']:<12.2f} {result['inference_time']:<15.2f} {result['description']}")
    
    # Print quantization statistics for quantized models
    print(f"\n{'='*80}")
    print("QUANTIZATION STATISTICS")
    print(f"{'='*80}")
    
    for name, result in results.items():
        if result["config"] is not None:
            print(f"\n{name}:")
            if hasattr(result["config"], 'to_dict'):
                config_dict = result["config"].to_dict()
                for key, value in config_dict.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  Config: {result['config']}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Find best accuracy
    best_accuracy = max(results.values(), key=lambda x: x["final_accuracy"])
    best_name = [name for name, result in results.items() if result == best_accuracy][0]
    print(f"Best Accuracy: {best_name} ({best_accuracy['final_accuracy']:.2f}%)")
    
    # Find fastest inference
    fastest_inference = min(results.values(), key=lambda x: x["inference_time"])
    fastest_name = [name for name, result in results.items() if result == fastest_inference][0]
    print(f"Fastest Inference: {fastest_name} ({fastest_inference['inference_time']:.2f}s)")
    
    # Compare scalar vs E8
    scalar_results = {k: v for k, v in results.items() if "Scalar" in k}
    e8_result = results.get("E8 Quantization")
    
    if scalar_results and e8_result:
        print(f"\nScalar vs E8 Comparison:")
        print(f"  E8 Accuracy: {e8_result['final_accuracy']:.2f}%")
        print(f"  E8 Inference Time: {e8_result['inference_time']:.2f}s")
        
        best_scalar = max(scalar_results.values(), key=lambda x: x["final_accuracy"])
        best_scalar_name = [name for name, result in scalar_results.items() if result == best_scalar][0]
        print(f"  Best Scalar: {best_scalar_name} ({best_scalar['final_accuracy']:.2f}%)")
        print(f"  Best Scalar Inference Time: {best_scalar['inference_time']:.2f}s")
        
        accuracy_diff = best_scalar['final_accuracy'] - e8_result['final_accuracy']
        time_diff = best_scalar['inference_time'] - e8_result['inference_time']
        print(f"  Accuracy Difference: {accuracy_diff:+.2f}%")
        print(f"  Time Difference: {time_diff:+.2f}s")


if __name__ == "__main__":
    main()
