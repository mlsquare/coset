"""
MNIST Quantization-Aware Training Example

This example demonstrates how to use the coset library for quantization-aware
training on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from coset import QLinear, QuantizationConfig, D4Lattice


class QuantizedMNIST(nn.Module):
    """Quantized MNIST model using hierarchical nested-lattice quantization."""
    
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.lattice = D4Lattice()
        
        # Quantized layers - smaller network
        self.fc1 = QLinear(
            784, 32, config, 
            lattice=self.lattice,
            quantize_weights=True,
            quantize_every=1
        )
        self.fc2 = QLinear(
            32, 16, config,
            lattice=self.lattice, 
            quantize_weights=True,
            quantize_every=1
        )
        
        # Standard output layer (not quantized)
        self.fc3 = nn.Linear(16, 10)
        
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


class StandardMNIST(nn.Module):
    """Standard MNIST model for comparison."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
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


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """Train a model and return training history."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
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
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
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
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }


def main():
    """Main training function."""
    print("MNIST Quantization-Aware Training Example")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use smaller subset of training data for faster execution
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Create smaller training subset (10% of original data)
    train_size = len(train_dataset) // 10
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Configuration
    config = QuantizationConfig(
        lattice_type="D4",
        q=4,
        M=2,
        beta=1.0,
        alpha=1.0,
        with_tie_dither=True,
        with_dither=False
    )
    
    print(f"\nQuantization Config: {config}")
    print(f"Compression ratio: {config.get_compression_ratio():.2f}x")
    
    # Train quantized model
    print("\nTraining Quantized Model...")
    quantized_model = QuantizedMNIST(config)
    quantized_history = train_model(quantized_model, train_loader, test_loader, epochs=3)
    
    # Train standard model for comparison
    print("\nTraining Standard Model...")
    standard_model = StandardMNIST()
    standard_history = train_model(standard_model, train_loader, test_loader, epochs=3)
    
    # Results comparison
    print("\nResults Comparison:")
    print("=" * 50)
    print(f"Quantized Model - Final Test Accuracy: {quantized_history['test_accuracies'][-1]:.2f}%")
    print(f"Standard Model - Final Test Accuracy: {standard_history['test_accuracies'][-1]:.2f}%")
    print(f"Accuracy Drop: {standard_history['test_accuracies'][-1] - quantized_history['test_accuracies'][-1]:.2f}%")
    
    # Model size comparison
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    quantized_params = count_parameters(quantized_model)
    standard_params = count_parameters(standard_model)
    
    print(f"\nModel Size Comparison:")
    print(f"Quantized Model Parameters: {quantized_params:,}")
    print(f"Standard Model Parameters: {standard_params:,}")
    print(f"Parameter Reduction: {(1 - quantized_params/standard_params)*100:.1f}%")
    
    # Quantization statistics
    print(f"\nQuantization Statistics:")
    for name, module in quantized_model.named_modules():
        if isinstance(module, QLinear):
            stats = module.get_quantization_stats()
            print(f"{name}: {stats['step_count']} quantization steps")


if __name__ == "__main__":
    main()
