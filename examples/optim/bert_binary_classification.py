"""
BERT Binary Classification with E8 Lattice Quantization

This example demonstrates how to use E8 lattice quantization for a BERT-based
text binary classification task. The example uses pre-trained BERT embeddings
and applies E8 quantization to the MLP layers while keeping the final output
layer unquantized (as it has only one output).

Key Features:
- Pre-trained BERT embeddings (frozen)
- E8-quantized MLP layers for feature processing
- Unquantized final linear layer (single output)
- Quantization-Aware Training (QAT)
- Performance comparison: Standard vs Quantized
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import E8 optimized modules
from coset.optim.e8 import (
    E8Config, 
    E8QLinear, 
    batch_e8_quantize,
    e8_cuda_available
)
from coset.lattices import E8Lattice


class TextDataset(Dataset):
    """Dataset for text binary classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and encode
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTBinaryClassifier(nn.Module):
    """Standard BERT-based binary classifier."""
    
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=512, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single output for binary classification
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings (frozen)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        
        # Apply MLP classifier
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return x.squeeze(-1)


class QuantizedBERTBinaryClassifier(nn.Module):
    """E8-quantized BERT-based binary classifier."""
    
    def __init__(self, config: E8Config, bert_model_name='bert-base-uncased', 
                 hidden_dim=512, dropout=0.3, use_cuda_kernel=False, scale_factor=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.config = config
        self.use_cuda_kernel = use_cuda_kernel and torch.cuda.is_available() and e8_cuda_available()
        self.lattice = E8Lattice(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Ensure hidden dimensions are divisible by 8 for E8 quantization
        if hidden_dim % 8 != 0:
            hidden_dim = ((hidden_dim + 7) // 8) * 8
        if (hidden_dim // 2) % 8 != 0:
            hidden_dim = ((hidden_dim + 7) // 8) * 8
        
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # E8-quantized MLP layers
        self.fc1 = E8QLinear(
            self.bert.config.hidden_size, hidden_dim, config,
            lattice=self.lattice,
            quantize_weights=True,
            quantize_every=1,
            scale_factor=scale_factor
        )
        self.fc2 = E8QLinear(
            hidden_dim, hidden_dim // 2, config,
            lattice=self.lattice,
            quantize_weights=True,
            quantize_every=1,
            scale_factor=scale_factor
        )
        
        # Standard output layer (not quantized - single output)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings (frozen)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        
        # Apply E8-quantized MLP classifier
        x = self.dropout(pooled_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Unquantized final layer
        
        return x.squeeze(-1)
    
    def get_implementation_type(self):
        """Get the implementation type for this model."""
        if self.use_cuda_kernel:
            return "CUDA-Accelerated"
        elif torch.cuda.is_available():
            return "GPU (PyTorch)"
        else:
            return "CPU"


def create_sample_data(num_samples=2000):
    """Create sample text data for binary classification."""
    # Sample positive and negative texts
    positive_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Amazing product, highly recommend to everyone!",
        "Great service, very satisfied with the purchase.",
        "Excellent quality, exceeded my expectations.",
        "Outstanding performance, worth every penny.",
        "Wonderful experience, will definitely buy again.",
        "Perfect solution for my needs, very happy.",
        "Incredible value, best purchase I've made.",
        "Superb quality, exactly what I wanted.",
        "Fantastic customer service, very professional."
    ] * (num_samples // 20)
    
    negative_texts = [
        "Terrible movie, complete waste of time and money.",
        "Poor quality product, would not recommend.",
        "Awful service, very disappointed with the purchase.",
        "Bad experience, did not meet expectations.",
        "Worst product I've ever bought, avoid at all costs.",
        "Horrible customer service, very unprofessional.",
        "Disappointing quality, not worth the money.",
        "Regret buying this, complete failure.",
        "Useless product, money down the drain.",
        "Terrible experience, will never buy again."
    ] * (num_samples // 20)
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, labels


def train_model(model, train_loader, test_loader, epochs=3, device='cpu'):
    """Train a BERT binary classification model."""
    model.to(device)
    model.train()
    
    # Freeze BERT parameters
    for param in model.bert.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'epoch_times': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Test accuracy
        test_acc = evaluate_model(model, test_loader, device)
        
        epoch_time = time.time() - start_time
        
        history['train_losses'].append(avg_loss)
        history['train_accuracies'].append(train_acc)
        history['test_accuracies'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")
    
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    model.train()
    return 100.0 * correct / total


def benchmark_forward_pass(model, test_loader, device, num_batches=20):
    """Benchmark forward pass performance."""
    model.eval()
    model.to(device)
    
    times = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Warmup
            if i == 0:
                _ = model(input_ids, attention_mask)
                torch.cuda.synchronize() if device == 'cuda' else None
            
            # Benchmark
            start_time = time.time()
            _ = model(input_ids, attention_mask)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(batch_time)
    
    model.train()
    return np.mean(times)


def main():
    """Main function to run BERT binary classification with E8 quantization."""
    print("BERT Binary Classification with E8 Lattice Quantization")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create sample data
    print("\nCreating sample data...")
    texts, labels = create_sample_data(num_samples=20000)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Initialize tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # E8 Quantization configuration
    config = E8Config(
        q=4,
        M=2,
        beta=0.3,
        alpha=1.0,
        disable_overload_protection=True
    )
    
    print(f"\nQuantization Config: {config}")
    print(f"Compression ratio: {config.get_compression_ratio():.2f}x")
    
    # Create models
    print("\nCreating models...")
    standard_model = BERTBinaryClassifier(hidden_dim=512)
    quantized_model = QuantizedBERTBinaryClassifier(
        config=config, 
        hidden_dim=512, 
        use_cuda_kernel=False
    )
    
    if torch.cuda.is_available():
        cuda_quantized_model = QuantizedBERTBinaryClassifier(
            config=config, 
            hidden_dim=512, 
            use_cuda_kernel=True
        )
    else:
        cuda_quantized_model = None
    
    print(f"Standard Model: {sum(p.numel() for p in standard_model.parameters()):,} parameters")
    print(f"Quantized Model: {sum(p.numel() for p in quantized_model.parameters()):,} parameters")
    if cuda_quantized_model:
        print(f"CUDA Quantized Model: {sum(p.numel() for p in cuda_quantized_model.parameters()):,} parameters")
    
    # Check CUDA availability
    cuda_available = e8_cuda_available()
    print(f"\nCUDA Kernel Availability: {'Yes' if cuda_available else 'No'}")
    if cuda_quantized_model:
        print(f"CUDA Model Implementation: {cuda_quantized_model.get_implementation_type()}")
    
    # Benchmark forward pass performance
    print("\n" + "="*70)
    print("FORWARD PASS PERFORMANCE BENCHMARK")
    print("="*70)
    
    print("Benchmarking Standard Model...")
    standard_time = benchmark_forward_pass(standard_model, test_loader, device, num_batches=20)
    print(f"Standard Forward Pass: {standard_time:.3f} ms/batch")
    
    print("Benchmarking Quantized Model...")
    quantized_time = benchmark_forward_pass(quantized_model, test_loader, device, num_batches=20)
    print(f"Quantized Forward Pass: {quantized_time:.3f} ms/batch")
    
    speedup = standard_time / quantized_time
    print(f"Quantized Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than Standard")
    
    if cuda_quantized_model and torch.cuda.is_available():
        print("Benchmarking CUDA-Accelerated Model...")
        cuda_time = benchmark_forward_pass(cuda_quantized_model, test_loader, device, num_batches=20)
        print(f"CUDA Forward Pass: {cuda_time:.3f} ms/batch")
        
        cuda_speedup = quantized_time / cuda_time
        print(f"CUDA Speedup: {cuda_speedup:.2f}x faster than Quantized")
    
    # Train models
    print("\n" + "="*70)
    print("QUANTIZATION-AWARE TRAINING")
    print("="*70)
    
    print("\nTraining Standard Model...")
    standard_history = train_model(standard_model, train_loader, test_loader, epochs=3, device=device)
    
    print("\nTraining Quantized Model...")
    quantized_history = train_model(quantized_model, train_loader, test_loader, epochs=3, device=device)
    
    if cuda_quantized_model and torch.cuda.is_available():
        print("\nTraining CUDA-Accelerated Model...")
        cuda_history = train_model(cuda_quantized_model, train_loader, test_loader, epochs=3, device=device)
    else:
        cuda_history = None
    
    # Results comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"Standard Model - Final Test Accuracy: {standard_history['test_accuracies'][-1]:.2f}%")
    print(f"Quantized Model - Final Test Accuracy: {quantized_history['test_accuracies'][-1]:.2f}%")
    if cuda_history:
        print(f"CUDA Quantized Model - Final Test Accuracy: {cuda_history['test_accuracies'][-1]:.2f}%")
    
    # Accuracy drop analysis
    quantized_accuracy_drop = standard_history['test_accuracies'][-1] - quantized_history['test_accuracies'][-1]
    print(f"\nAccuracy Drop (Quantized vs Standard): {quantized_accuracy_drop:.2f}%")
    
    if cuda_history:
        cuda_accuracy_drop = standard_history['test_accuracies'][-1] - cuda_history['test_accuracies'][-1]
        print(f"Accuracy Drop (CUDA Quantized vs Standard): {cuda_accuracy_drop:.2f}%")
    
    # Performance summary
    print(f"\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Standard Forward Pass: {standard_time:.3f} ms/batch")
    print(f"Quantized Forward Pass: {quantized_time:.3f} ms/batch")
    print(f"Quantized Speedup: {speedup:.2f}x")
    if cuda_quantized_model and torch.cuda.is_available():
        print(f"CUDA Forward Pass: {cuda_time:.3f} ms/batch")
        print(f"CUDA Speedup: {cuda_speedup:.2f}x")
    
    print(f"\nFinal Accuracies:")
    print(f"  Standard Model: {standard_history['test_accuracies'][-1]:.2f}%")
    print(f"  Quantized Model: {quantized_history['test_accuracies'][-1]:.2f}%")
    if cuda_history:
        print(f"  CUDA Quantized Model: {cuda_history['test_accuracies'][-1]:.2f}%")
    
    # Quantization statistics
    print(f"\nQuantization Statistics (Quantized Model):")
    for name, module in quantized_model.named_modules():
        if isinstance(module, E8QLinear):
            stats = module.get_quantization_stats()
            print(f"  {name}: {stats['step_count']} quantization steps")
    
    print("\nBERT Binary Classification with E8 Quantization Complete!")
    print("\n" + "="*70)
    print("IMPLEMENTATION SUMMARY")
    print("="*70)
    print("✓ BERT embeddings: Frozen pre-trained features")
    print("✓ MLP layers: E8-quantized for compression")
    print("✓ Output layer: Unquantized (single output)")
    print("✓ QAT: Quantization-Aware Training enabled")
    if cuda_available:
        print("✓ CUDA: Custom kernels available (with compilation issues)")
    else:
        print("✗ CUDA: Not available")
    print("="*70)


if __name__ == "__main__":
    main()
