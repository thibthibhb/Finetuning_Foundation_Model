#!/usr/bin/env python3
"""
Debug script to investigate kappa=0.0 issue in CBraMod training.
This script systematically checks all potential causes of the model predicting only one class.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.metrics import cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, '/root/cbramod/CBraMod')

def test_model_architecture():
    """Test if the model architecture is correct for 5-class classification"""
    print("=== Testing Model Architecture ===")
    
    # Create a simple parameter object
    class SimpleParams:
        def __init__(self):
            self.num_of_classes = 5
            self.use_pretrained_weights = False
            self.cuda = 0
    
    params = SimpleParams()
    
    # Import and create model
    from cbramod.models import model_for_idun
    model = model_for_idun.Model(params)
    
    # Check model architecture
    print(f"Model classifier output features: {model.classifier.out_features}")
    print(f"Expected classes: {params.num_of_classes}")
    
    # Test forward pass
    batch_size = 4
    channels = 1
    seq_len = 30
    epoch_size = 200
    
    x = torch.randn(batch_size, channels, seq_len, epoch_size)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output logits sample: {output[0].tolist()}")
        
        # Check if predictions are diverse
        predictions = torch.argmax(output, dim=1)
        print(f"Predictions: {predictions.tolist()}")
        print(f"Unique predictions: {torch.unique(predictions).tolist()}")
    
    return model, x

def test_loss_function():
    """Test if the loss function is working correctly"""
    print("\n=== Testing Loss Function ===")
    
    # Create sample data
    batch_size = 8
    num_classes = 5
    
    # Create diverse predictions (not all same class)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])  # Mix of classes
    
    # Test CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"Target distribution: {Counter(targets.tolist())}")
    
    # Test with all same predictions (should give kappa=0)
    same_predictions = torch.zeros(batch_size, num_classes)
    same_predictions[:, 0] = 10  # All predict class 0
    
    pred_classes = torch.argmax(same_predictions, dim=1)
    kappa = cohen_kappa_score(targets.numpy(), pred_classes.numpy())
    
    print(f"When all predictions are class 0:")
    print(f"  Predicted classes: {pred_classes.tolist()}")
    print(f"  Kappa: {kappa:.4f}")
    
    return criterion

def test_gradient_flow(model, criterion):
    """Test if gradients are flowing properly through the model"""
    print("\n=== Testing Gradient Flow ===")
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 30, 200)
    targets = torch.tensor([0, 1, 2, 3])
    
    # Forward pass
    model.train()
    output = model(x)
    loss = criterion(output, targets)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    total_grad_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm > 0:
                has_gradients = True
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"  {name}: no gradient")
    
    print(f"Has gradients: {has_gradients}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    
    return has_gradients

def test_learning_rate():
    """Test if learning rate is too high causing gradient explosion"""
    print("\n=== Testing Learning Rate Effects ===")
    
    # Test different learning rates
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for lr in learning_rates:
        print(f"\nTesting LR: {lr}")
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Simulate training step
        x = torch.randn(32, 100)
        y = torch.randint(0, 5, (32,))
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient magnitude
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
        
        print(f"  Loss: {loss.item():.4f}, Grad norm: {total_grad_norm:.6f}")
        
        # Check if gradients are exploding
        if total_grad_norm > 100:
            print(f"  WARNING: Gradient explosion detected!")
        elif total_grad_norm < 1e-6:
            print(f"  WARNING: Vanishing gradients detected!")

def test_label_distribution():
    """Test if there's severe class imbalance in the dataset"""
    print("\n=== Testing Label Distribution ===")
    
    # Try to load a sample of the dataset
    try:
        from cbramod.datasets import idun_datasets
        
        # Create dummy params
        class DummyParams:
            def __init__(self):
                self.dataset_names = ['ORP']
                self.num_datasets = 1
                self.datasets_dir = 'Datasets/Final_dataset'
        
        params = DummyParams()
        
        # Try to load dataset
        if os.path.exists(params.datasets_dir):
            loader = idun_datasets.LoadDataset(params)
            pairs = loader.get_all_pairs()
            
            if pairs:
                dataset = idun_datasets.MemoryEfficientKFoldDataset(pairs[:5])  # Load first 5 files
                
                # Check label distribution
                labels = []
                for i in range(min(1000, len(dataset))):  # Sample first 1000 or all samples
                    _, label = dataset[i]
                    labels.append(label.item())
                
                label_counts = Counter(labels)
                print(f"Label distribution (first {len(labels)} samples): {dict(label_counts)}")
                
                # Check for severe imbalance
                total_samples = len(labels)
                for label, count in label_counts.items():
                    proportion = count / total_samples
                    print(f"  Class {label}: {count}/{total_samples} ({proportion:.2%})")
                    
                    if proportion < 0.01:  # Less than 1%
                        print(f"    WARNING: Class {label} severely underrepresented!")
                    elif proportion > 0.90:  # More than 90%
                        print(f"    WARNING: Class {label} dominates the dataset!")
                
                return label_counts
            else:
                print("No dataset files found!")
        else:
            print(f"Dataset directory not found: {params.datasets_dir}")
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def test_data_preprocessing():
    """Test if data preprocessing is causing issues"""
    print("\n=== Testing Data Preprocessing ===")
    
    # Test the remap_label function
    from cbramod.datasets.idun_datasets import MemoryEfficientKFoldDataset
    
    # Test label remapping
    test_labels = [0, 1, 2, 3, 4]
    
    # Create dummy dataset to test remap_label
    class TestDataset(MemoryEfficientKFoldDataset):
        def __init__(self):
            pass
    
    dataset = TestDataset()
    
    print("Testing label remapping:")
    for label in test_labels:
        try:
            remapped = dataset.remap_label(label)
            print(f"  {label} -> {remapped}")
        except Exception as e:
            print(f"  {label} -> ERROR: {e}")
    
    # Test if remap_label is reducing classes incorrectly
    original_labels = [0, 1, 2, 3, 4] * 100  # 500 samples
    remapped_labels = [dataset.remap_label(l) for l in original_labels]
    
    print(f"\nOriginal label distribution: {Counter(original_labels)}")
    print(f"Remapped label distribution: {Counter(remapped_labels)}")
    
    # Check if all labels are being mapped to the same class
    unique_remapped = set(remapped_labels)
    if len(unique_remapped) == 1:
        print("ERROR: All labels are being remapped to the same class!")
    else:
        print(f"Remapped labels span {len(unique_remapped)} classes: {sorted(unique_remapped)}")

def test_model_initialization():
    """Test if model weights are being initialized correctly"""
    print("\n=== Testing Model Initialization ===")
    
    class SimpleParams:
        def __init__(self):
            self.num_of_classes = 5
            self.use_pretrained_weights = False
            self.cuda = 0
    
    params = SimpleParams()
    
    # Test multiple model initializations
    for i in range(3):
        print(f"\nModel {i+1}:")
        from cbramod.models import model_for_idun
        model = model_for_idun.Model(params)
        
        # Check classifier weights
        classifier_weights = model.classifier.weight
        classifier_bias = model.classifier.bias
        
        print(f"  Classifier weight shape: {classifier_weights.shape}")
        print(f"  Classifier weight mean: {classifier_weights.mean().item():.6f}")
        print(f"  Classifier weight std: {classifier_weights.std().item():.6f}")
        print(f"  Classifier bias: {classifier_bias.tolist()}")
        
        # Test forward pass with random input
        x = torch.randn(1, 1, 30, 200)
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            print(f"  Output logits: {output[0].tolist()}")
            print(f"  Output probs: {probs[0].tolist()}")
            print(f"  Predicted class: {torch.argmax(output, dim=1).item()}")

def main():
    """Run all diagnostic tests"""
    print("🔍 CBraMod Kappa=0.0 Diagnostic Tool")
    print("=" * 60)
    
    # Test 1: Model architecture
    model, sample_input = test_model_architecture()
    
    # Test 2: Loss function
    criterion = test_loss_function()
    
    # Test 3: Gradient flow
    has_gradients = test_gradient_flow(model, criterion)
    
    # Test 4: Learning rate
    test_learning_rate()
    
    # Test 5: Label distribution
    label_counts = test_label_distribution()
    
    # Test 6: Data preprocessing
    test_data_preprocessing()
    
    # Test 7: Model initialization
    test_model_initialization()
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    # Summarize findings
    print("Key findings:")
    print(f"- Model has correct output size: {model.classifier.out_features == 5}")
    print(f"- Gradients are flowing: {has_gradients}")
    
    if label_counts:
        min_class_count = min(label_counts.values())
        max_class_count = max(label_counts.values())
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        print(f"- Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 100:
            print("  WARNING: Severe class imbalance detected!")
    
    print("\n🎯 LIKELY CAUSES OF KAPPA=0.0:")
    print("1. Model predicting only one class consistently")
    print("2. Learning rate too high (gradient explosion)")
    print("3. Learning rate too low (no learning)")
    print("4. Severe class imbalance in dataset")
    print("5. Label remapping error")
    print("6. Model architecture mismatch")

if __name__ == "__main__":
    main()