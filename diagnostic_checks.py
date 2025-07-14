"""
Diagnostic checks for class configuration system

This module provides comprehensive checks to verify that the class
configuration system is working correctly.
"""

import numpy as np
import torch
import logging
from collections import Counter
from typing import Dict, List, Tuple, Any

def check_class_configuration(params, dataset, model) -> Dict[str, Any]:
    """Comprehensive check of class configuration"""
    
    results = {
        'params_check': {},
        'dataset_check': {},
        'model_check': {},
        'label_distribution': {},
        'issues': []
    }
    
    # 1. Check parameters
    results['params_check'] = {
        'num_of_classes': getattr(params, 'num_of_classes', 'NOT_SET'),
        'classification_scheme': getattr(params, 'classification_scheme', 'NOT_SET'),
        'class_names': getattr(params, 'class_names', 'NOT_SET')
    }
    
    # 2. Check dataset configuration
    if hasattr(dataset, 'class_config'):
        results['dataset_check'] = {
            'has_class_config': True,
            'scheme': dataset.classification_scheme,
            'num_of_classes': dataset.class_config.num_of_classes,
            'class_names': dataset.class_config.class_names,
            'label_mapping': dataset.class_config.label_mapping
        }
    else:
        results['dataset_check'] = {
            'has_class_config': False,
            'using_original_remap': True
        }
    
    # 3. Check model configuration
    if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
        model_output_classes = model.fc.out_features
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        model_output_classes = model.classifier.out_features
    else:
        # Try to find the final layer
        model_output_classes = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                model_output_classes = module.out_features
    
    results['model_check'] = {
        'output_classes': model_output_classes,
        'matches_params': model_output_classes == getattr(params, 'num_classes', None)
    }
    
    # 4. Check actual label distribution in dataset
    all_labels = []
    for i in range(min(1000, len(dataset))):  # Sample first 1000
        try:
            _, label = dataset[i]
            all_labels.append(label.item() if torch.is_tensor(label) else label)
        except Exception as e:
            results['issues'].append(f"Error getting label at index {i}: {e}")
            break
    
    if all_labels:
        label_counts = Counter(all_labels)
        unique_labels = sorted(label_counts.keys())
        
        results['label_distribution'] = {
            'unique_labels': unique_labels,
            'counts': dict(label_counts),
            'min_label': min(unique_labels),
            'max_label': max(unique_labels),
            'expected_max': getattr(params, 'num_classes', 4) - 1
        }
        
        # Check if labels are in expected range
        expected_max = getattr(params, 'num_classes', 4) - 1
        if max(unique_labels) > expected_max:
            results['issues'].append(f"Found label {max(unique_labels)} but expected max is {expected_max}")
        
        if len(unique_labels) != getattr(params, 'num_classes', 4):
            results['issues'].append(f"Found {len(unique_labels)} unique labels but expected {getattr(params, 'num_classes', 4)}")
    
    # 5. Test label remapping function
    if hasattr(dataset, 'remap_label'):
        remap_test_results = {}
        test_labels = [0, 1, 2, 3, 4]
        for orig_label in test_labels:
            try:
                remapped = dataset.remap_label(orig_label)
                remap_test_results[orig_label] = remapped
            except Exception as e:
                remap_test_results[orig_label] = f"ERROR: {e}"
        
        results['remap_test'] = remap_test_results
    
    return results

def print_diagnostic_report(results: Dict[str, Any]):
    """Print a comprehensive diagnostic report"""
    
    print("\n" + "="*60)
    print("🔍 CLASS CONFIGURATION DIAGNOSTIC REPORT")
    print("="*60)
    
    # Parameters check
    print("\n📋 Parameter Configuration:")
    for key, value in results['params_check'].items():
        print(f"  {key}: {value}")
    
    # Dataset check
    print("\n📊 Dataset Configuration:")
    for key, value in results['dataset_check'].items():
        print(f"  {key}: {value}")
    
    # Model check
    print("\n🏗️ Model Configuration:")
    for key, value in results['model_check'].items():
        status = "✅" if value else "❌" if key == 'matches_params' else ""
        print(f"  {key}: {value} {status}")
    
    # Label distribution
    print("\n🏷️ Actual Label Distribution:")
    if 'label_distribution' in results:
        dist = results['label_distribution']
        print(f"  Unique labels found: {dist['unique_labels']}")
        print(f"  Label range: {dist['min_label']} - {dist['max_label']}")
        print(f"  Expected range: 0 - {dist['expected_max']}")
        print(f"  Label counts:")
        for label, count in dist['counts'].items():
            print(f"    Label {label}: {count} samples")
    
    # Remap test
    if 'remap_test' in results:
        print("\n🔄 Label Remapping Test:")
        for orig, remapped in results['remap_test'].items():
            print(f"  {orig} → {remapped}")
    
    # Issues
    if results['issues']:
        print("\n❌ Issues Found:")
        for issue in results['issues']:
            print(f"  • {issue}")
    else:
        print("\n✅ No issues found!")
    
    print("="*60)

def verify_training_setup(params, dataset, model, sample_batch_size=32):
    """Verify that training setup is correct"""
    
    print(f"\n🧪 TRAINING SETUP VERIFICATION")
    print("="*50)
    
    # 1. Check if we can create a data loader
    try:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=sample_batch_size, shuffle=False)
        
        # Get a batch
        batch = next(iter(loader))
        X_batch, y_batch = batch
        
        print(f"✅ DataLoader created successfully")
        print(f"  Batch shape: {X_batch.shape}")
        print(f"  Labels shape: {y_batch.shape}")
        print(f"  Label range in batch: {y_batch.min().item()} - {y_batch.max().item()}")
        print(f"  Unique labels in batch: {torch.unique(y_batch).tolist()}")
        
        # 2. Test forward pass
        try:
            model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                    model = model.cuda()
                
                output = model(X_batch)
                print(f"✅ Forward pass successful")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected output shape: ({sample_batch_size}, {getattr(params, 'num_classes', 4)})")
                
                if output.shape[1] != getattr(params, 'num_classes', 4):
                    print(f"❌ OUTPUT SHAPE MISMATCH!")
                    print(f"  Model outputs {output.shape[1]} classes")
                    print(f"  But params expects {getattr(params, 'num_classes', 4)} classes")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
    
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")

def quick_diagnostic_check(params, dataset, model):
    """Quick diagnostic check with summary"""
    
    print("\n🔍 QUICK DIAGNOSTIC CHECK")
    print("="*40)
    
    # Check the key indicators
    issues = []
    
    # 1. Check params - ensure it's an integer
    expected_classes = getattr(params, 'num_of_classes', 4)
    expected_classes = int(expected_classes)  # Ensure it's an integer
    print(f"Expected classes: {expected_classes}")
    
    # 2. Check dataset labels - sample more broadly to catch all classes
    sample_labels = []
    dataset_size = len(dataset)
    sample_size = min(1000, dataset_size)
    
    # Sample from different parts of the dataset to catch all classes
    import numpy as np
    if dataset_size > sample_size:
        # Sample evenly distributed indices across the dataset
        indices = np.linspace(0, dataset_size - 1, sample_size, dtype=int)
    else:
        indices = range(dataset_size)
    
    for i in indices:
        _, label = dataset[i]
        sample_labels.append(label.item() if torch.is_tensor(label) else label)
    
    unique_labels = sorted(set(sample_labels))
    print(f"Actual unique labels: {unique_labels}")
    print(f"Label range: {min(unique_labels)} - {max(unique_labels)}")
    
    if max(unique_labels) >= expected_classes:
        issues.append(f"Label {max(unique_labels)} found but only {expected_classes} classes expected!")
    
    if len(unique_labels) != expected_classes:
        issues.append(f"Found {len(unique_labels)} unique labels but expected {expected_classes}")
        # Additional diagnostic: check if this is a mapping issue
        if len(unique_labels) == 4 and expected_classes == 5:
            issues.append("This suggests 4-class mapping was applied when 5-class was expected")
    
    # 3. Check model output
    try:
        sample_input, _ = dataset[0]
        sample_input = sample_input.unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
            model = model.cuda()
        
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
            model_classes = output.shape[1]
            
        print(f"Model output classes: {model_classes}")
        
        if model_classes != expected_classes:
            issues.append(f"Model outputs {model_classes} classes but {expected_classes} expected!")
    
    except Exception as e:
        issues.append(f"Model test failed: {e}")
    
    # Summary
    if issues:
        print(f"\n❌ {len(issues)} ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        print("\n💡 This likely explains the poor performance!")
    else:
        print(f"\n✅ Configuration looks correct!")
    
    return len(issues) == 0