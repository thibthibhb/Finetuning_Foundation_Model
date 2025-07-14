"""
MLOps-Ready Class Configuration System for CBraMod

This module provides flexible class mapping configurations for different
sleep staging scenarios. Easily switch between 4-class and 5-class systems
via configuration without code changes.

Supports:
- 4-class system (current): Wake, Light, Deep, REM
- 5-class system: Wake, Movement, Light, Deep, REM  
- Custom class mappings
- Class weight calculation
- Label validation
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

class ClassificationScheme(Enum):
    """Available classification schemes"""
    FOUR_CLASS = "4_class"
    FIVE_CLASS = "5_class"

@dataclass
class ClassConfig:
    """Configuration for a specific classification scheme"""
    name: str
    num_of_classes: int
    label_mapping: Dict[int, int]  # original_label -> new_label
    class_names: List[str]
    class_descriptions: List[str]
    class_weights: Optional[Dict[int, float]] = None

class ClassConfigurationManager:
    """Manages different class configuration schemes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.configurations = self._initialize_configurations()
    
    def _initialize_configurations(self) -> Dict[str, ClassConfig]:
        """Initialize all available class configurations"""
        
        configs = {}
        
        # 4-Class Configuration (Current)
        configs["4_class"] = ClassConfig(
            name="4_class",
            num_of_classes=4,
            label_mapping={
                0: 0,  # Wake (W) → Wake
                1: 0,  # Movement (M) → Wake (merged)
                2: 1,  # Light sleep (N1) → Light
                3: 2,  # Deep sleep (N2) → Deep
                4: 3,  # REM sleep → REM
            },
            class_names=["Wake", "Light", "Deep", "REM"],
            class_descriptions=[
                "Wake state (includes movement)",
                "Light sleep (N1)",
                "Deep sleep (N2)",
                "REM sleep"
            ],
            class_weights={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
        )
        
        # 5-Class Configuration (New)
        configs["5_class"] = ClassConfig(
            name="5_class",
            num_of_classes=5,
            label_mapping={
                0: 0,  # Wake (W) → Wake
                1: 1,  # Movement (M) → Movement (separate)
                2: 2,  # Light sleep (N1) → Light
                3: 3,  # Deep sleep (N2) → Deep
                4: 4,  # REM sleep → REM
            },
            class_names=["Wake", "Movement", "Light", "Deep", "REM"],
            class_descriptions=[
                "Wake state (alert)",
                "Movement/Transition state",
                "Light sleep (N1)",
                "Deep sleep (N2)",
                "REM sleep"
            ],
            class_weights={0: 1.0, 1: 1.2, 2: 1.0, 3: 1.0, 4: 1.0}  # Movement gets higher weight
        )
        
        
        return configs
    
    def get_configuration(self, scheme: str) -> ClassConfig:
        """Get class configuration by name"""
        if scheme not in self.configurations:
            available = list(self.configurations.keys())
            raise ValueError(f"Unknown classification scheme: {scheme}. Available: {available}")
        
        return self.configurations[scheme]
    
    def list_available_schemes(self) -> List[str]:
        """List all available classification schemes"""
        return list(self.configurations.keys())
    
    def validate_labels(self, labels: np.ndarray, scheme: str) -> Tuple[bool, List[str]]:
        """Validate that labels are compatible with the chosen scheme"""
        config = self.get_configuration(scheme)
        
        unique_labels = np.unique(labels)
        valid_original_labels = set(config.label_mapping.keys())
        
        issues = []
        
        for label in unique_labels:
            if label not in valid_original_labels:
                issues.append(f"Label {label} not supported in {scheme} scheme")
        
        return len(issues) == 0, issues
    
    def remap_labels(self, labels: np.ndarray, scheme: str) -> np.ndarray:
        """Remap labels according to the chosen scheme"""
        config = self.get_configuration(scheme)
        
        # Validate first
        is_valid, issues = self.validate_labels(labels, scheme)
        if not is_valid:
            raise ValueError(f"Label validation failed: {issues}")
        
        # Remap labels
        remapped = np.array([config.label_mapping[label] for label in labels])
        
        self.logger.info(f"Remapped {len(labels)} labels using {scheme} scheme")
        self.logger.info(f"Original range: {labels.min()}-{labels.max()}, New range: {remapped.min()}-{remapped.max()}")
        
        return remapped
    
    def get_class_weights_tensor(self, scheme: str, device: str = "cpu"):
        """Get class weights as PyTorch tensor"""
        import torch
        
        config = self.get_configuration(scheme)
        if config.class_weights is None:
            # Default to equal weights
            weights = torch.ones(config.num_of_classes)
        else:
            weights = torch.tensor([config.class_weights[i] for i in range(config.num_of_classes)])
        
        return weights.to(device)
    
    def calculate_dataset_class_distribution(self, labels: np.ndarray, scheme: str) -> Dict[str, Any]:
        """Calculate class distribution for a dataset"""
        config = self.get_configuration(scheme)
        remapped_labels = self.remap_labels(labels, scheme)
        
        unique, counts = np.unique(remapped_labels, return_counts=True)
        total = len(remapped_labels)
        
        distribution = {}
        for i in range(config.num_of_classes):
            class_name = config.class_names[i]
            count = counts[unique == i][0] if i in unique else 0
            percentage = (count / total) * 100
            
            distribution[class_name] = {
                'count': int(count),
                'percentage': float(percentage),
                'class_id': i
            }
        
        return {
            'scheme': scheme,
            'total_samples': total,
            'num_classes': config.num_of_classes,
            'distribution': distribution
        }
    
    def print_scheme_info(self, scheme: str):
        """Print detailed information about a classification scheme"""
        config = self.get_configuration(scheme)
        
        print(f"\n📊 Classification Scheme: {config.name}")
        print("=" * 50)
        print(f"Number of classes: {config.num_of_classes}")
        print("\nLabel Mapping:")
        for orig, new in config.label_mapping.items():
            class_name = config.class_names[new]
            description = config.class_descriptions[new]
            weight = config.class_weights[new] if config.class_weights else 1.0
            print(f"  {orig} → {new} ({class_name}) - {description} [weight: {weight}]")
        
        print("\nClass Names:")
        for i, (name, desc) in enumerate(zip(config.class_names, config.class_descriptions)):
            print(f"  {i}: {name} - {desc}")
        print("=" * 50)
    
    def export_configuration(self, scheme: str) -> Dict[str, Any]:
        """Export configuration for use in training scripts"""
        config = self.get_configuration(scheme)
        
        return {
            'classification_scheme': scheme,
            'num_classes': config.num_of_classes,
            'class_names': config.class_names,
            'class_descriptions': config.class_descriptions,
            'label_mapping': config.label_mapping,
            'class_weights': config.class_weights
        }

def create_enhanced_dataset_with_classes(original_dataset_class, classification_scheme: str = "4_class"):
    """Factory function to create dataset with configurable class mapping"""
    
    class ConfigurableClassDataset(original_dataset_class):
        """Enhanced dataset with configurable class mapping"""
        
        def __init__(self, *args, **kwargs):
            # Extract classification scheme from kwargs
            self.classification_scheme = kwargs.pop('classification_scheme', classification_scheme)
            self.class_manager = ClassConfigurationManager()
            
            # Initialize parent class
            super().__init__(*args, **kwargs)
            
            # Log class configuration
            config = self.class_manager.get_configuration(self.classification_scheme)
            logging.getLogger(__name__).info(f"🏷️ Using {config.name} classification with {config.num_of_classes} classes")
        
        def remap_label(self, l):
            """Override the remap_label method to use configurable mapping"""
            try:
                config = self.class_manager.get_configuration(self.classification_scheme)
                if l in config.label_mapping:
                    return config.label_mapping[l]
                else:
                    raise ValueError(f"Unknown label value: {l} for scheme {self.classification_scheme}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Error remapping label {l}: {e}")
                raise
        
        def get_class_info(self):
            """Get information about the current class configuration"""
            return self.class_manager.export_configuration(self.classification_scheme)
    
    return ConfigurableClassDataset

# Utility functions for easy integration
def get_num_classes_for_scheme(scheme: str) -> int:
    """Quick utility to get number of classes for a scheme"""
    manager = ClassConfigurationManager()
    config = manager.get_configuration(scheme)
    return config.num_of_classes

def validate_scheme_name(scheme: str) -> bool:
    """Validate that a scheme name exists"""
    manager = ClassConfigurationManager()
    return scheme in manager.list_available_schemes()

def print_all_schemes():
    """Print information about all available schemes"""
    manager = ClassConfigurationManager()
    
    print("🎯 Available Classification Schemes:")
    print("=" * 60)
    
    for scheme in manager.list_available_schemes():
        manager.print_scheme_info(scheme)
        print()

if __name__ == "__main__":
    # Demo the class configuration system
    print_all_schemes()