#!/usr/bin/env python3
"""
Enhanced finetuning pipeline for CBraMod with optimized pretrained weight loading
and comprehensive data quality integration.

Key improvements:
1. Integrated data pipeline for quality assurance
2. Optimized pretrained weight loading strategies
3. Enhanced training loop with better monitoring
4. Advanced learning rate scheduling
5. Improved early stopping and model selection
6. Comprehensive experiment tracking
7. Advanced hyperparameter optimization with Optuna

Usage:
  # Normal training (default: ORP only)
  python enhanced_finetune_main.py --config-env finetuning_optimized
  
  # Train with multiple datasets
  python enhanced_finetune_main.py --datasets ORP,2023_Open_N
  
  # Train with all available datasets
  python enhanced_finetune_main.py --datasets ORP,2017_Open_N,2019_Open_N,2023_Open_N
  
  # List available datasets
  python enhanced_finetune_main.py --list-datasets
  
  # List available class schemes
  python enhanced_finetune_main.py --list-class-schemes
  
  # Run diagnostic checks for 5-class system
  python enhanced_finetune_main.py --num-classes 5 --diagnose
  
  # Hyperparameter optimization with multiple datasets
  python enhanced_finetune_main.py --tune --datasets ORP,2023_Open_N --n-trials 20
  
  # Quick test
  python enhanced_finetune_main.py --quick-test
"""

import sys
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import Dict, Any, Optional, Tuple, List

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import configuration system
from config.utils import ConfigManager
from experiments.tracking import create_unified_tracker
from experiments.validation import EEGModelValidator
from experiments.reproducibility import ReproducibilityManager

# Import enhanced data pipeline
from cbramod.datasets.enhanced_dataset import EnhancedLoadDataset, EnhancedEEGDataset
from data_pipeline.validators import EEGDataValidator, DataQualityChecker

# Import advanced training strategies
from enhanced_training_strategies import create_enhanced_training_components

# Import class configuration system
from class_configurations import ClassConfigurationManager, get_num_classes_for_scheme

# Import diagnostic checks
from diagnostic_checks import check_class_configuration, print_diagnostic_report, quick_diagnostic_check

# Import existing modules
from cbramod.models import model_for_idun
from cbramod.Finetuning.finetune_trainer import Trainer
from torch.utils.data import DataLoader
import wandb

class EnhancedFinetunePipeline:
    """Enhanced finetuning pipeline with optimized pretrained weight loading"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize experiment components
        self.tracker = None
        self.validator = None
        self.repro_manager = None
        self.advanced_components = None
        
        # Setup experiment tracking and validation
        self._setup_experiment_infrastructure()
    
    def _setup_experiment_infrastructure(self):
        """Setup experiment tracking and validation infrastructure"""
        # Initialize unified tracker
        self.tracker = create_unified_tracker(
            experiment_name="enhanced_cbramod_finetuning",
            config=self.config_manager.config
        )
        
        # Initialize model validator
        validation_config = self.config_manager.get('validation', {})
        self.validator = EEGModelValidator(validation_config)
        
        # Initialize reproducibility manager
        self.repro_manager = ReproducibilityManager(self.config_manager.config)
    
    def setup_reproducibility(self, seed: int, experiment_id: str):
        """Setup reproducible environment"""
        experiment_seed = self.repro_manager.setup_reproducible_environment(
            master_seed=seed,
            experiment_id=experiment_id
        )
        
        # Capture environment state
        self.repro_manager.capture_environment_state(experiment_id)
        self.repro_manager.save_config_snapshot(experiment_id, self.config_manager.config)
        
        return experiment_seed
    
    def create_enhanced_model(self, params) -> nn.Module:
        """Create model with optimized pretrained weight loading"""
        self.logger.info("🏗️ Creating enhanced model with optimized pretrained loading...")
        
        # Create base model
        self.logger.info(f"DEBUG: Creating model with params.num_of_classes = {getattr(params, 'num_of_classes', 'NOT_SET')}")
        model = model_for_idun.Model(params)
        
        # Check actual model output size
        if hasattr(model, 'classifier'):
            actual_output_classes = model.classifier.out_features
            self.logger.info(f"DEBUG: Model classifier output features = {actual_output_classes}")
            expected_classes = getattr(params, 'num_of_classes', 5)
            if actual_output_classes != expected_classes:
                self.logger.error(f"❌ MODEL MISMATCH: Expected {expected_classes} classes, got {actual_output_classes}")
        else:
            self.logger.warning("DEBUG: Model has no classifier attribute")
        
        # Enhanced pretrained weight loading
        if params.use_pretrained_weights:
            self._load_pretrained_weights_optimized(model, params)
        
        return model
    
    def _load_pretrained_weights_optimized(self, model: nn.Module, params):
        """Optimized pretrained weight loading with better strategies"""
        self.logger.info(f"🔄 Loading pretrained weights from: {params.foundation_dir}")
        
        try:
            # Load pretrained weights
            map_location = torch.device(f'cuda:{params.cuda}')
            pretrained_dict = torch.load(params.foundation_dir, map_location=map_location)
            
            # Strategy 1: Progressive loading with layer-wise adaptation
            model_dict = model.backbone.state_dict()
            
            # Filter out unnecessary keys and handle size mismatches
            filtered_dict = {}
            mismatched_layers = []
            
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        mismatched_layers.append(k)
                        self.logger.warning(f"Shape mismatch for {k}: model {model_dict[k].shape} vs pretrained {v.shape}")
                        
                        # Handle specific mismatches intelligently
                        if 'proj_out' in k and len(v.shape) == 2:
                            # Skip or adapt output projection layers
                            continue
                        elif 'embedding' in k and len(v.shape) >= 2:
                            # Adapt embedding layers if needed
                            if model_dict[k].shape[0] <= v.shape[0]:
                                filtered_dict[k] = v[:model_dict[k].shape[0]]
                            else:
                                # Expand with random initialization
                                expanded_weight = torch.zeros_like(model_dict[k])
                                expanded_weight[:v.shape[0]] = v
                                # Initialize additional weights
                                nn.init.xavier_uniform_(expanded_weight[v.shape[0]:])
                                filtered_dict[k] = expanded_weight
            
            # Load filtered weights
            missing_keys, unexpected_keys = model.backbone.load_state_dict(filtered_dict, strict=False)
            
            # Report loading statistics
            self.logger.info(f"✅ Loaded {len(filtered_dict)}/{len(pretrained_dict)} pretrained weights")
            if missing_keys:
                self.logger.info(f"Missing keys: {len(missing_keys)} (will use random initialization)")
            if unexpected_keys:
                self.logger.info(f"Unexpected keys: {len(unexpected_keys)} (ignored)")
            if mismatched_layers:
                self.logger.info(f"Mismatched layers: {len(mismatched_layers)} (handled intelligently)")
            
            # Replace output projection with identity
            model.backbone.proj_out = nn.Identity()
            
            # Strategy 2: Adaptive learning rates for pretrained vs new layers
            self._setup_adaptive_learning_rates(model, params, filtered_dict.keys())
            
        except Exception as e:
            self.logger.error(f"Failed to load pretrained weights: {e}")
            self.logger.warning("Proceeding with random initialization")
    
    def _setup_adaptive_learning_rates(self, model: nn.Module, params, pretrained_layer_names: set):
        """Setup adaptive learning rates for pretrained vs new parameters"""
        if not hasattr(params, 'adaptive_lr') or not params.adaptive_lr:
            return
        
        # Classify parameters
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if any(pretrained_name in name for pretrained_name in pretrained_layer_names):
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        # Set different learning rates
        pretrained_lr = params.lr * 0.1  # Lower LR for pretrained layers
        new_lr = params.lr  # Full LR for new layers
        
        self.logger.info(f"🎯 Adaptive LR: pretrained layers={pretrained_lr:.2e}, new layers={new_lr:.2e}")
        
        # Store for optimizer creation
        params.param_groups = [
            {'params': pretrained_params, 'lr': pretrained_lr, 'name': 'pretrained'},
            {'params': new_params, 'lr': new_lr, 'name': 'new'}
        ]
    
    def create_enhanced_trainer(self, params, data_loader, model) -> Trainer:
        """Create trainer with enhanced capabilities"""
        # Create base trainer
        trainer = Trainer(params, data_loader, model)
        
        # Enhance with better learning rate scheduling
        self._setup_enhanced_scheduler(trainer, params)
        
        # Add early stopping
        self._setup_early_stopping(trainer, params)
        
        return trainer
    
    def _setup_enhanced_scheduler(self, trainer: Trainer, params):
        """Setup enhanced learning rate scheduler compatible with existing trainer"""
        enhanced_scheduler_type = getattr(params, 'enhanced_scheduler', 'cosine')
        
        # The trainer already sets up optimizer_scheduler, so we need to override it
        # for enhanced types only
        
        if enhanced_scheduler_type == 'cosine_warmup':
            # Use CosineAnnealingWarmRestarts as an enhanced version
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            trainer.optimizer_scheduler = CosineAnnealingWarmRestarts(
                trainer.optimizer, 
                T_0=max(1, params.epochs // 4),  # Restart every 1/4 of total epochs
                T_mult=1,
                eta_min=params.lr * 0.01
            )
            self.logger.info(f"📈 Setup enhanced cosine scheduler with warm restarts")
        elif enhanced_scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            trainer.optimizer_scheduler = ReduceLROnPlateau(
                trainer.optimizer,
                mode='max',  # Maximize validation kappa
                factor=0.5,
                patience=10,
                verbose=True
            )
            self.logger.info(f"📈 Setup plateau scheduler")
        elif enhanced_scheduler_type == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            trainer.optimizer_scheduler = ExponentialLR(
                trainer.optimizer,
                gamma=0.95
            )
            self.logger.info(f"📈 Setup exponential decay scheduler")
        else:
            # For 'cosine', 'step', 'none' - let the trainer handle it
            self.logger.info(f"📈 Using trainer's default {enhanced_scheduler_type} scheduler")
    
    def _setup_early_stopping(self, trainer: Trainer, params):
        """Setup early stopping mechanism"""
        trainer.early_stopping_patience = getattr(params, 'early_stopping_patience', 20)
        trainer.early_stopping_delta = getattr(params, 'early_stopping_delta', 0.001)
        trainer.best_val_score = float('-inf')
        trainer.early_stopping_counter = 0
        trainer.should_stop_early = False
        
        self.logger.info(f"⏰ Setup early stopping: patience={trainer.early_stopping_patience}, delta={trainer.early_stopping_delta}")
    
    def load_enhanced_dataset(self, params) -> Tuple[EnhancedEEGDataset, Dict[str, DataLoader]]:
        """Load dataset with enhanced data pipeline"""
        self.logger.info("📊 Loading dataset with enhanced data pipeline...")
        
        # For 5-class training, use original dataset loading to maintain performance
        if hasattr(params, 'classification_scheme') and params.classification_scheme == "5_class":
            self.logger.info("🎯 Using original dataset loading for 5-class to maintain performance")
            
            # Use original dataset loading
            from cbramod.datasets import idun_datasets
            load_dataset = idun_datasets.LoadDataset(params)
            seqs_labels_path_pair = load_dataset.get_all_pairs()
            
            # Create original dataset but with 5-class identity mapping
            dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair)
            
            # Override the remap_label method to use identity mapping for 5-class
            def identity_remap_label(self, l):
                return l  # Identity mapping for 5-class
            
            # Monkey patch the method
            import types
            dataset.remap_label = types.MethodType(identity_remap_label, dataset)
            
            # Validate the remapping works correctly
            self.logger.debug("Testing identity remapping for 5-class labels")
            for test_label in [0, 1, 2, 3, 4]:
                try:
                    result = dataset.remap_label(test_label)
                    self.logger.debug(f"Label mapping: {test_label} -> {result}")
                except Exception as e:
                    self.logger.error(f"Error mapping label {test_label}: {e}")
            
            # Validate dataset labels after loading
            self.logger.info("Validating dataset labels after loading...")
            sample_labels = []
            for i in range(min(100, len(dataset))):
                _, label = dataset[i]
                sample_labels.append(label.item() if hasattr(label, 'item') else label)
            
            import numpy as np
            unique_labels = sorted(set(sample_labels))
            self.logger.info(f"Found unique labels: {unique_labels}")
            self.logger.info(f"Label distribution: {dict(zip(*np.unique(sample_labels, return_counts=True)))}")
            
            # For 5-class original loading, we need to handle splitting and training here
            # since we're bypassing the enhanced system
            self.logger.info("🔄 Using original training flow for 5-class")
            
            # Use original data splitting
            from cbramod.datasets.idun_datasets import get_custom_split
            
            # Analyze dataset composition
            self.logger.info(f"Dataset size: {len(dataset)} samples")
            metadata = dataset.get_metadata()
            subjects = set(meta['subject'] for meta in metadata)
            orp_subjects = [s for s in subjects if s.startswith('S')]
            openneuro_subjects = [s for s in subjects if not s.startswith('S')]
            self.logger.info(f"ORP subjects: {len(orp_subjects)}")
            self.logger.info(f"OpenNeuro subjects: {len(openneuro_subjects)}")
            
            if len(orp_subjects) < 3:
                self.logger.warning(f"Only {len(orp_subjects)} ORP subjects found, need at least 3 for train/val/test split")
                self.logger.info("Falling back to simple random split")
                
                # Fallback to simple random split
                import numpy as np
                n_total = len(dataset)
                indices = list(range(n_total))
                np.random.shuffle(indices)
                
                n_train = int(0.7 * n_total)
                n_val = int(0.15 * n_total)
                
                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train+n_val]
                test_idx = indices[n_train+n_val:]
                
            else:
                fold, train_idx, val_idx, test_idx = next(get_custom_split(dataset, seed=42, orp_train_frac=getattr(params, 'data_ORP', 0.5)))
            
            print(f"\n▶️ Using split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
            
            # Create data subsets and loaders
            import torch
            from torch.utils.data import DataLoader, Subset
            
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            test_set = Subset(dataset, test_idx)
            
            train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=getattr(params, 'num_workers', 4))
            val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False, num_workers=getattr(params, 'num_workers', 4))
            test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=getattr(params, 'num_workers', 4))
            
            data_loader = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
            }
            
            # Create model
            self.logger.info(f"Creating model with {getattr(params, 'num_of_classes', 'undefined')} classes")
            from cbramod.models import model_for_idun
            model = model_for_idun.Model(params)
            
            # Validate model architecture
            if hasattr(model, 'classifier'):
                actual_output_classes = model.classifier.out_features
                expected_classes = getattr(params, 'num_of_classes', 5)
                self.logger.info(f"Model classifier output: {actual_output_classes} classes")
                if actual_output_classes != expected_classes:
                    self.logger.error(f"Model architecture mismatch: expected {expected_classes} classes, got {actual_output_classes}")
                    raise ValueError(f"Model output size ({actual_output_classes}) doesn't match expected classes ({expected_classes})")
            
            # Create trainer and start training
            from cbramod.Finetuning.finetune_trainer import Trainer
            trainer = Trainer(params, data_loader, model)
            
            self.logger.info("🚀 Starting original training flow...")
            kappa = trainer.train_for_multiclass()
            
            # Get test results for consistency with enhanced path
            acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)
            
            # Return results in the same format as enhanced path
            results = {
                'kappa': kappa,
                'accuracy': acc,
                'f1_score': f1,
                'trainer': trainer,
                'model': model,
                'dataset': dataset,
                'data_loader': data_loader
            }
            
            return results
            
        else:
            # Use enhanced dataset for 4-class or other schemes
            enhanced_loader = EnhancedLoadDataset(params, enable_data_pipeline=False)

            # Create enhanced dataset with quality checks and class configuration
            if hasattr(params, 'classification_scheme') and params.classification_scheme != "4_class":
                dataset = enhanced_loader.create_enhanced_dataset(
                    dataset_name="ORP",
                    dataset_version="2.0",
                    classification_scheme=params.classification_scheme
                )
            else:
                # Use default dataset without custom classification scheme
                dataset = enhanced_loader.create_enhanced_dataset(
                    dataset_name="ORP",
                    dataset_version="2.0"
                )
        
        # Generate and log quality report (only for enhanced dataset)
        if hasattr(dataset, 'get_data_quality_report'):
            quality_report = dataset.get_data_quality_report()
            
            if 'error' in quality_report:
                self.logger.warning(f"Quality report error: {quality_report['error']}")
            else:
                self.logger.info("📊 Data Quality Summary:")
                if 'total_files' in quality_report:
                    self.logger.info(f"  Files processed: {quality_report['total_files']}")
                    self.logger.info(f"  Files passed: {quality_report['passed_files']}")
                    self.logger.info(f"  Files failed: {quality_report['failed_files']}")
                    self.logger.info(f"  Warnings: {quality_report['warnings']}")
                    self.logger.info(f"  Errors: {quality_report['errors']}")
            
            # Track quality metrics
            if self.tracker:
                self.tracker.log_params({
                    'data_quality/total_files': quality_report.get('total_files', 0),
                    'data_quality/passed_files': quality_report.get('passed_files', 0),
                    'data_quality/failed_files': quality_report.get('failed_files', 0),
                    'data_quality/warnings': quality_report.get('warnings', 0),
                    'data_quality/errors': quality_report.get('errors', 0)
                })
        
        # Create data splits with improved strategy
        fold, train_idx, val_idx, test_idx = next(
            self._create_enhanced_data_split(dataset, params)
        )
        
        self.logger.info(f"📊 Enhanced data split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        # Create data subsets
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)
        
        # Create enhanced data loaders
        data_loader = self._create_enhanced_data_loaders(
            train_set, val_set, test_set, params
        )
        
        return dataset, data_loader
    
    def _create_enhanced_data_split(self, dataset, params):
        """Create enhanced data split with better stratification"""
        # Import the existing split function
        from cbramod.datasets.idun_datasets import get_custom_split
        
        # Use the existing split but with enhanced parameters
        seed = getattr(params, 'seed', 42)
        orp_train_frac = getattr(params, 'data_ORP', 0.6)
        
        return get_custom_split(dataset, seed=seed, orp_train_frac=orp_train_frac)
    
    def _create_enhanced_data_loaders(self, train_set, val_set, test_set, params) -> Dict[str, DataLoader]:
        """Create enhanced data loaders with better sampling strategies"""
        
        # Enhanced training loader with weighted sampling if enabled
        if getattr(params, 'use_weighted_sampler', False):
            # Calculate class weights for weighted sampling
            labels = [train_set[i][1].item() for i in range(len(train_set))]
            class_counts = torch.bincount(torch.tensor(labels))
            class_weights = 1.0 / class_counts.float()
            sample_weights = [class_weights[label] for label in labels]
            
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            train_loader = DataLoader(
                train_set, 
                batch_size=params.batch_size, 
                sampler=sampler,
                num_workers=params.num_workers,
                pin_memory=True,
                persistent_workers=True if params.num_workers > 0 else False
            )
            
            self.logger.info("🎯 Using weighted random sampler for balanced training")
        else:
            train_loader = DataLoader(
                train_set, 
                batch_size=params.batch_size, 
                shuffle=True,
                num_workers=params.num_workers,
                pin_memory=True,
                persistent_workers=True if params.num_workers > 0 else False
            )
        
        # Validation and test loaders
        val_loader = DataLoader(
            val_set, 
            batch_size=params.batch_size, 
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=params.batch_size, 
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def run_enhanced_training(self, params) -> Dict[str, float]:
        """Run enhanced training pipeline"""
        experiment_id = f"enhanced_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Setup reproducibility
            experiment_seed = self.setup_reproducibility(params.seed, experiment_id)
            
            with self.tracker:
                # Log experiment parameters
                self.tracker.log_params(vars(params))
                self.tracker.log_params({
                    'experiment_id': experiment_id,
                    'experiment_seed': experiment_seed,
                    'enhanced_pipeline': True
                })
                
                # Load enhanced dataset
                dataset_results = self.load_enhanced_dataset(params)
                
                # Handle different return formats
                if isinstance(dataset_results, dict):
                    # If results dict is returned (5-class path), extract components
                    dataset = dataset_results.get('dataset')
                    data_loader = dataset_results.get('data_loader')
                    
                    # If training was already done in 5-class path, return results immediately
                    if 'kappa' in dataset_results:
                        return {
                            'kappa': dataset_results['kappa'],
                            'accuracy': dataset_results['accuracy'],
                            'f1_score': dataset_results['f1_score']
                        }
                else:
                    # Legacy tuple format
                    dataset, data_loader = dataset_results
                
                # Print dataset information
                dataset_names = getattr(params, 'dataset_names', ['ORP'])
                print(f"📁 Using datasets: {', '.join(dataset_names)}")
                if hasattr(params, 'data_ORP'):
                    print(f"📊 ORP train fraction: {params.data_ORP}")
                
                # Create enhanced model
                model = self.create_enhanced_model(params)
                
                # Setup advanced training components
                advanced_config = self.config_manager.get('advanced', {})
                self.advanced_components = create_enhanced_training_components(advanced_config, model)
                
                if self.advanced_components:
                    self.logger.info(f"🚀 Enabled advanced components: {list(self.advanced_components.keys())}")
                
                # Run diagnostic checks if custom classification is used
                if hasattr(params, 'classification_scheme') and params.classification_scheme != "4_class":
                    self.logger.info("🔍 Running diagnostic checks for custom classification...")
                    
                    # Quick check first
                    is_config_correct = quick_diagnostic_check(params, dataset, model)
                    
                    if not is_config_correct:
                        self.logger.error("❌ Configuration issues detected! See diagnostic output above.")
                        # Still continue but with warning
                        print("\n⚠️ WARNING: Configuration issues detected. Training may fail or perform poorly!")
                        print("Fix the issues above before proceeding for best results.\n")
                
                # Log model information
                self.tracker.log_model_info(model)
                
                # Create enhanced trainer
                trainer = self.create_enhanced_trainer(params, data_loader, model)
                
                # Start training
                self.logger.info("🚀 Starting enhanced training...")
                
                # Training loop with enhanced monitoring
                kappa = self._run_training_loop(trainer, params)
                
                # Final evaluation
                acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)
                
                # Results
                results = {
                    'kappa': kappa,
                    'accuracy': acc,
                    'f1_score': f1
                }
                
                # Log final results
                self.tracker.log_metrics({
                    'final/kappa': kappa,
                    'final/accuracy': acc,
                    'final/f1_score': f1
                })
                
                # Validate model
                if self.validator:
                    validation_results = self.validator.validate_model(model, data_loader['test'])
                    validation_summary = {f"validation_{r.test_name}": r.status for r in validation_results}
                    self.tracker.log_params(validation_summary)
                
                self.logger.info("✅ Enhanced training completed successfully!")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _run_training_loop(self, trainer: Trainer, params) -> float:
        """Run enhanced training loop with monitoring"""
        # Use the existing trainer's train_for_multiclass method
        # but with enhanced monitoring
        
        self.logger.info(f"Training for {params.epochs} epochs...")
        kappa = trainer.train_for_multiclass()
        
        return kappa

def create_enhanced_params(config_manager: ConfigManager, args: argparse.Namespace) -> argparse.Namespace:
    """Create enhanced parameters with optimized defaults"""
    from finetune_main_with_config import create_params_from_config
    
    # Start with base params
    params = create_params_from_config(config_manager, args)
    
    # Add enhanced parameters
    params.adaptive_lr = getattr(args, 'adaptive_lr', config_manager.get('training.adaptive_lr', True))
    params.use_weighted_sampler = getattr(args, 'weighted_sampler', config_manager.get('data.weighted_sampler.enabled', True))
    params.early_stopping_patience = getattr(args, 'early_stopping_patience', config_manager.get('training.early_stopping.patience', 20))
    params.early_stopping_delta = getattr(args, 'early_stopping_delta', config_manager.get('training.early_stopping.delta', 0.001))
    
    # Enhanced scheduler - ensure it's compatible with trainer
    scheduler_type = getattr(args, 'scheduler', config_manager.get('training.scheduler.type', 'cosine'))
    # Store the original scheduler type for enhanced setup
    params.enhanced_scheduler = scheduler_type
    # Map enhanced scheduler types to compatible ones for trainer
    if scheduler_type in ['cosine_warmup', 'exponential', 'plateau']:
        params.scheduler = 'cosine'  # Use cosine as base, we'll override later
    else:
        params.scheduler = scheduler_type
    
    # CRITICAL FIX: Ensure model architecture matches classification scheme
    # This MUST be done before model creation to ensure correct output layer size
    if hasattr(args, 'num_classes') and args.num_classes is not None:
        # User explicitly specified number of classes  
        num_of_classes = int(args.num_classes)
        params.num_of_classes = num_of_classes
        params.classification_scheme = f"{num_of_classes}_class"
        print(f"🏗️ Model architecture: {num_of_classes} output classes (user override)")
    else:
        # Use config default
        classification_config = config_manager.get('model.classification', {})
        scheme = classification_config.get('scheme', '4_class')
        
        if scheme == '5_class':
            params.num_of_classes = 5
        else:
            params.num_of_classes = 4
        
        params.classification_scheme = scheme
        print(f"🏗️ Model architecture: {params.num_of_classes} output classes (from config: {scheme})")
    
    # Add missing attributes
    params.run_name = getattr(args, 'run_name', None)
    
    return params

def list_available_datasets(datasets_dir: str = "Datasets/Final_dataset") -> List[str]:
    """List all available datasets"""
    from pathlib import Path
    
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        print(f"❌ Datasets directory not found: {datasets_dir}")
        return []
    
    available_datasets = []
    for item in datasets_path.iterdir():
        if item.is_dir() and (item / "eeg_data_npy").exists():
            available_datasets.append(item.name)
    
    return sorted(available_datasets)

def validate_datasets(dataset_names: List[str], datasets_dir: str = "Datasets/Final_dataset") -> Tuple[List[str], List[str]]:
    """Validate that requested datasets exist and return valid/invalid lists"""
    from pathlib import Path
    
    valid_datasets = []
    invalid_datasets = []
    
    for dataset_name in dataset_names:
        dataset_path = Path(datasets_dir) / dataset_name / "eeg_data_npy"
        if dataset_path.exists():
            valid_datasets.append(dataset_name)
        else:
            invalid_datasets.append(dataset_name)
    
    return valid_datasets, invalid_datasets

def print_dataset_info(dataset_names: List[str], datasets_dir: str = "Datasets/Final_dataset"):
    """Print information about selected datasets"""
    from pathlib import Path
    import os
    
    print(f"\n📊 Dataset Information:")
    print("=" * 50)
    
    total_files = 0
    total_subjects = set()
    
    for dataset_name in dataset_names:
        dataset_path = Path(datasets_dir) / dataset_name / "eeg_data_npy"
        
        if dataset_path.exists():
            files = list(dataset_path.glob("*.npy"))
            subjects = set(f.stem.split("_")[0] for f in files)
            
            print(f"📁 {dataset_name}:")
            print(f"   Files: {len(files)}")
            print(f"   Subjects: {len(subjects)}")
            print(f"   Example subjects: {list(subjects)[:3]}...")
            
            total_files += len(files)
            total_subjects.update(subjects)
        else:
            print(f"❌ {dataset_name}: Not found")
    
    print("-" * 50)
    print(f"📈 Total: {total_files} files, {len(total_subjects)} unique subjects")
    print("=" * 50)

def main():
    """Main function for enhanced finetuning"""
    parser = argparse.ArgumentParser(description='Enhanced CBraMod finetuning pipeline')
    
    # Configuration arguments
    parser.add_argument('--config-env', type=str, default='development',
                        help='Configuration environment')
    
    # Enhanced arguments
    parser.add_argument('--adaptive-lr', action='store_true',
                        help='Use adaptive learning rates for pretrained vs new layers')
    parser.add_argument('--weighted-sampler', action='store_true',
                        help='Use weighted random sampler for balanced training')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--early-stopping-delta', type=float, default=0.001,
                        help='Early stopping minimum improvement')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine', 'cosine_warmup', 'plateau', 'exponential'],
                        help='Learning rate scheduler type')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with 2 epochs')
    
    # Standard arguments
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning with Optuna')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    
    # Dataset selection arguments
    parser.add_argument('--datasets', type=str, default='ORP', 
                        help='Comma-separated dataset names (default: ORP). Available: ORP,2017_Open_N,2019_Open_N,2023_Open_N')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets and exit')
    
    # Classification scheme arguments
    parser.add_argument('--num-classes', type=str, default=None, choices=['4', '5'],
                        help='Number of classes: 4 (current) or 5 (with separate movement class)')
    parser.add_argument('--list-class-schemes', action='store_true',
                        help='List available classification schemes and exit')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run detailed diagnostic checks and exit')
    
    args = parser.parse_args()
    
    # Handle dataset listing
    if args.list_datasets:
        available_datasets = list_available_datasets()
        print("📊 Available Datasets:")
        print("=" * 40)
        for dataset in available_datasets:
            print(f"  📁 {dataset}")
        print("=" * 40)
        print(f"\nUsage examples:")
        print(f"  --datasets ORP                    # Use only ORP")
        print(f"  --datasets ORP,2023_Open_N        # Use ORP and 2023_Open_N")
        print(f"  --datasets 2017_Open_N,2019_Open_N,2023_Open_N  # Use all Open_N datasets")
        sys.exit(0)
    
    # Handle class scheme listing
    if args.list_class_schemes:
        class_manager = ClassConfigurationManager()
        print("🏷️ Available Classification Schemes:")
        print("=" * 60)
        
        for scheme in ["4_class", "5_class"]:
            class_manager.print_scheme_info(scheme)
        
        print("\nUsage examples:")
        print("  --num-classes 4          # Use 4-class system (default)")
        print("  --num-classes 5          # Use 5-class system (separate movement)")
        sys.exit(0)
    
    # Parse and validate datasets
    datasets_dir = "Datasets/Final_dataset"
    requested_datasets = [d.strip() for d in args.datasets.split(',')]
    valid_datasets, invalid_datasets = validate_datasets(requested_datasets, datasets_dir)
    
    if invalid_datasets:
        print(f"❌ Invalid datasets: {invalid_datasets}")
        available = list_available_datasets(datasets_dir)
        print(f"💡 Available datasets: {available}")
        sys.exit(1)
    
    if not valid_datasets:
        print("❌ No valid datasets specified!")
        sys.exit(1)
    
    # Print dataset information
    print_dataset_info(valid_datasets, datasets_dir)
    
    # Configure class scheme only if explicitly requested
    class_scheme = None
    class_config = None
    class_manager = None
    
    if args.num_classes is not None:
        # User explicitly requested a specific number of classes
        class_scheme = f"{args.num_classes}_class"
        class_manager = ClassConfigurationManager()
        class_config = class_manager.get_configuration(class_scheme)
        
        print(f"\n🏷️ Classification Configuration (Override):")
        print(f"  Scheme: {class_config.name}")
        print(f"  Classes: {class_config.num_of_classes}")
        print(f"  Class names: {', '.join(class_config.class_names)}")
        print(f"  Label mapping: {class_config.label_mapping}")
    else:
        # Use default from configuration - no override
        print(f"\n🏷️ Using default classification from config file")
    
    # Handle quick test
    if args.quick_test:
        args.epochs = 2
        print("🧪 Quick test mode: Running with 2 epochs")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize configuration
        config_manager = ConfigManager(environment=args.config_env)
        
        # Create enhanced parameters
        params = create_enhanced_params(config_manager, args)
        
        # Override datasets with command line selection
        params.datasets = ','.join(valid_datasets)
        params.dataset_names = valid_datasets
        params.num_datasets = len(valid_datasets)
        
        # Override class configuration only if user specified --num-classes
        if class_config is not None:
            params.classification_scheme = class_scheme
            params.class_names = class_config.class_names
            params.num_of_classes = class_config.num_of_classes
            print(f"🔄 Overriding num_of_classes: {class_config.num_of_classes}")
        else:
            # Keep defaults from config file
            params.classification_scheme = "4_class"  # Default fallback
            if not hasattr(params, 'num_of_classes'):
                params.num_of_classes = 4
            print(f"📋 Using default num_of_classes: {params.num_of_classes}")
        
        print("🚀 Enhanced CBraMod Finetuning Pipeline")
        print("=" * 60)
        print("📋 Pipeline Features:")
        print("  🔍 Data Quality Pipeline - EEG validation & monitoring")
        print("  📊 Data Versioning - DVC-based version management")
        print("  🏗️ Optimized Finetuning - Smart pretrained weight loading")
        print("  📈 Enhanced Training - Advanced schedulers & early stopping")
        print("  🎯 Experiment Tracking - Unified MLflow + WandB + local")
        print("")
        print("⚙️ Configuration:")
        print(f"  Environment: {config_manager.config.get('_meta', {}).get('environment', 'unknown')}")
        print(f"  📁 Datasets: {', '.join(params.dataset_names)} ({params.num_datasets} total)")
        if hasattr(params, 'class_names'):
            print(f"  🏷️ Classes: {params.num_of_classes} ({', '.join(params.class_names)})")
        else:
            print(f"  🏷️ Classes: {params.num_of_classes} (using default mapping)")
        print(f"  Epochs: {params.epochs}")
        print(f"  Batch size: {params.batch_size}")
        print(f"  Learning rate: {params.lr}")
        print(f"  Adaptive LR: {params.adaptive_lr}")
        print(f"  Weighted sampler: {params.use_weighted_sampler}")
        print(f"  Scheduler: {params.enhanced_scheduler} (mapped to: {params.scheduler})")
        print(f"  Early stopping patience: {params.early_stopping_patience}")
        print("=" * 60)
        
        # Run diagnostic checks if requested
        if args.diagnose:
            print("🔍 Running detailed diagnostic checks...")
            
            # Create pipeline and load dataset/model for diagnostics
            pipeline = EnhancedFinetunePipeline(config_manager)
            dataset, data_loader = pipeline.load_enhanced_dataset(params)
            model = pipeline.create_enhanced_model(params)
            
            # Run comprehensive diagnostic checks
            diagnostic_results = check_class_configuration(params, dataset, model)
            print_diagnostic_report(diagnostic_results)
            
            # Also run training setup verification
            from diagnostic_checks import verify_training_setup
            verify_training_setup(params, dataset, model)
            
            print("\n✅ Diagnostic checks completed!")
            return
        
        # Run hyperparameter tuning if requested
        if args.tune:
            from enhanced_hyperopt import run_enhanced_hyperopt
            print("🔍 Starting enhanced hyperparameter optimization...")
            results = run_enhanced_hyperopt(config_manager, args, n_trials=args.n_trials)
            
            if "best_trial" in results:
                best_params = results["best_trial"]["params"]
                print(f"\n🏆 Best hyperparameters found:")
                for key, value in best_params.items():
                    print(f"  {key}: {value}")
            
            return results
        
        # Create and run enhanced pipeline
        pipeline = EnhancedFinetunePipeline(config_manager)
        results = pipeline.run_enhanced_training(params)
        
        print("\n🎉 Training Results:")
        print(f"  Kappa: {results['kappa']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        raise

if __name__ == '__main__':
    main()