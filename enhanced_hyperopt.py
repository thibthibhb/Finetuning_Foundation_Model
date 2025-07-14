"""
Enhanced Hyperparameter Optimization for CBraMod

This module provides MLOps-compliant hyperparameter optimization using Optuna
with integration to the enhanced finetuning pipeline.

Features:
- Optuna study with advanced pruning
- Integration with enhanced pipeline
- Experiment tracking for all trials
- Smart parameter spaces for finetuning
- Study persistence and resumption
"""

import sys
import os
import optuna
import copy
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch

# Import comprehensive logging and error handling
from utils.comprehensive_logging import CBraModLogger, StandardizedErrorHandler, log_performance
from utils.comprehensive_logging import ConfigurationError, ValidationError, ResourceError, TrainingError

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config.utils import ConfigManager
from enhanced_finetune_main import EnhancedFinetunePipeline, create_enhanced_params

class EnhancedHyperparameterOptimizer:
    """Enhanced hyperparameter optimization with MLOps integration"""
    
    def __init__(self, config_manager: ConfigManager, base_args):
        self.config_manager = config_manager
        self.base_args = base_args
        
        # Setup comprehensive logging
        self.logger = CBraModLogger(config_manager, "enhanced_hyperopt")
        self.error_handler = StandardizedErrorHandler(self.logger)
        
        # Setup study configuration
        self.study_name = f"cbramod_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study_dir = Path("optuna_studies")
        self.study_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = EnhancedFinetunePipeline(config_manager)
    
    def create_study(self, n_trials: int = 20, resume: bool = True) -> optuna.Study:
        """Create or resume Optuna study"""
        
        # Study storage for persistence
        storage_path = self.study_dir / f"{self.study_name}.db"
        storage = f"sqlite:///{storage_path}"
        
        # Advanced pruning strategy
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,      # Wait 3 trials before pruning
            n_warmup_steps=5,        # Wait 5 epochs before pruning
            interval_steps=5         # Check every 5 epochs
        )
        
        # Create study with advanced sampler
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="maximize",
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=5,
                n_ei_candidates=24,
                multivariate=True,
                warn_independent_sampling=False
            ),
            load_if_exists=resume
        )
        
        self.logger.info(f"📊 Created study: {self.study_name}")
        self.logger.info(f"💾 Study storage: {storage_path}")
        
        return study
    
    def validate_search_space_config(self, search_space_config: Dict[str, Any], validation_config: Dict[str, Any]) -> None:
        """Validate search space configuration to prevent runtime errors"""
        
        # Validate learning rate range
        if 'learning_rate' in search_space_config:
            lr_range = search_space_config['learning_rate']
            if not isinstance(lr_range, list) or len(lr_range) != 2:
                raise ValueError(f"learning_rate must be a list of 2 values, got: {lr_range}")
            
            # Convert to float for comparison
            try:
                lr_min = float(lr_range[0])
                lr_max = float(lr_range[1])
                if lr_min >= lr_max:
                    raise ValueError(f"learning_rate min ({lr_min}) must be < max ({lr_max})")
                if lr_min <= 0 or lr_max <= 0:
                    raise ValueError(f"learning_rate values must be positive, got: [{lr_min}, {lr_max}]")
            except (ValueError, TypeError) as e:
                raise ValueError(f"learning_rate values must be numeric, got: {lr_range}") from e
        
        # Validate weight decay range
        if 'weight_decay' in search_space_config:
            wd_range = search_space_config['weight_decay']
            if not isinstance(wd_range, list) or len(wd_range) != 2:
                raise ValueError(f"weight_decay must be a list of 2 values, got: {wd_range}")
            
            try:
                wd_min = float(wd_range[0])
                wd_max = float(wd_range[1])
                if wd_min >= wd_max:
                    raise ValueError(f"weight_decay min ({wd_min}) must be < max ({wd_max})")
                if wd_min < 0 or wd_max < 0:
                    raise ValueError(f"weight_decay values must be non-negative, got: [{wd_min}, {wd_max}]")
            except (ValueError, TypeError) as e:
                raise ValueError(f"weight_decay values must be numeric, got: {wd_range}") from e
        
        # Validate batch sizes
        if 'batch_size' in search_space_config:
            batch_sizes = search_space_config['batch_size']
            if not isinstance(batch_sizes, list) or len(batch_sizes) == 0:
                raise ValueError(f"batch_size must be a non-empty list, got: {batch_sizes}")
            for bs in batch_sizes:
                try:
                    bs_int = int(bs)
                    if bs_int <= 0:
                        raise ValueError(f"batch_size values must be positive integers, got: {bs}")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"batch_size values must be integers, got: {bs}") from e
        
        # Validate optimizer options
        if 'optimizer' in search_space_config:
            optimizers = search_space_config['optimizer']
            if not isinstance(optimizers, list) or len(optimizers) == 0:
                raise ValueError(f"optimizer must be a non-empty list, got: {optimizers}")
            valid_optimizers = ["AdamW", "Lion", "SGD", "Adam"]
            for opt in optimizers:
                if opt not in valid_optimizers:
                    raise ValueError(f"optimizer '{opt}' not in valid options: {valid_optimizers}")
        
        # Validate scheduler options
        if 'scheduler' in search_space_config:
            schedulers = search_space_config['scheduler']
            if not isinstance(schedulers, list) or len(schedulers) == 0:
                raise ValueError(f"scheduler must be a non-empty list, got: {schedulers}")
            valid_schedulers = ["cosine", "cosine_warmup", "plateau", "exponential", "step", "none"]
            for sched in schedulers:
                if sched not in valid_schedulers:
                    raise ValueError(f"scheduler '{sched}' not in valid options: {valid_schedulers}")
        
        # Validate data_ORP range
        if 'data_ORP' in search_space_config:
            orp_range = search_space_config['data_ORP']
            if not isinstance(orp_range, list) or len(orp_range) != 2:
                raise ValueError(f"data_ORP must be a list of 2 values, got: {orp_range}")
            
            try:
                orp_min = float(orp_range[0])
                orp_max = float(orp_range[1])
                if orp_min >= orp_max:
                    raise ValueError(f"data_ORP min ({orp_min}) must be < max ({orp_max})")
                if orp_min < 0 or orp_max > 1:
                    raise ValueError(f"data_ORP values must be between 0 and 1, got: [{orp_min}, {orp_max}]")
            except (ValueError, TypeError) as e:
                raise ValueError(f"data_ORP values must be numeric, got: {orp_range}") from e
        
        # Validate label smoothing range
        if 'label_smoothing' in search_space_config:
            ls_range = search_space_config['label_smoothing']
            if not isinstance(ls_range, list) or len(ls_range) != 2:
                raise ValueError(f"label_smoothing must be a list of 2 values, got: {ls_range}")
            
            try:
                ls_min = float(ls_range[0])
                ls_max = float(ls_range[1])
                if ls_min >= ls_max:
                    raise ValueError(f"label_smoothing min ({ls_min}) must be < max ({ls_max})")
                if ls_min < 0 or ls_max > 1:
                    raise ValueError(f"label_smoothing values must be between 0 and 1, got: [{ls_min}, {ls_max}]")
            except (ValueError, TypeError) as e:
                raise ValueError(f"label_smoothing values must be numeric, got: {ls_range}") from e
        
        self.logger.info("✅ Search space configuration validation passed")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define enhanced search space from YAML configuration"""
        
        # Get hyperopt configuration from YAML
        hyperopt_config = self.config_manager.get('advanced.hyperopt', {})
        search_space_config = hyperopt_config.get('search_space', {})
        validation_config = hyperopt_config.get('validation', {})
        search_ranges = hyperopt_config.get('search_ranges', {})
        
        # Validate configuration before proceeding
        self.validate_search_space_config(search_space_config, validation_config)
        
        search_params = {}
        
        # Learning rate
        if 'learning_rate' in search_space_config:
            lr_range = search_space_config['learning_rate']
            min_lr = float(lr_range[0])
            max_lr = float(lr_range[1])
            search_params['lr'] = trial.suggest_float("lr", min_lr, max_lr, log=True)
        else:
            # Use configurable multipliers
            default_lr = float(self.config_manager.get('training.learning_rate', 0.0005))
            lr_multipliers = search_ranges.get('lr_multiplier_range', [0.1, 10.0])
            search_params['lr'] = trial.suggest_float("lr", default_lr * lr_multipliers[0], default_lr * lr_multipliers[1], log=True)
        
        # Weight decay
        if 'weight_decay' in search_space_config:
            wd_range = search_space_config['weight_decay']
            min_wd = float(wd_range[0])
            max_wd = float(wd_range[1])
            search_params['weight_decay'] = trial.suggest_float("weight_decay", min_wd, max_wd, log=True)
        else:
            default_wd = float(self.config_manager.get('training.weight_decay', 0.01))
            wd_multipliers = search_ranges.get('wd_multiplier_range', [0.01, 10.0])
            search_params['weight_decay'] = trial.suggest_float("weight_decay", default_wd * wd_multipliers[0], default_wd * wd_multipliers[1], log=True)
        
        # Batch size
        if 'batch_size' in search_space_config:
            batch_sizes = search_space_config['batch_size']
            # Ensure all batch sizes are integers
            batch_sizes = [int(b) for b in batch_sizes]
            search_params['batch_size'] = trial.suggest_categorical("batch_size", batch_sizes)
        else:
            default_batch = int(self.config_manager.get('training.batch_size', 64))
            batch_factors = search_ranges.get('batch_size_factors', [0.5, 1.0, 2.0])
            batch_sizes = [int(default_batch * factor) for factor in batch_factors]
            search_params['batch_size'] = trial.suggest_categorical("batch_size", batch_sizes)
        
        # Label smoothing
        if 'label_smoothing' in search_space_config:
            ls_range = search_space_config['label_smoothing']
            min_ls = float(ls_range[0])
            max_ls = float(ls_range[1])
            step = search_ranges.get('label_smoothing_step', 0.05)
            search_params['label_smoothing'] = trial.suggest_float("label_smoothing", min_ls, max_ls, step=step)
        else:
            default_ls = float(self.config_manager.get('training.label_smoothing', 0.1))
            ls_multiplier = search_ranges.get('label_smoothing_multiplier', 3.0)
            step = search_ranges.get('label_smoothing_step', 0.05)
            search_params['label_smoothing'] = trial.suggest_float("label_smoothing", 0.0, default_ls * ls_multiplier, step=step)
        
        # Clip value
        default_clip = float(self.config_manager.get('training.clip_value', 1.0))
        clip_range = search_ranges.get('clip_value_range', [0.5, 2.0])
        clip_step = search_ranges.get('clip_value_step', 0.1)
        search_params['clip_value'] = trial.suggest_float("clip_value", clip_range[0], max(clip_range[1], default_clip * 2), step=clip_step)
        
        # Enhanced finetuning features
        if 'adaptive_lr' in search_space_config:
            search_params['adaptive_lr'] = trial.suggest_categorical("adaptive_lr", search_space_config['adaptive_lr'])
        else:
            default_adaptive = self.config_manager.get('training.adaptive_lr', True)
            search_params['adaptive_lr'] = trial.suggest_categorical("adaptive_lr", [not default_adaptive, default_adaptive])
        
        # Multi LR
        default_multi_lr = self.config_manager.get('training.multi_lr', True)
        search_params['multi_lr'] = trial.suggest_categorical("multi_lr", [not default_multi_lr, default_multi_lr])
        
        # Weighted sampler
        default_weighted = self.config_manager.get('data.weighted_sampler.enabled', True)
        search_params['use_weighted_sampler'] = trial.suggest_categorical("use_weighted_sampler", [not default_weighted, default_weighted])
        
        # Optimizer
        if 'optimizer' in search_space_config:
            optimizers = search_space_config['optimizer']
            search_params['optimizer'] = trial.suggest_categorical("optimizer", optimizers)
        else:
            default_optimizer = self.config_manager.get('training.optimizer', 'AdamW')
            available_optimizers = ["AdamW", "Lion", "SGD"]
            if default_optimizer not in available_optimizers:
                available_optimizers.append(default_optimizer)
            search_params['optimizer'] = trial.suggest_categorical("optimizer", available_optimizers)
        
        # Scheduler
        if 'scheduler' in search_space_config:
            schedulers = search_space_config['scheduler']
            search_params['scheduler'] = trial.suggest_categorical("scheduler", schedulers)
        else:
            default_scheduler = self.config_manager.get('training.scheduler.type', 'cosine')
            available_schedulers = ["cosine", "cosine_warmup", "plateau", "exponential"]
            if default_scheduler not in available_schedulers:
                available_schedulers.append(default_scheduler)
            search_params['scheduler'] = trial.suggest_categorical("scheduler", available_schedulers)
        
        # Data ORP fraction
        if 'data_ORP' in search_space_config:
            orp_range = search_space_config['data_ORP']
            # Ensure values are numeric
            min_orp = float(orp_range[0])
            max_orp = float(orp_range[1])
            step = search_ranges.get('orp_fraction_step', 0.1)
            search_params['data_ORP'] = trial.suggest_float("data_ORP", min_orp, max_orp, step=step)
        else:
            default_orp = float(self.config_manager.get('datasets.orp_train_fraction', 0.6))
            delta = search_ranges.get('orp_fraction_delta', 0.3)
            step = search_ranges.get('orp_fraction_step', 0.1)
            min_val = max(0.1, default_orp - delta)
            max_val = min(0.9, default_orp + delta)
            search_params['data_ORP'] = trial.suggest_float("data_ORP", min_val, max_val, step=step)
        
        # Early stopping
        default_patience = int(self.config_manager.get('training.early_stopping.patience', 20))
        patience_factors = search_ranges.get('early_stopping_patience_factor', [0.5, 2.0])
        patience_step = search_ranges.get('early_stopping_patience_step', 5)
        search_params['early_stopping_patience'] = trial.suggest_int("early_stopping_patience", 
                                                                     max(5, int(default_patience * patience_factors[0])), 
                                                                     int(default_patience * patience_factors[1]), 
                                                                     step=patience_step)
        
        default_delta = float(self.config_manager.get('training.early_stopping.delta', 0.001))
        delta_multipliers = search_ranges.get('early_stopping_delta_multiplier', [0.1, 10.0])
        search_params['early_stopping_delta'] = trial.suggest_float("early_stopping_delta", 
                                                                   default_delta * delta_multipliers[0], 
                                                                   default_delta * delta_multipliers[1], 
                                                                   log=True)
        
        # Pretrained weights - get from config
        default_weights_path = self.config_manager.get('model.pretrained_weights.path', "./cbramod/weights/pretrained_weights.pth")
        
        # Look for other weight files in the same directory
        from pathlib import Path
        weights_dir = Path(default_weights_path).parent
        weight_options = [default_weights_path]
        
        if weights_dir.exists():
            for weight_file in weights_dir.glob("*.pth"):
                weight_path = str(weight_file)
                if weight_path not in weight_options:
                    weight_options.append(weight_path)
        
        search_params['foundation_dir'] = trial.suggest_categorical("foundation_dir", weight_options)
        
        # Split seed for reproducibility analysis
        seed_range = search_ranges.get('split_seed_range', [0, 10000])
        search_params['split_seed'] = trial.suggest_int("split_seed", seed_range[0], seed_range[1])
        
        # Conditional parameters based on scheduler
        if search_params['scheduler'] == 'plateau':
            factor_range = search_ranges.get('plateau_factor_range', [0.3, 0.8])
            factor_step = search_ranges.get('plateau_factor_step', 0.1)
            patience_range = search_ranges.get('plateau_patience_range', [5, 15])
            search_params['plateau_factor'] = trial.suggest_float("plateau_factor", factor_range[0], factor_range[1], step=factor_step)
            search_params['plateau_patience'] = trial.suggest_int("plateau_patience", patience_range[0], patience_range[1])
        
        return search_params
    
    def _run_training_with_oom_handling(self, params, trial):
        """Run training with out-of-memory handling and batch size reduction"""
        
        memory_config = self.config_manager.get('advanced.hyperopt.memory_optimization', {})
        max_attempts = memory_config.get('max_batch_size_reduction_attempts', 3)
        reduce_on_oom = memory_config.get('reduce_batch_size_on_oom', True)
        
        original_batch_size = params.batch_size
        
        for attempt in range(max_attempts):
            try:
                # Clear GPU cache before each attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"🔄 Trial {trial.number}, attempt {attempt + 1}/{max_attempts} with batch_size={params.batch_size}")
                
                # Run training
                results = self.pipeline.run_enhanced_training(params)
                return results
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and reduce_on_oom and attempt < max_attempts - 1:
                    # Reduce batch size and try again
                    new_batch_size = max(1, params.batch_size // 2)
                    self.logger.warning(f"💾 OOM detected in trial {trial.number}, reducing batch_size from {params.batch_size} to {new_batch_size}")
                    
                    params.batch_size = new_batch_size
                    trial.set_user_attr(f"oom_attempt_{attempt + 1}", f"batch_size_reduced_to_{new_batch_size}")
                    
                    # Clear cache again
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    continue
                else:
                    # Re-raise if not OOM or no more attempts
                    raise
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Trial {trial.number} failed after {max_attempts} attempts with batch size reduction")
    
    @log_performance("hyperopt_trial")
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        
        with self.logger.context(trial_number=trial.number, operation="hyperopt_trial"):
            with self.error_handler.handle_errors("hyperparameter_trial", reraise=False, default_return=0.0):
                
                # Get search parameters
                search_params = self.define_search_space(trial)
                
                # Create trial-specific parameters
                trial_args = copy.deepcopy(self.base_args)
                
                # Ensure epochs is set before any comparisons
                if not hasattr(trial_args, 'epochs') or trial_args.epochs is None:
                    trial_args.epochs = 50  # Default for hyperopt trials
                
                # Update parameters with trial suggestions
                for key, value in search_params.items():
                    setattr(trial_args, key, value)
                
                # Force shorter training for hyperparameter search (configurable)
                validation_config = self.config_manager.get('advanced.hyperopt.validation', {})
                max_epochs_per_trial = validation_config.get('max_epochs_per_trial', 50)
                trial_args.epochs = min(trial_args.epochs, max_epochs_per_trial)
                
                # Create enhanced parameters - PRESERVE USER'S CRITICAL SETTINGS
                params = create_enhanced_params(self.config_manager, trial_args)
                
                # Apply memory optimization settings for hyperopt
                memory_config = self.config_manager.get('advanced.hyperopt.memory_optimization', {})
                if memory_config.get('enabled', True):
                    # Enable memory optimizations for hyperopt trials
                    if memory_config.get('gradient_checkpointing', True):
                        params.gradient_checkpointing = True
                    if memory_config.get('mixed_precision', True):
                        params.use_amp = True
                    
                    # Reduce memory usage for hyperopt
                    params.pin_memory = False  # Reduce memory usage
                    params.prefetch_factor = 2  # Reduce prefetch buffer
                
                # CRITICAL: Preserve user's dataset and classification choices
                # These should NOT be overridden by hyperparameter optimization
                if hasattr(self.base_args, 'datasets'):
                    params.datasets = self.base_args.datasets
                    params.dataset_names = [d.strip() for d in self.base_args.datasets.split(',')]
                    params.num_datasets = len(params.dataset_names)
                    self.logger.info(f"Using datasets: {params.dataset_names}")
                
                if hasattr(self.base_args, 'num_classes') and self.base_args.num_classes is not None:
                    # User explicitly requested a specific number of classes
                    class_scheme = f"{self.base_args.num_classes}_class"
                    params.classification_scheme = class_scheme
                    params.num_of_classes = int(self.base_args.num_classes)
                    self.logger.info(f"Using {self.base_args.num_classes}-class system (user override)")
                else:
                    # Use default from config
                    params.classification_scheme = "4_class"
                    if not hasattr(params, 'num_of_classes'):
                        params.num_of_classes = 4
                    self.logger.info("Using default 4-class system")
                
                # Add trial-specific hyperparameters (but not dataset/class configs)
                for key, value in search_params.items():
                    # Don't override critical user settings
                    if key not in ['datasets', 'dataset_names', 'num_datasets', 'classification_scheme', 'num_of_classes']:
                        setattr(params, key, value)
                
                # Add trial tracking
                params.trial_number = trial.number
                params.run_name = f"trial_{trial.number:03d}"
                
                self.logger.info(f"Starting trial with lr={search_params['lr']:.2e}, batch_size={search_params['batch_size']}, classes={params.num_of_classes}")
                
                # Run enhanced training with OOM handling
                try:
                    results = self._run_training_with_oom_handling(params, trial)
                except optuna.exceptions.TrialPruned:
                    self.logger.info("Trial was pruned")
                    raise
                except Exception as e:
                    # Log detailed error information
                    trial.set_user_attr("error_type", type(e).__name__)
                    trial.set_user_attr("error_message", str(e))
                    self.logger.exception(f"Trial failed with {type(e).__name__}")
                    return 0.0
                
                # Extract metrics
                kappa = results.get('kappa', 0.0)
                accuracy = results.get('accuracy', 0.0)
                f1_score = results.get('f1_score', 0.0)
                
                # Log results to trial
                trial.set_user_attr("test_kappa", kappa)
                trial.set_user_attr("test_accuracy", accuracy)
                trial.set_user_attr("test_f1", f1_score)
                trial.set_user_attr("search_params", search_params)
                trial.set_user_attr("classification_scheme", params.classification_scheme)
                trial.set_user_attr("num_of_classes", params.num_of_classes)
                trial.set_user_attr("datasets", getattr(params, 'dataset_names', ['ORP']))
                
                # Enhanced early stopping for poor performing trials
                validation_config = self.config_manager.get('advanced.hyperopt.validation', {})
                min_kappa_threshold = validation_config.get('min_kappa_threshold', 0.05)
                
                # Prune trials with very poor performance
                if kappa <= min_kappa_threshold:
                    self.logger.warning(f"Pruning trial due to very low kappa: {kappa:.4f} <= {min_kappa_threshold}")
                    trial.set_user_attr("pruned_reason", f"low_kappa_{kappa:.4f}")
                    raise optuna.exceptions.TrialPruned()
                
                # Additional early stopping criteria
                if accuracy < 0.25:  # Random chance for 4-class is 0.25
                    self.logger.warning(f"Pruning trial due to very low accuracy: {accuracy:.4f} < 0.25")
                    trial.set_user_attr("pruned_reason", f"low_accuracy_{accuracy:.4f}")
                    raise optuna.exceptions.TrialPruned()
                
                if f1_score < 0.20:  # Very poor F1 score
                    self.logger.warning(f"Pruning trial due to very low f1_score: {f1_score:.4f} < 0.20")
                    trial.set_user_attr("pruned_reason", f"low_f1_{f1_score:.4f}")
                    raise optuna.exceptions.TrialPruned()
                
                # Check for NaN or infinite values
                if not np.isfinite(kappa) or not np.isfinite(accuracy) or not np.isfinite(f1_score):
                    self.logger.warning(f"Pruning trial due to invalid metrics: kappa={kappa}, acc={accuracy}, f1={f1_score}")
                    trial.set_user_attr("pruned_reason", "invalid_metrics")
                    raise optuna.exceptions.TrialPruned()
                
                self.logger.info(f"Trial completed successfully: kappa={kappa:.4f}, acc={accuracy:.4f}, f1={f1_score:.4f}")
                
                return kappa
    
    def optimize(self, n_trials: int = 20, resume: bool = True) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        # Create study
        study = self.create_study(n_trials, resume)
        
        # Get execution settings
        execution_config = self.config_manager.get('advanced.hyperopt.execution', {})
        timeout_per_trial = execution_config.get('timeout_per_trial', 3600)  # 1 hour default
        
        self.logger.info(f"🔍 Starting hyperparameter optimization with {n_trials} trials")
        self.logger.info("🔄 Using sequential execution")
        
        # Run optimization
        try:
            study.optimize(self.objective, n_trials=n_trials, timeout=timeout_per_trial)
        except KeyboardInterrupt:
            self.logger.info("🛑 Optimization interrupted by user")
        
        # Generate results
        results = self.generate_results(study)
        
        return results
    
    def generate_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Generate comprehensive results from study"""
        
        if not study.trials:
            self.logger.warning("No completed trials found")
            return {"error": "No completed trials"}
        
        # Get best trial
        best_trial = study.best_trial
        
        # Get top trials by test kappa
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            return {"error": "No completed trials"}
        
        top_trials = sorted(
            completed_trials, 
            key=lambda t: t.user_attrs.get("test_kappa", -1), 
            reverse=True
        )[:5]
        
        # Create results summary
        results = {
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "n_completed": len(completed_trials),
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs
            },
            "top_trials": [
                {
                    "number": t.number,
                    "val_kappa": t.value,
                    "test_kappa": t.user_attrs.get("test_kappa", 0),
                    "test_accuracy": t.user_attrs.get("test_accuracy", 0),
                    "test_f1": t.user_attrs.get("test_f1", 0),
                    "params": t.params
                }
                for t in top_trials
            ]
        }
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save optimization results"""
        
        # Save JSON results
        results_file = self.study_dir / f"{self.study_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV for analysis
        if "top_trials" in results:
            df_data = []
            for trial in results["top_trials"]:
                row = {
                    "trial_number": trial["number"],
                    "val_kappa": trial["val_kappa"],
                    "test_kappa": trial["test_kappa"],
                    "test_accuracy": trial["test_accuracy"],
                    "test_f1": trial["test_f1"]
                }
                row.update(trial["params"])
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = self.study_dir / f"{self.study_name}_trials.csv"
            df.to_csv(csv_file, index=False)
            
            self.logger.info(f"💾 Results saved to {results_file}")
            self.logger.info(f"📊 Trial data saved to {csv_file}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print optimization results"""
        
        print("\n" + "=" * 80)
        print("🎯 HYPERPARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)
        
        print(f"Study: {results['study_name']}")
        print(f"Total trials: {results['n_trials']}")
        print(f"Completed trials: {results['n_completed']}")
        
        if "best_trial" in results:
            best = results["best_trial"]
            print(f"\n🏆 Best Trial (by validation kappa):")
            print(f"  Trial #{best['number']}")
            print(f"  Validation Kappa: {best['value']:.4f}")
            test_kappa = best['user_attrs'].get('test_kappa', 0.0)
            test_accuracy = best['user_attrs'].get('test_accuracy', 0.0)
            test_f1 = best['user_attrs'].get('test_f1', 0.0)
            
            print(f"  Test Kappa: {test_kappa:.4f}" if isinstance(test_kappa, (int, float)) else f"  Test Kappa: {test_kappa}")
            print(f"  Test Accuracy: {test_accuracy:.4f}" if isinstance(test_accuracy, (int, float)) else f"  Test Accuracy: {test_accuracy}")
            print(f"  Test F1: {test_f1:.4f}" if isinstance(test_f1, (int, float)) else f"  Test F1: {test_f1}")
            
            print(f"\n📋 Best Parameters:")
            for key, value in best['params'].items():
                print(f"  {key}: {value}")
        
        if "top_trials" in results and results["top_trials"]:
            print(f"\n🔝 Top 5 Trials by Test Kappa:")
            for i, trial in enumerate(results["top_trials"], 1):
                print(f"  #{i} Trial {trial['number']}: "
                      f"Val={trial['val_kappa']:.4f}, "
                      f"Test={trial['test_kappa']:.4f}, "
                      f"Acc={trial['test_accuracy']:.4f}, "
                      f"F1={trial['test_f1']:.4f}")
        
        print("=" * 80)

def run_enhanced_hyperopt(config_manager: ConfigManager, args, n_trials: int = 20):
    """Run enhanced hyperparameter optimization"""
    
    optimizer = EnhancedHyperparameterOptimizer(config_manager, args)
    results = optimizer.optimize(n_trials=n_trials, resume=True)
    
    return results