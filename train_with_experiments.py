"""
Enhanced training script for CBraMod with comprehensive experiment management.

This script demonstrates the complete experiment management system including:
- Configuration management
- Multi-run experiments with different test sets
- Comprehensive experiment tracking (MLflow + WandB)
- Model validation and performance regression detection
- Reproducibility management
- Model registry integration
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import configuration system
from config.utils import ConfigManager, init_config_system

# Import experiment management
from experiments.tracking import create_unified_tracker
from experiments.registry import LocalModelRegistry
from experiments.validation import EEGModelValidator
from experiments.reproducibility import ReproducibilityManager
from experiments.multi_run import MultiRunExperiment, EEGTestSetManager

# Import existing CBraMod components
from cbramod.datasets import idun_datasets
from cbramod.models import model_for_idun
from cbramod.Finetuning.finetune_trainer import Trainer

# Standard imports
import torch
import numpy as np
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Comprehensive experiment manager that orchestrates all experiment components.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize experiment manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        
        # Initialize components
        self.tracker = None
        self.model_registry = None
        self.validator = None
        self.reproducibility_manager = None
        self.multi_run_experiment = None
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup all experiment management components."""
        # Setup experiment tracker
        self.tracker = create_unified_tracker(
            experiment_name="cbramod_training",
            config=self.config
        )
        
        # Setup model registry
        registry_dir = self.config.get('paths', {}).get('model_registry_dir', 'model_registry')
        self.model_registry = LocalModelRegistry(registry_dir)
        
        # Setup model validator
        validation_config = self.config.get('validation', {})
        validation_config.update({
            'accuracy_threshold': 0.65,
            'kappa_threshold': 0.55,
            'f1_threshold': 0.65,
            'per_class_f1_threshold': 0.5,
            'sleep_stages': ['Wake', 'Light', 'Deep', 'REM'],
            'max_inference_time_ms': 500,
            'max_model_size_mb': 200
        })
        self.validator = EEGModelValidator(validation_config)
        
        # Setup reproducibility manager
        self.reproducibility_manager = ReproducibilityManager(self.config)
        
        # Setup test set manager for multi-run experiments
        test_set_config = {
            'test_split_ratio': 0.2,
            'val_split_ratio': 0.15,
            'stratify_by': 'subject',
            'ensure_subject_split': True
        }
        test_set_manager = EEGTestSetManager(test_set_config)
        
        # Setup multi-run experiment
        multi_run_config = {
            'n_runs': 3,
            'base_seed': self.config.get('reproducibility', {}).get('seed', 42),
            'vary_test_sets': True,
            'vary_seeds': False
        }
        self.multi_run_experiment = MultiRunExperiment(
            experiment_name="cbramod_multi_run",
            config=multi_run_config,
            test_set_manager=test_set_manager
        )
        
        logger.info("All experiment components initialized successfully")
    
    def run_single_experiment(self, hyperparameters: Dict[str, Any], 
                            test_set: Any, seed: int, run_id: str,
                            tracker=None, validator=None) -> Dict[str, Any]:
        """
        Run a single training experiment.
        
        Args:
            hyperparameters: Hyperparameters for this run
            test_set: Test dataset
            seed: Random seed
            run_id: Unique run identifier
            tracker: Experiment tracker
            validator: Model validator
            
        Returns:
            Dictionary with results
        """
        try:
            # Setup reproducible environment
            experiment_seed = self.reproducibility_manager.setup_reproducible_environment(
                master_seed=seed,
                experiment_id=run_id
            )
            
            # Capture environment state
            self.reproducibility_manager.capture_environment_state(run_id)
            self.reproducibility_manager.save_config_snapshot(run_id, self.config)
            
            # Load dataset
            params = self._create_params_from_hyperparameters(hyperparameters)
            load_dataset = idun_datasets.LoadDataset(params)
            seqs_labels_path_pair = load_dataset.get_all_pairs()
            
            # Use provided test set or create new one
            if hasattr(test_set, 'train'):
                # Test set already contains train/val/test splits
                train_loader = DataLoader(test_set.train, batch_size=params.batch_size, 
                                        shuffle=True, num_workers=params.num_workers)
                val_loader = DataLoader(test_set.val, batch_size=params.batch_size, 
                                      shuffle=False, num_workers=params.num_workers)
                test_loader = DataLoader(test_set.test, batch_size=params.batch_size, 
                                       shuffle=False, num_workers=params.num_workers)
            else:
                # Create splits from full dataset
                dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair)
                fold, train_idx, val_idx, test_idx = next(
                    idun_datasets.get_custom_split(dataset, seed=seed, orp_train_frac=params.data_ORP)
                )
                
                train_set = torch.utils.data.Subset(dataset, train_idx)
                val_set = torch.utils.data.Subset(dataset, val_idx)
                test_set_subset = torch.utils.data.Subset(dataset, test_idx)
                
                train_loader = DataLoader(train_set, batch_size=params.batch_size, 
                                        shuffle=True, num_workers=params.num_workers)
                val_loader = DataLoader(val_set, batch_size=params.batch_size, 
                                      shuffle=False, num_workers=params.num_workers)
                test_loader = DataLoader(test_set_subset, batch_size=params.batch_size, 
                                       shuffle=False, num_workers=params.num_workers)
            
            data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
            
            # Create and train model
            model = model_for_idun.Model(params)
            trainer = Trainer(params, data_loader, model)
            
            # Log model info to tracker
            if tracker:
                tracker.log_model_info(model, "cbramod")
                tracker.log_dataset_info({
                    'train_size': len(train_loader.dataset),
                    'val_size': len(val_loader.dataset),
                    'test_size': len(test_loader.dataset),
                    'batch_size': params.batch_size,
                    'num_workers': params.num_workers
                })
            
            # Train model
            logger.info(f"Starting training for {run_id}")
            kappa = trainer.train_for_multiclass()
            
            # Evaluate model
            acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)
            
            # Save model
            model_filename = f"{run_id}_model.pth"
            model_path = Path(self.config.get('paths', {}).get('model_dir', 'saved_models')) / model_filename
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            # Validate model if validator provided
            validation_results = []
            if validator:
                try:
                    validation_results = validator.validate_model(model, test_loader)
                    
                    # Log validation results to tracker
                    if tracker:
                        validation_summary = validator.get_summary()
                        for key, value in validation_summary.items():
                            tracker.log_metric(f"validation_{key}", value)
                        
                        # Log individual validation results
                        for result in validation_results:
                            if result.score is not None:
                                tracker.log_metric(f"validation_{result.test_name}_score", result.score)
                            tracker.log_param(f"validation_{result.test_name}_status", result.status.value)
                
                except Exception as e:
                    logger.error(f"Model validation failed: {e}")
            
            # Register model
            try:
                model_version = self.model_registry.register_model(
                    name="cbramod",
                    model_path=str(model_path),
                    description=f"CBraMod trained with {run_id}",
                    tags={
                        'run_id': run_id,
                        'accuracy': f"{acc:.4f}",
                        'kappa': f"{kappa:.4f}",
                        'f1_score': f"{f1:.4f}"
                    },
                    metadata={
                        'hyperparameters': hyperparameters,
                        'validation_results': [r.to_dict() for r in validation_results],
                        'seed': seed
                    },
                    run_id=run_id
                )
                
                # Add metrics to model registry
                self.model_registry.add_model_metrics(
                    "cbramod", 
                    model_version.version, 
                    {'accuracy': acc, 'kappa': kappa, 'f1_score': f1}
                )
                
                logger.info(f"Registered model version {model_version.version} in registry")
                
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
            
            # Prepare results
            results = {
                'metrics': {
                    'accuracy': acc,
                    'kappa': kappa,
                    'f1_score': f1
                },
                'model_path': str(model_path),
                'validation_results': validation_results,
                'metadata': {
                    'experiment_seed': experiment_seed.to_dict(),
                    'training_time': getattr(trainer, 'training_time', None),
                    'best_epoch': getattr(trainer, 'best_epoch', None)
                }
            }
            
            logger.info(f"Completed {run_id}: Acc={acc:.4f}, Kappa={kappa:.4f}, F1={f1:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run experiment {run_id}: {e}")
            raise
    
    def _create_params_from_hyperparameters(self, hyperparameters: Dict[str, Any]) -> argparse.Namespace:
        """Create parameters object from hyperparameters and config."""
        params = argparse.Namespace()
        
        # Set defaults from config
        training_config = self.config_manager.get_training_config()
        data_config = self.config_manager.get_data_config()
        
        # Basic parameters
        params.epochs = hyperparameters.get('epochs', training_config.get('epochs', 70))
        params.batch_size = hyperparameters.get('batch_size', training_config.get('batch_size', 64))
        params.lr = hyperparameters.get('learning_rate', training_config.get('learning_rate', 0.0001))
        params.weight_decay = hyperparameters.get('weight_decay', training_config.get('weight_decay', 0.0))
        params.optimizer = hyperparameters.get('optimizer', training_config.get('optimizer', 'AdamW'))
        
        # Data parameters
        params.num_workers = data_config.get('num_workers', 4)
        params.sample_rate = data_config.get('sample_rate', 200)
        params.data_ORP = hyperparameters.get('data_ORP', 0.6)
        
        # Model parameters
        params.num_of_classes = 4
        params.downstream_dataset = 'IDUN_EEG'
        params.use_pretrained_weights = True
        params.foundation_dir = self.config.get('model', {}).get('pretrained_weights', {}).get('path', 
                                                               'cbramod/weights/pretrained_weights.pth')
        
        # Other parameters
        params.cuda = self.config.get('device', {}).get('cuda', {}).get('device_id', 0)
        params.seed = self.config.get('reproducibility', {}).get('seed', 42)
        params.multi_lr = training_config.get('multi_lr', False)
        params.frozen = training_config.get('frozen', False)
        params.label_smoothing = training_config.get('label_smoothing', 0.1)
        params.clip_value = training_config.get('clip_value', 1.0)
        params.scheduler = training_config.get('scheduler', {}).get('type', 'cosine')
        
        # Dataset configuration
        params.datasets = 'ORP'
        params.dataset_names = ['ORP']
        params.num_datasets = 1
        params.class_weights = None
        params.use_weighted_sampler = False
        params.datasets_dir = hyperparameters.get('datasets_dir', 'Datasets/Final_dataset')
        
        return params
    
    def run_multi_run_experiment(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run multi-run experiment with different test sets.
        
        Args:
            hyperparameters: Hyperparameters to test (can include lists for grid search)
            
        Returns:
            Experiment results and analysis
        """
        logger.info("Starting multi-run experiment")
        
        # Load full dataset for test set creation
        params = self._create_params_from_hyperparameters(hyperparameters)
        load_dataset = idun_datasets.LoadDataset(params)
        seqs_labels_path_pair = load_dataset.get_all_pairs()
        full_dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair)
        
        # Run multi-run experiment
        results = self.multi_run_experiment.run_experiment(
            train_function=self.run_single_experiment,
            dataset=full_dataset,
            hyperparameters=hyperparameters,
            tracker=self.tracker,
            validator=self.validator
        )
        
        # Analyze results
        aggregated_metrics = self.multi_run_experiment.get_aggregated_metrics()
        
        # Find best hyperparameters
        try:
            best_params, best_kappa = self.multi_run_experiment.get_best_hyperparameters('kappa', maximize=True)
            logger.info(f"Best hyperparameters (by kappa): {best_params} -> {best_kappa:.4f}")
        except Exception as e:
            logger.warning(f"Could not determine best hyperparameters: {e}")
            best_params, best_kappa = None, None
        
        # Save results
        results_file = f"multi_run_results_{self.multi_run_experiment.experiment_name}.json"
        self.multi_run_experiment.save_results(results_file)
        
        # Create summary
        summary = {
            'total_runs': len(results),
            'aggregated_metrics': aggregated_metrics,
            'best_hyperparameters': best_params,
            'best_kappa': best_kappa,
            'results_file': results_file
        }
        
        logger.info(f"Multi-run experiment completed: {len(results)} runs")
        return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CBraMod training with experiment management')
    
    # Configuration arguments
    parser.add_argument('--config-env', type=str, default='development',
                        help='Configuration environment')
    parser.add_argument('--experiment-type', type=str, default='single',
                        choices=['single', 'multi-run'],
                        help='Type of experiment to run')
    
    # Hyperparameter arguments
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--data-ORP', type=float, default=None, help='Fraction of ORP data')
    
    # Dataset arguments
    parser.add_argument('--datasets-dir', type=str, default='Datasets/Final_dataset', help='Directory containing datasets')
    
    # Multi-run specific arguments
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs for multi-run experiment')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration system
        config_manager = ConfigManager(environment=args.config_env)
        
        # Initialize experiment manager
        experiment_manager = ExperimentManager(config_manager)
        
        # Setup hyperparameters
        hyperparameters = {}
        if args.epochs is not None:
            hyperparameters['epochs'] = args.epochs
        if args.batch_size is not None:
            hyperparameters['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            hyperparameters['learning_rate'] = args.learning_rate
        if args.data_ORP is not None:
            hyperparameters['data_ORP'] = args.data_ORP
        if args.datasets_dir is not None:
            hyperparameters['datasets_dir'] = args.datasets_dir
        
        # For demonstration, add some hyperparameter variations for multi-run
        if args.experiment_type == 'multi-run' and len([k for k in hyperparameters.keys() if k != 'datasets_dir']) == 0:
            hyperparameters.update({
                'learning_rate': [0.0001, 0.0005],
                'batch_size': [512,1024],
                'epochs': 20  # Shorter for demo
            })
        
        # Run experiment
        if args.experiment_type == 'single':
            # Run single experiment
            experiment_manager.tracker.start_run(
                run_name="single_experiment",
                tags={'experiment_type': 'single'}
            )
            
            try:
                # Create test set
                params = experiment_manager._create_params_from_hyperparameters(hyperparameters)
                load_dataset = idun_datasets.LoadDataset(params)
                seqs_labels_path_pair = load_dataset.get_all_pairs()
                dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair)
                
                fold, train_idx, val_idx, test_idx = next(
                    idun_datasets.get_custom_split(dataset, seed=42, orp_train_frac=params.data_ORP)
                )
                
                train_set = torch.utils.data.Subset(dataset, train_idx)
                val_set = torch.utils.data.Subset(dataset, val_idx)
                test_set = torch.utils.data.Subset(dataset, test_idx)
                
                # Create test set object
                class TestSetContainer:
                    def __init__(self, train, val, test):
                        self.train = train
                        self.val = val
                        self.test = test
                
                test_set_container = TestSetContainer(train_set, val_set, test_set)
                
                # Run experiment
                results = experiment_manager.run_single_experiment(
                    hyperparameters=hyperparameters,
                    test_set=test_set_container,
                    seed=42,
                    run_id="single_experiment",
                    tracker=experiment_manager.tracker,
                    validator=experiment_manager.validator
                )
                
                print("\n🎯 Single Experiment Results:")
                print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
                print(f"  Kappa: {results['metrics']['kappa']:.4f}")
                print(f"  F1 Score: {results['metrics']['f1_score']:.4f}")
                
            finally:
                experiment_manager.tracker.end_run()
        
        else:
            # Run multi-run experiment
            summary = experiment_manager.run_multi_run_experiment(hyperparameters)
            
            print("\n🚀 Multi-Run Experiment Summary:")
            print(f"  Total runs: {summary['total_runs']}")
            if summary['best_hyperparameters']:
                print(f"  Best hyperparameters: {summary['best_hyperparameters']}")
                print(f"  Best kappa: {summary['best_kappa']:.4f}")
            
            print("\n📊 Aggregated Metrics:")
            for metric, stats in summary['aggregated_metrics'].items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        print("\n✅ Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == '__main__':
    main()