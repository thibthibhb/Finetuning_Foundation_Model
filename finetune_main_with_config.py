"""
Enhanced finetune_main.py using the new configuration management system.

This script demonstrates how to integrate the new config system with existing code
while maintaining backward compatibility.
"""

import sys
import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import configuration system
from config import load_config, ConfigError, ValidationError
from config.utils import (
    init_config_system, 
    ConfigManager, 
    convert_config_to_args,
    get_wandb_config,
    setup_logging
)

# Import existing modules
from cbramod.datasets import idun_datasets
from cbramod.Finetuning.finetune_trainer import Trainer
from cbramod.models import model_for_idun
from cbramod.Finetuning.finetune_tuner import run_optuna_tuning
from statistics import mean, stdev
from torch.utils.data import DataLoader
import wandb


def setup_seed(seed):
    """Setup random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_params_from_config(config_manager: ConfigManager, args: argparse.Namespace) -> argparse.Namespace:
    """
    Create parameters object by merging configuration and command line arguments.
    Command line arguments take precedence over configuration.
    
    Args:
        config_manager: Configuration manager instance
        args: Command line arguments
        
    Returns:
        Merged parameters object
    """
    # Get configuration as namespace
    config_params = convert_config_to_args(config_manager.config)
    
    # Create new params object
    params = argparse.Namespace()
    
    # Start with configuration values
    for key, value in vars(config_params).items():
        setattr(params, key, value)
    
    # Override with command line arguments (only if explicitly provided)
    for key, value in vars(args).items():
        if value is not None:  # Only override if explicitly set
            setattr(params, key, value)
    
    # Map configuration to expected parameter names
    # Model parameters
    if hasattr(config_params, 'model_architecture'):
        params.model_name = config_params.model_architecture
    
    # Training parameters
    if hasattr(config_params, 'training_epochs'):
        params.epochs = config_params.training_epochs
    if hasattr(config_params, 'training_batch_size'):
        params.batch_size = config_params.training_batch_size
    if hasattr(config_params, 'training_learning_rate'):
        params.lr = config_params.training_learning_rate
    if hasattr(config_params, 'training_weight_decay'):
        params.weight_decay = config_params.training_weight_decay
    if hasattr(config_params, 'training_optimizer'):
        params.optimizer = config_params.training_optimizer
    if hasattr(config_params, 'training_clip_value'):
        params.clip_value = config_params.training_clip_value
    if hasattr(config_params, 'training_label_smoothing'):
        params.label_smoothing = config_params.training_label_smoothing
    
    # Data parameters
    if hasattr(config_params, 'data_sample_rate'):
        params.sample_rate = config_params.data_sample_rate
    if hasattr(config_params, 'data_num_workers'):
        params.num_workers = config_params.data_num_workers
    
    # Device parameters
    if hasattr(config_params, 'device_cuda_device_id'):
        params.cuda = config_params.device_cuda_device_id
    
    # Reproducibility
    if hasattr(config_params, 'reproducibility_seed'):
        params.seed = config_params.reproducibility_seed
    
    # Paths
    if hasattr(config_params, 'paths_datasets_dir'):
        params.datasets_dir = config_params.paths_datasets_dir
    if hasattr(config_params, 'paths_model_dir'):
        params.model_dir = config_params.paths_model_dir
    
    # Model weights
    if hasattr(config_params, 'model_pretrained_weights_enabled'):
        params.use_pretrained_weights = config_params.model_pretrained_weights_enabled
    if hasattr(config_params, 'model_pretrained_weights_path'):
        params.foundation_dir = config_params.model_pretrained_weights_path
    
    # Additional parameters for backward compatibility
    if not hasattr(params, 'downstream_dataset'):
        params.downstream_dataset = 'IDUN_EEG'
    if not hasattr(params, 'num_of_classes'):
        params.num_of_classes = 4
    if not hasattr(params, 'multi_lr'):
        params.multi_lr = config_manager.get('training.multi_lr', False)
    if not hasattr(params, 'frozen'):
        params.frozen = config_manager.get('training.frozen', False)
    if not hasattr(params, 'use_weighted_sampler'):
        params.use_weighted_sampler = config_manager.get('data.weighted_sampler.enabled', False)
    
    # Dataset-specific parameters
    if not hasattr(params, 'datasets'):
        params.datasets = 'ORP'
    if not hasattr(params, 'data_ORP'):
        params.data_ORP = 0.6
    if not hasattr(params, 'scheduler'):
        params.scheduler = config_manager.get('training.scheduler.type', 'cosine')
    
    # Compute derived parameters
    params.dataset_names = [name.strip() for name in params.datasets.split(',')]
    params.num_datasets = len(params.dataset_names)
    
    # Handle class weights
    weight_class_path = getattr(params, 'weight_class', None)
    if weight_class_path is not None and os.path.exists(weight_class_path + ".npy"):
        weights_array = np.load(weight_class_path + ".npy")
        params.class_weights = torch.tensor(weights_array, dtype=torch.float32).cuda()
    else:
        params.class_weights = None
    
    return params


def setup_experiment_tracking(config_manager: ConfigManager, params: argparse.Namespace) -> None:
    """Setup experiment tracking with WandB."""
    if config_manager.get('experiment.tracking.enabled', True):
        wandb_config = get_wandb_config(config_manager.config)
        
        # Initialize WandB
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=getattr(params, 'run_name', None),
            config=vars(params),
            mode=wandb_config['mode']
        )


def main():
    """Main training function with configuration management."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CBraMod finetuning with configuration management')
    
    # Configuration arguments
    parser.add_argument('--config-env', type=str, default=None,
                        help='Configuration environment (development, production, testing)')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='Configuration directory path')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name')
    parser.add_argument('--model-size', type=int, default=None,
                        help='Model size (total params)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='Optimizer (AdamW, SGD, Lion)')
    
    # Data arguments
    parser.add_argument('--datasets-dir', type=str, default=None,
                        help='Datasets directory')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names')
    parser.add_argument('--data-ORP', type=float, default=None,
                        help='Fraction of ORP data to use')
    
    # Device arguments
    parser.add_argument('--cuda', type=int, default=None,
                        help='CUDA device ID')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Experiment run name')
    
    # Legacy arguments for backward compatibility
    parser.add_argument('--downstream-dataset', type=str, default='IDUN_EEG',
                        help='Downstream dataset name')
    parser.add_argument('--num-of-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--weight-class', type=str, default=None,
                        help='Path to class weights file')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration system
        config_manager = ConfigManager(environment=args.config_env)
        
        # Create parameters by merging config and args
        params = create_params_from_config(config_manager, args)
        
        print("Configuration loaded successfully:")
        print(f"  Environment: {config_manager.config.get('_meta', {}).get('environment', 'unknown')}")
        print(f"  Model: {params.model_name if hasattr(params, 'model_name') else 'default'}")
        print(f"  Training epochs: {params.epochs}")
        print(f"  Batch size: {params.batch_size}")
        print(f"  Learning rate: {params.lr}")
        print(f"  Device: {config_manager.get_device()}")
        
        # Setup reproducibility
        setup_seed(params.seed)
        torch.cuda.set_device(params.cuda)
        
        # Setup experiment tracking
        setup_experiment_tracking(config_manager, params)
        
        # Run hyperparameter tuning if requested
        if params.tune:
            run_optuna_tuning(params)
            print("Tuning completed. Exiting.")
            return
        
        print(f'The downstream dataset is {params.downstream_dataset}')
        
        # Main training logic (same as original)
        if params.downstream_dataset == 'IDUN_EEG':
            load_dataset = idun_datasets.LoadDataset(params)
            seqs_labels_path_pair = load_dataset.get_all_pairs()
            
            # Load dataset
            dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair)
            
            # Use single subject-level split
            fold, train_idx, val_idx, test_idx = next(
                idun_datasets.get_custom_split(dataset, seed=42, orp_train_frac=params.data_ORP)
            )
            
            print(f"\n▶️ Using split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
            
            if params.num_subjects_train < 0:
                params.num_subjects_train = dataset.num_subjects_train
            
            # Create data subsets
            train_set = torch.utils.data.Subset(dataset, train_idx)
            val_set = torch.utils.data.Subset(dataset, val_idx)
            test_set = torch.utils.data.Subset(dataset, test_idx)
            
            # Create data loaders
            train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, 
                                    num_workers=params.num_workers)
            val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False, 
                                  num_workers=params.num_workers)
            test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, 
                                   num_workers=params.num_workers)
            
            data_loader = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
            }
            
            # Create model and trainer
            model = model_for_idun.Model(params)
            trainer = Trainer(params, data_loader, model)
            
            print(f"🚀 Training model on fixed split")
            kappa = trainer.train_for_multiclass()
            acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)
            
            print("\n📊 Evaluation Results:")
            print(f"  Kappa: {kappa:.4f}")
            print(f"  Acc:   {acc:.4f}")
            print(f"  F1:    {f1:.4f}")
            
            # Log results to WandB
            if config_manager.get('experiment.tracking.enabled', True):
                wandb.log({
                    'final/kappa': kappa,
                    'final/accuracy': acc,
                    'final/f1_score': f1
                })
                wandb.finish()
            
            return kappa, acc, f1
        
        print('Done!!!!!')
        
    except (ConfigError, ValidationError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()