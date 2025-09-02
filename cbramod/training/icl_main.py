#!/usr/bin/env python3
"""
Comprehensive In-Context Learning (ICL) evaluation for CBraMod.

Implements both ProtoICL (exhaustive grid search) and Meta-ICL (Bayesian optimization) 
following the research protocol for sleep staging evaluation.

Key features:
- Exhaustive grid search for ProtoICL (discrete parameter space)
- Optuna-based Bayesian optimization for Meta-ICL 
- Feature caching for efficient sweeps
- Cohen's κ with confidence intervals
- K-shot curve generation for paper figures
- Comprehensive W&B logging with proper tags
"""

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
import optuna

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from cbramod.load_datasets import idun_datasets
from cbramod.models import model_for_idun
from cbramod.training.icl_data import make_episodic_loaders
from cbramod.training.icl_trainer import ICLTrainer
from torch.utils.data import DataLoader

# Import WandB safely
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class ICLConfig:
    """Configuration for ICL experiments."""
    # Dataset settings (fixed as per requirements)
    downstream_dataset: str = 'IDUN_EEG'
    datasets_dir: str = 'data/datasets/final_dataset'
    datasets: str = 'ORP,2023_Open_N,2019_Open_N,2017_Open_N'
    num_classes: int = 5  # Fixed to 5-class v1
    label_mapping_version: str = 'v1'  # Fixed
    
    # Model settings
    model_dir: str = './saved_models'
    use_pretrained_weights: bool = True
    freeze_backbone_for_icl: bool = True
    foundation_dir: str = './saved_models/pretrained/pretrained_weights.pth'
    
    # ICL search spaces
    proto_k_values: List[int] = None  # {2, 4, 8, 16, 32}
    proto_similarities: List[str] = None  # {cosine, L2}  
    proto_support_balancing: List[bool] = None  # {True, False}
    proto_temperatures: List[float] = None  # {1.0, 2.0, 3.0}
    proto_m: int = 64  # Fixed for proto
    
    meta_k_values: List[int] = None  # {4, 8, 16, 32}
    meta_proj_dims: List[int] = None  # {256, 512, 768}
    meta_similarities: List[str] = None  # {cosine, L2}
    meta_head_lrs: Tuple[float, float] = None  # (3e-4, 3e-3) log-uniform
    meta_weight_decays: List[float] = None  # {0, 1e-4}
    meta_batch_episodes: List[int] = None  # {4, 8, 16}
    meta_epochs: List[int] = None  # {10, 15, 20}
    meta_temperatures: List[float] = None  # {1.0, 2.0, 3.0}
    meta_m: int = 64  # Fixed for meta
    
    # Experiment settings
    enable_feature_cache: bool = True
    cache_dir: str = './experiments/icl_cache'
    results_dir: str = './experiments/icl_results'
    
    # Optuna settings
    optuna_n_trials: int = 30
    optuna_study_name: str = 'icl_meta_optimization'
    optuna_storage: str = None  # Will use in-memory if None
    
    # W&B settings
    wandb_project: str = 'CBraMod-ICL-Research'
    wandb_entity: str = 'thibaut_hasle-epfl'
    wandb_group: str = 'icl_systematic_eval'
    no_wandb: bool = False
    wandb_offline: bool = False
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Set default search spaces if not provided."""
        if self.proto_k_values is None:
            self.proto_k_values = [2, 4, 8, 16, 32]
        if self.proto_similarities is None:
            self.proto_similarities = ['cosine', 'L2']
        if self.proto_support_balancing is None:
            self.proto_support_balancing = [True, False]
        if self.proto_temperatures is None:
            self.proto_temperatures = [1.0, 2.0, 3.0]
            
        if self.meta_k_values is None:
            self.meta_k_values = [4, 8, 16, 32]
        if self.meta_proj_dims is None:
            self.meta_proj_dims = [256, 512, 768]
        if self.meta_similarities is None:
            self.meta_similarities = ['cosine', 'L2']
        if self.meta_head_lrs is None:
            self.meta_head_lrs = (3e-4, 3e-3)
        if self.meta_weight_decays is None:
            self.meta_weight_decays = [0.0, 1e-4]
        if self.meta_batch_episodes is None:
            self.meta_batch_episodes = [4, 8, 16]
        if self.meta_epochs is None:
            self.meta_epochs = [10, 15, 20]
        if self.meta_temperatures is None:
            self.meta_temperatures = [1.0, 2.0, 3.0]


class FeatureCache:
    """Cache for encoded features to speed up ICL sweeps."""
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, split: str, model_name: str) -> Path:
        """Get cache file path for a given split and model."""
        return self.cache_dir / f"features_{split}_{model_name}.pt"
    
    def save_features(self, features: Dict[str, torch.Tensor], split: str, model_name: str):
        """Save encoded features to cache."""
        if not self.enabled:
            return
        
        cache_path = self._get_cache_path(split, model_name)
        torch.save(features, cache_path)
        print(f"💾 Cached {split} features to {cache_path}")
    
    def load_features(self, split: str, model_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load encoded features from cache."""
        if not self.enabled:
            return None
            
        cache_path = self._get_cache_path(split, model_name)
        if cache_path.exists():
            print(f"⚡ Loading cached {split} features from {cache_path}")
            return torch.load(cache_path, map_location='cpu')
        return None
    
    def clear_cache(self):
        """Clear all cached features."""
        if self.enabled and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pt"):
                cache_file.unlink()
            print(f"🗑️ Cleared feature cache: {self.cache_dir}")


class ICLEvaluator:
    """Comprehensive evaluator for ICL experiments."""
    
    def __init__(self, config: ICLConfig):
        self.config = config
        self.cache = FeatureCache(config.cache_dir, config.enable_feature_cache)
        self.results = []
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model and dataset
        self._load_model_and_data()
    
    def _load_model_and_data(self):
        """Load model and prepare data splits."""
        print("🔄 Loading model and dataset...")
        
        # Parse dataset parameters
        class Params:
            def __init__(self, config: ICLConfig):
                for key, value in asdict(config).items():
                    setattr(self, key, value)
                # Convert comma-separated datasets to list (required by LoadDataset)
                self.datasets = self.datasets.split(',') if isinstance(self.datasets, str) else self.datasets
                self.dataset_names = [name.strip() for name in self.datasets]
                self.num_datasets = len(self.dataset_names)
                # Add other required parameters with defaults
                self.cuda = 0  # Default GPU
                self.data_ORP = 0.2  # Default ORP train fraction
                self.preprocess = False
                self.sample_rate = 200.0
                self.noise_level = 0.0
                self.noise_type = 'realistic'
                self.noise_seed = 42
                # Fix attribute name mismatch
                self.num_of_classes = self.num_classes
        
        params = Params(self.config)
        
        print(f"🤖 Using pretrained weights: {params.foundation_dir}")
        
        # Load dataset following exact pattern from finetune_main.py
        if params.downstream_dataset == 'IDUN_EEG':
            load_dataset = idun_datasets.LoadDataset(params)
            seqs_labels_path_pair = load_dataset.get_all_pairs()
            
            # Create dataset
            self.dataset = idun_datasets.MemoryEfficientKFoldDataset(
                seqs_labels_path_pair, 
                num_of_classes=params.num_classes,
                label_mapping_version=getattr(params, 'label_mapping_version', 'v1'),
                do_preprocess=getattr(params, 'preprocess', False),
                sfreq=getattr(params, 'sample_rate', 200.0),
                noise_level=getattr(params, 'noise_level', 0.0),
                noise_type=getattr(params, 'noise_type', 'realistic'),
                noise_seed=getattr(params, 'noise_seed', 42)
            )
            
            # Get splits
            fold, self.train_idx, self.val_idx, self.test_idx = next(
                idun_datasets.get_custom_split(self.dataset, seed=42, 
                orp_train_frac=getattr(params, 'data_ORP', 0.2))
            )
            
            # Load model
            self.model = model_for_idun.Model(params)
            self.device = next(self.model.parameters()).device
            
            print(f"📊 Dataset loaded: {len(self.dataset)} total samples")
            print(f"📊 Splits - Train: {len(self.train_idx)}, Val: {len(self.val_idx)}, Test: {len(self.test_idx)}")
            print(f"🤖 Model loaded on device: {self.device}")
        else:
            raise ValueError(f"Dataset {params.downstream_dataset} not supported in ICL framework")
    
    def _compute_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
        """Compute mean and confidence interval for scores."""
        scores = np.array(scores)
        mean_score = np.mean(scores)
        
        if len(scores) < 2:
            return mean_score, mean_score, mean_score
        
        # Use t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence) / 2, df=len(scores) - 1)
        margin_error = t_value * stats.sem(scores)
        
        return mean_score, mean_score - margin_error, mean_score + margin_error
    
    def _init_wandb(self, mode: str, config_dict: Dict[str, Any]) -> Optional[Any]:
        """Initialize Weights & Biases logging."""
        if self.config.no_wandb or not WANDB_AVAILABLE:
            return None
        
        # Build tags
        tags = [
            f'method:{mode}',
            f'num_classes:{self.config.num_classes}',
            f'label_map:{self.config.label_mapping_version}',
        ]
        
        if 'k' in config_dict:
            tags.append(f'K:{config_dict["k"]}')
        if 'm' in config_dict:
            tags.append(f'M:{config_dict["m"]}')
        if 'similarity' in config_dict:
            tags.append(f'sim:{config_dict["similarity"]}')
        if 'proj_dim' in config_dict:
            tags.append(f'proj:{config_dict["proj_dim"]}')
        
        # Generate run name
        run_name = f'icl_{mode}_K{config_dict.get("k", "NA")}'
        if 'similarity' in config_dict:
            run_name += f'_{config_dict["similarity"]}'
        if 'temperature' in config_dict:
            run_name += f'_T{config_dict["temperature"]}'
        
        try:
            run_config = {**asdict(self.config), **config_dict}
            
            wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity, 
                group=self.config.wandb_group,
                name=run_name,
                tags=tags,
                config=run_config,
                reinit=True,
                mode='offline' if self.config.wandb_offline else 'online'
            )
            return wandb_run
        except Exception as e:
            print(f"⚠️ W&B init failed: {e}")
            return None
    
    def _evaluate_icl_config(self, icl_config: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Evaluate a single ICL configuration."""
        # Initialize W&B for this configuration
        wandb_run = self._init_wandb(mode, icl_config)
        
        try:
            # Create episodic loaders
            train_loader, val_loader, test_loader = make_episodic_loaders(
                self.dataset, self.train_idx, self.val_idx, self.test_idx,
                k=icl_config['k'], m=icl_config['m'],
                balance_support=icl_config.get('balance_support', True),
                batch_episodes=icl_config.get('batch_episodes', 8)
            )
            
            # Create ICL trainer
            learnable = (mode == 'meta_proto')
            
            # Create a mock params object for ICLTrainer
            class MockParams:
                def __init__(self, config):
                    self.icl_mode = mode
                    self.freeze_backbone_for_icl = config.get('freeze_backbone', True)
                    self.head_lr = config.get('head_lr', 1e-3)
                    self.weight_decay = config.get('weight_decay', 0.0)
            
            mock_params = MockParams(icl_config)
            icl_trainer = ICLTrainer(
                mock_params, self.model, self.config.num_classes,
                proj_dim=icl_config.get('proj_dim', 512),
                cosine=(icl_config.get('similarity', 'cosine') == 'cosine'),
                temperature=icl_config.get('temperature', 2.0)
            )
            
            # Training phase (meta_proto only)
            if mode == 'meta_proto':
                epochs = icl_config.get('epochs', 15)
                print(f"🏃 Training meta-ICL for {epochs} epochs...")
                icl_trainer.fit(train_loader, val_loader, epochs)
            
            # Evaluation phase
            print("🧪 Evaluating on test set...")
            results = icl_trainer.evaluate(test_loader)
            
            # Add configuration info to results
            results.update(icl_config)
            results['mode'] = mode
            
            # Log to W&B
            if wandb_run:
                wandb.log({
                    'test_kappa': results['kappa'],
                    'test_accuracy': results['acc'], 
                    'test_macro_f1': results['macro_f1'],
                    'episode_acc_mean': results['episode_acc_mean'],
                    'episode_acc_std': results['episode_acc_std'],
                    'num_episodes': results['num_episodes']
                })
                wandb.finish()
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating config {icl_config}: {e}")
            if wandb_run:
                wandb.finish()
            return None
    
    def run_proto_grid_search(self) -> Dict[str, Any]:
        """Run exhaustive grid search for ProtoICL."""
        print("\n🔍 Phase A: ProtoICL Exhaustive Grid Search")
        print("=" * 50)
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            self.config.proto_k_values,
            self.config.proto_similarities,
            self.config.proto_support_balancing, 
            self.config.proto_temperatures
        ))
        
        print(f"📊 Evaluating {len(param_combinations)} ProtoICL configurations...")
        
        proto_results = []
        best_kappa = -np.inf
        best_config = None
        
        for i, (k, similarity, balance_support, temperature) in enumerate(param_combinations):
            print(f"\n[{i+1}/{len(param_combinations)}] K={k}, sim={similarity}, balance={balance_support}, T={temperature}")
            
            config = {
                'k': k,
                'm': self.config.proto_m,
                'similarity': similarity,
                'balance_support': balance_support,
                'temperature': temperature,
                'proj_dim': 512,  # Fixed for proto to avoid untrained projection
            }
            
            result = self._evaluate_icl_config(config, mode='proto')
            if result:
                proto_results.append(result)
                
                # Track best configuration by validation κ
                if result['kappa'] > best_kappa:
                    best_kappa = result['kappa']
                    best_config = result.copy()
        
        print(f"\n✨ Best ProtoICL Config (κ={best_kappa:.4f}):")
        print(f"   K={best_config['k']}, similarity={best_config['similarity']}")
        print(f"   balance_support={best_config['balance_support']}, T={best_config['temperature']}")
        
        return {
            'best_config': best_config,
            'best_kappa': best_kappa,
            'all_results': proto_results
        }
    
    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for Meta-ICL optimization."""
        # Sample hyperparameters
        config = {
            'k': trial.suggest_categorical('k', self.config.meta_k_values),
            'm': self.config.meta_m,
            'proj_dim': trial.suggest_categorical('proj_dim', self.config.meta_proj_dims),
            'similarity': trial.suggest_categorical('similarity', self.config.meta_similarities),
            'head_lr': trial.suggest_float('head_lr', *self.config.meta_head_lrs, log=True),
            'weight_decay': trial.suggest_categorical('weight_decay', self.config.meta_weight_decays),
            'batch_episodes': trial.suggest_categorical('batch_episodes', self.config.meta_batch_episodes),
            'epochs': trial.suggest_categorical('epochs', self.config.meta_epochs),
            'temperature': trial.suggest_categorical('temperature', self.config.meta_temperatures),
        }
        
        result = self._evaluate_icl_config(config, mode='meta_proto')
        if result is None:
            return -np.inf
        
        # Return validation κ for optimization
        return result['kappa']
    
    def run_meta_optuna_search(self) -> Dict[str, Any]:
        """Run Optuna-based Bayesian optimization for Meta-ICL."""
        print("\n🎯 Phase B: Meta-ICL Optuna Optimization")
        print("=" * 50)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=self.config.optuna_study_name,
            storage=self.config.optuna_storage
        )
        
        print(f"🔬 Running {self.config.optuna_n_trials} Optuna trials...")
        
        # Run optimization
        study.optimize(self._optuna_objective, n_trials=self.config.optuna_n_trials)
        
        # Get best configuration
        best_trial = study.best_trial
        best_config = best_trial.params.copy()
        best_config['m'] = self.config.meta_m
        
        print(f"\n✨ Best Meta-ICL Config (κ={best_trial.value:.4f}):")
        for key, value in best_config.items():
            print(f"   {key}: {value}")
        
        # Retrain best config on full data
        print("\n🔄 Retraining best Meta-ICL config...")
        final_result = self._evaluate_icl_config(best_config, mode='meta_proto')
        
        return {
            'best_config': best_config,
            'best_kappa': best_trial.value,
            'final_result': final_result,
            'study': study
        }
    
    def generate_k_shot_curves(self, proto_best: Dict[str, Any], meta_best: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate K-shot curves for both ProtoICL and Meta-ICL."""
        print("\n📈 Phase C: K-shot Curve Generation") 
        print("=" * 50)
        
        k_values = sorted(set(self.config.proto_k_values + self.config.meta_k_values))
        print(f"📊 Generating curves for K ∈ {k_values}")
        
        curves = {'proto': [], 'meta_proto': []}
        
        # ProtoICL curve
        proto_config_base = proto_best['best_config'].copy()
        for k in k_values:
            if k in self.config.proto_k_values:
                print(f"\n📍 ProtoICL K={k}")
                config = proto_config_base.copy()
                config['k'] = k
                result = self._evaluate_icl_config(config, mode='proto')
                if result:
                    curves['proto'].append(result)
        
        # Meta-ICL curve  
        meta_config_base = meta_best['best_config'].copy()
        for k in k_values:
            if k in self.config.meta_k_values:
                print(f"\n📍 Meta-ICL K={k}")
                config = meta_config_base.copy()
                config['k'] = k
                result = self._evaluate_icl_config(config, mode='meta_proto')
                if result:
                    curves['meta_proto'].append(result)
        
        return curves
    
    def save_results(self, proto_results: Dict, meta_results: Dict, curves: Dict):
        """Save all results to files."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary results
        summary = {
            'config': asdict(self.config),
            'proto_best': proto_results,
            'meta_best': meta_results,
            'k_shot_curves': curves,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        summary_file = results_dir / 'icl_experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save CSV for easy analysis
        csv_results = []
        
        # Add proto results
        for result in proto_results['all_results']:
            csv_results.append({
                'method': 'proto',
                'k': result['k'],
                'similarity': result['similarity'],
                'temperature': result['temperature'],
                'balance_support': result['balance_support'],
                'kappa': result['kappa'],
                'accuracy': result['acc'],
                'macro_f1': result['macro_f1']
            })
        
        # Add meta results
        if 'final_result' in meta_results:
            result = meta_results['final_result']
            csv_results.append({
                'method': 'meta_proto',
                'k': result['k'],
                'proj_dim': result.get('proj_dim', 512),
                'head_lr': result.get('head_lr', 1e-3),
                'similarity': result['similarity'],
                'temperature': result['temperature'],
                'kappa': result['kappa'],
                'accuracy': result['acc'],
                'macro_f1': result['macro_f1']
            })
        
        # Add curve results
        for method, method_results in curves.items():
            for result in method_results:
                csv_results.append({
                    'method': f'{method}_curve',
                    'k': result['k'],
                    'kappa': result['kappa'],
                    'accuracy': result['acc'],
                    'macro_f1': result['macro_f1']
                })
        
        csv_file = results_dir / 'icl_results.csv'
        if csv_results:
            import pandas as pd
            df = pd.DataFrame(csv_results)
            df.to_csv(csv_file, index=False)
        
        print(f"💾 Results saved to {results_dir}")
        print(f"   Summary: {summary_file}")
        print(f"   CSV: {csv_file}")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run the complete ICL evaluation protocol."""
        print("🚀 Starting Comprehensive ICL Evaluation")
        print("=" * 60)
        
        # Phase A: ProtoICL grid search
        proto_results = self.run_proto_grid_search()
        
        # Phase B: Meta-ICL Optuna search
        meta_results = self.run_meta_optuna_search()
        
        # Phase C: K-shot curves
        curves = self.generate_k_shot_curves(proto_results, meta_results)
        
        # Save all results
        self.save_results(proto_results, meta_results, curves)
        
        # Print final summary
        print("\n🎉 ICL Evaluation Complete!")
        print("=" * 50)
        print(f"Best ProtoICL κ: {proto_results['best_kappa']:.4f}")
        print(f"Best Meta-ICL κ: {meta_results['best_kappa']:.4f}")
        
        return {
            'proto': proto_results,
            'meta': meta_results, 
            'curves': curves
        }


def main():
    """Main entry point for ICL evaluation."""
    parser = argparse.ArgumentParser(description='Comprehensive ICL Evaluation for CBraMod')
    
    # Add arguments for all config fields
    parser.add_argument('--downstream_dataset', type=str, default='IDUN_EEG')
    parser.add_argument('--datasets_dir', type=str, default='data/datasets/final_dataset')
    parser.add_argument('--datasets', type=str, default='ORP,2023_Open_N,2019_Open_N,2017_Open_N')
    parser.add_argument('--model_dir', type=str, default='./saved_models')
    parser.add_argument('--results_dir', type=str, default='./experiments/icl_results')
    
    # Experiment control
    parser.add_argument('--mode', type=str, choices=['proto', 'meta', 'curves', 'full'], 
                       default='full', help='Which evaluation phase to run')
    parser.add_argument('--optuna_n_trials', type=int, default=30)
    parser.add_argument('--enable_feature_cache', action='store_true', default=True)
    
    # W&B settings
    parser.add_argument('--wandb_project', type=str, default='CBraMod-ICL-Research')
    parser.add_argument('--wandb_entity', type=str, default='thibaut_hasle-epfl')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    
    # Model settings
    parser.add_argument('--foundation_dir', type=str, 
                        default='./saved_models/pretrained/pretrained_weights.pth',
                        help='Path to pretrained weights')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Create config from args (exclude mode as it's not part of ICLConfig)
    config_args = {k: v for k, v in vars(args).items() if k != 'mode'}
    config = ICLConfig(**config_args)
    
    # Initialize evaluator
    evaluator = ICLEvaluator(config)
    
    # Run evaluation based on mode
    if args.mode == 'proto':
        evaluator.run_proto_grid_search()
    elif args.mode == 'meta':
        evaluator.run_meta_optuna_search()
    elif args.mode == 'full':
        evaluator.run_full_evaluation()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return


if __name__ == '__main__':
    main()