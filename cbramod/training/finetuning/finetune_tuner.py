import sys
import os
import optuna
import copy
import argparse
import json
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from cbramod.load_datasets import idun_datasets
from cbramod.models import model_for_idun

try:
    from .finetune_trainer import Trainer
except ImportError:
    from finetune_trainer import Trainer

import torch
from torch.utils.data import DataLoader, Subset

def run_multi_eval(params, subjects):
    print("\n🔁 Starting multi-subject evaluation...")
    results = []
    for subj in subjects:
        print(f"  → Evaluating on subject: {subj}")
        config = copy.deepcopy(params)
        config.subject_id = subj

        dataset_loader = idun_datasets.LoadDataset(config)
        seqs_labels_path_pair = dataset_loader.get_all_pairs()
        dataset = idun_datasets.MemoryEfficientKFoldDataset(
            seqs_labels_path_pair, 
            num_of_classes=config.num_of_classes,
            label_mapping_version=getattr(config, 'label_mapping_version', 'v1'),
            do_preprocess=getattr(config, 'preprocess', False),
            sfreq=getattr(config, 'sample_rate', 200.0),
            noise_level=getattr(config, 'noise_level', 0.0),
            noise_type=getattr(config, 'noise_type', 'realistic'),
            noise_seed=getattr(config, 'noise_seed', 42)
        )

        fold, train_idx, val_idx, test_idx = next(idun_datasets.get_custom_split(dataset, seed=42, orp_train_frac=config.data_ORP))
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        model = model_for_idun.Model(config)
        trainer = Trainer(config, data_loader, model)

        trainer.train_for_multiclass()
        test_kappa, _, test_f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)

        results.append({"subject": subj, "test_kappa": test_kappa, "f1": test_f1})

    avg_kappa = mean([r['test_kappa'] for r in results])
    std_kappa = stdev([r['test_kappa'] for r in results])

    os.makedirs("experiments/results", exist_ok=True)
    with open(f"experiments/results/multi_eval_trial_results.json", "w") as f:
        json.dump({
            "params": vars(params),
            "avg_kappa": avg_kappa,
            "std_kappa": std_kappa,
            "per_subject": results
        }, f, indent=2)

    print(f"\n✅ Multi-subject evaluation done. Avg kappa: {avg_kappa:.4f}, Std: {std_kappa:.4f}\n")

def objective(trial, base_params, multi_eval=False, multi_eval_subjects=None):
    print("🚀 Starting new trial")
    params = copy.deepcopy(base_params)

    # Suggest hyperparams
    params.lr = trial.suggest_float("lr",  1e-5, 5e-3, log=True)
    params.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.01)
    params.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    params.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    params.clip_value = trial.suggest_float("clip_value", 0.0, 2.0, step=0.5)
    params.multi_lr = trial.suggest_categorical("multi_lr", [True, False])
    params.use_weighted_sampler = trial.suggest_categorical("use_weighted_sampler", [True, False])
    params.optimizer = trial.suggest_categorical("optimizer", ["AdamW","Lion"])
    params.scheduler = trial.suggest_categorical("scheduler", ["cosine"])
    params.head_type = trial.suggest_categorical("head_type", ["simple", "deep", "attention"]) #, "deep", "attention"
    params.use_focal_loss = trial.suggest_categorical("use_focal_loss", [False, True]) #True,
    params.datasets = trial.suggest_categorical("datasets", [
        "ORP",
        "ORP,2023_Open_N", 
        "ORP,2019_Open_N", 
        "ORP,2017_Open_N", 
        "ORP,2023_Open_N,2019_Open_N,2017_Open_N"
    ])
    print(f"🐛 DEBUG: Optuna selected datasets: {repr(params.datasets)}")
    
    # Process datasets after Optuna selection (this should happen here, not in main)
    params.dataset_names = [name.strip() for name in params.datasets.split(',')]
    params.num_datasets = len(params.dataset_names)
    print(f"🐛 DEBUG: Updated dataset_names: {params.dataset_names} (count: {params.num_datasets})")
    if params.use_focal_loss:
        params.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)
    else:
        params.focal_gamma = 2.0
    params.data_ORP = trial.suggest_float("data_ORP", 0.5 ,0.6, step=0.1)
    params.two_phase_training = trial.suggest_categorical("two_phase_training", [False, True]) #True, 

    if params.two_phase_training:
        params.phase1_epochs = trial.suggest_int("phase1_epochs", 2, 5)
        base_head_lr = trial.suggest_float("base_head_lr", 0.5, 2.0)
        head_to_backbone_ratio = trial.suggest_float("head_to_backbone_ratio", 5, 20)
        params.head_lr = params.lr * base_head_lr
        params.backbone_lr = params.head_lr / head_to_backbone_ratio
    else:
        pass

    params.foundation_dir = trial.suggest_categorical(
        "foundation_dir", [
            "./saved_models/pretrained/BEST__loss10527.31.pth",
            "./saved_models/pretrained/pretrained_weights.pth"
        ]
    )

    params.num_of_classes = trial.suggest_categorical("num_of_classes", [4, 5])
    params.label_mapping_version = trial.suggest_categorical("label_mapping_version", ["v0", "v1"])
    params.use_amp = trial.suggest_categorical("use_amp", [True, False])
    params.epochs = trial.suggest_int("epochs", 50, 200)
    params.use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
    # params.icl_mode = trial.suggest_categorical("icl_mode", ["none"]) #"none", "proto", cnp", "set COMMENTED OUT
    # if params.icl_mode != "none":
    #     params.k_support = trial.suggest_int("k_support", 1, 20)
    #     params.proto_temp = trial.suggest_float("proto_temp", 0.01, 1.0, log=True)
    #     params.icl_hidden = trial.suggest_categorical("icl_hidden", [128, 256, 512])
    #     params.icl_layers = trial.suggest_int("icl_layers", 1, 4)

    # params.use_metric_friendly_training = trial.suggest_categorical("use_metric_friendly_training", [True, False])
    # if params.use_metric_friendly_training:
    #     params.contrastive_weight = trial.suggest_float("contrastive_weight", 0.01, 0.5)
    #     params.prototypical_weight = trial.suggest_float("prototypical_weight", 0.01, 0.5)

    # params.use_temporal_smoothing = trial.suggest_categorical("use_temporal_smoothing", [True, False])
    # if params.use_temporal_smoothing:
    #     params.temporal_smoothing_window = trial.suggest_int("temporal_smoothing_window", 3, 7)

    params.preprocess = trial.suggest_categorical("preprocess", [True, False])

    # === Noise Injection for Robustness Analysis ===
    # Enable comprehensive noise injection parameter sweeps
    params.noise_level = trial.suggest_categorical("noise_level", [
        # 0.0,    # Clean baseline
        # 0.05,   # 5% noise (light)
        # 0.10,   # 10% noise (moderate) 
        0.15,   # 15% noise (heavy)
        0.20    # 20% noise (very heavy)
    ])
    
    # Only suggest noise type if noise is enabled
    if params.noise_level > 0.0:
        params.noise_type = trial.suggest_categorical("noise_type", [
            "realistic",    # Mixed artifacts (RECOMMENDED)
            "emg",         # Muscle artifacts only
            "movement",    # Motion artifacts only  
            "electrode",   # Contact artifacts only
            "gaussian"     # Simple white noise baseline
        ])
        # Fixed noise seed for reproducible robustness comparisons
        params.noise_seed = 42
    else:
        params.noise_type = "realistic"  # Default (unused when noise_level=0.0)
        params.noise_seed = 42

    params.frozen = trial.suggest_categorical("frozen", [True, False])
    
    load_dataset = idun_datasets.LoadDataset(params)
    seqs_labels_path_pair = load_dataset.get_all_pairs()
    dataset = idun_datasets.MemoryEfficientKFoldDataset(
        seqs_labels_path_pair, 
        num_of_classes=params.num_of_classes,
        label_mapping_version=getattr(params, 'label_mapping_version', 'v1'),
        do_preprocess=getattr(params, 'preprocess', False),
        sfreq=getattr(params, 'sample_rate', 200.0),
        noise_level=getattr(params, 'noise_level', 0.0),
        noise_type=getattr(params, 'noise_type', 'realistic'),
        noise_seed=getattr(params, 'noise_seed', 42)
    )
    seed = trial.suggest_int("split_seed", 0, 10000)
    fold, train_idx, val_idx, test_idx = next(idun_datasets.get_custom_split(dataset, seed=seed, orp_train_frac=params.data_ORP))

    if params.num_subjects_train < 0:
        params.num_subjects_train = dataset.num_subjects_train

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    model = model_for_idun.Model(params)
    trainer = Trainer(params, data_loader, model)

    try:
        kappa = trainer.train_for_multiclass(trial=trial)
    except optuna.exceptions.TrialPruned:
        raise

    acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)

    trial.set_user_attr("test_kappa", kappa)
    trial.set_user_attr("test_f1", f1)
    trial.set_user_attr("test_accuracy", acc)

    if kappa > 0.55 and multi_eval:
        if not multi_eval_subjects:
            multi_eval_subjects = ['S001', 'S002','S003', 'S004', 'S005', 'S006', 'S007', 'S009', 'S010', 'S012', 'S013', 'S014','S016']
        run_multi_eval(params, multi_eval_subjects)

    if kappa <= 0.05:
        raise optuna.exceptions.TrialPruned()

    return kappa

def run_optuna_tuning(params, multi_eval=False, multi_eval_subjects=None, n_trials=20):
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))
    study.optimize(lambda trial: objective(trial, params, multi_eval, multi_eval_subjects), n_trials=n_trials)

    best_trial = study.best_trial
    print("=== Top Trials by Test Kappa ===")
    sorted_trials = sorted(study.trials, key=lambda t: t.user_attrs.get("test_kappa", -1), reverse=True)
    for i, trial in enumerate(sorted_trials[:3], 1):
        print(f"Top {i} - Trial {trial.number} | Val Kappa: {trial.value:.4f} | Test Kappa: {trial.user_attrs.get('test_kappa'):.4f}")
        print(f"  ↪ Params: {trial.params}")

    print("\n✅ Best Trial Summary:")
    print(f"  Trial #{best_trial.number}")
    print(f"  Val Kappa: {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

def run_robustness_study(base_params, n_trials=50, study_name="CBraMod-robustness-focused"):
    """
    Specialized robustness study focusing on systematic noise injection analysis.
    
    This study runs comprehensive robustness experiments across all noise types
    and levels to generate data for robustness analysis plots.
    """
    
    storage_url = f"sqlite:///experiments/optuna_studies/{study_name}.db"
    
    def robustness_objective(trial):
        """
        Objective function focused on robustness evaluation.
        
        Systematically explores noise injection space while keeping 
        other hyperparameters more constrained.
        """
        # Copy base parameters
        params = copy.deepcopy(base_params)
        
        # === FOCUSED: Noise injection parameters (systematic exploration) ===
        params.noise_level = trial.suggest_categorical("noise_level", [
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25  # Extended range for robustness
        ])
        
        if params.noise_level > 0.0:
            params.noise_type = trial.suggest_categorical("noise_type", [
                "realistic", "emg", "movement", "electrode", "gaussian"
            ])
        else:
            params.noise_type = "realistic"
        
        params.noise_seed = 42  # Fixed for reproducibility
        
        # === CONSTRAINED: Other hyperparameters (focus on noise, not hyperopt) ===
        # Keep architecture relatively fixed for fair robustness comparison
        params.num_of_classes = trial.suggest_categorical("num_of_classes", [4, 5])
        params.label_mapping_version = "v1"  # Use consistent mapping
        
        # Basic architecture choices (constrained)
        params.embedding_dim = trial.suggest_categorical("embedding_dim", [256, 512])
        params.layers = trial.suggest_categorical("layers", [8, 12])
        params.heads = trial.suggest_categorical("heads", [4, 8])
        
        # Learning parameters (focused range)
        params.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        params.batch_size = trial.suggest_categorical("batch_size", [32, 64])
        params.epochs = trial.suggest_int("epochs", 30, 80)  # Shorter for more trials
        
        # Training strategy
        params.two_phase_training = trial.suggest_categorical("two_phase_training", [True, False])
        if params.two_phase_training:
            params.phase1_epochs = trial.suggest_int("phase1_epochs", 2, 5)
            head_mult = trial.suggest_float("head_lr_mult", 1.0, 3.0)
            lr_ratio = trial.suggest_float("lr_ratio", 5, 15)
            params.head_lr = params.lr * head_mult
            params.backbone_lr = params.head_lr / lr_ratio
        
        # Loss function choices
        params.use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
        if params.use_focal_loss:
            params.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        
        params.use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
        
        # Dataset choices (focused on robustness comparison)
        params.datasets = trial.suggest_categorical("datasets", [
            "ORP", "ORP,2023_Open_N"  # Keep dataset simpler for robustness focus
        ])
        params.dataset_names = [name.strip() for name in params.datasets.split(',')]
        params.num_datasets = len(params.dataset_names)
        
        params.data_ORP = trial.suggest_float("data_ORP", 0.5, 0.6, step=0.1)
        params.preprocess = trial.suggest_categorical("preprocess", [True, False])
        params.frozen = False  # Always fine-tune for robustness evaluation
        
        # Fixed configurations for consistency
        params.foundation_dir = "./saved_models/pretrained/pretrained_weights.pth"
        params.use_amp = True  # Enable for efficiency
        
        # Create experiment identifier for robustness tracking
        noise_desc = f"clean" if params.noise_level == 0.0 else f"{params.noise_type}_{int(params.noise_level*100)}pct"
        params.run_name = f"robust_{noise_desc}_t{trial.number}"
        
        # Use the existing objective function
        return objective(trial, params)
    
    # Create study directory
    os.makedirs("experiments/optuna_studies", exist_ok=True)
    
    # Create and run study
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name, 
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=0)
    )
    
    print(f"🎯 Starting ROBUSTNESS-FOCUSED study: {study_name}")
    print(f"   Target: {n_trials} trials focused on noise injection analysis")
    print(f"   Storage: {storage_url}")
    print(f"   Noise levels: 0%, 5%, 10%, 15%, 20%, 25%")
    print(f"   Noise types: realistic, emg, movement, electrode, gaussian")
    
    study.optimize(robustness_objective, n_trials=n_trials)
    
    # Print robustness analysis summary
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        print(f"\n🔊 ROBUSTNESS ANALYSIS SUMMARY ({len(completed_trials)} completed trials)")
        print("=" * 70)
        
        # Group by noise characteristics
        noise_groups = {}
        for trial in completed_trials:
            noise_level = trial.params.get('noise_level', 0.0)
            noise_type = trial.params.get('noise_type', 'clean')
            key = f"{noise_type}_{noise_level:.2f}"
            
            if key not in noise_groups:
                noise_groups[key] = []
            noise_groups[key].append(trial.value)
        
        # Print summary by noise type
        for noise_key in sorted(noise_groups.keys()):
            values = noise_groups[noise_key]
            avg_performance = np.mean(values)
            std_performance = np.std(values)
            print(f"  {noise_key:>20}: {avg_performance:.4f} ± {std_performance:.4f} (n={len(values)})")
        
        print(f"\n💾 Results saved to: {storage_url}")
        print("   Use Plot_Clean/load_and_structure_runs.py to extract results for plotting")
    
    return study

if __name__ == "__main__":
    from finetune_main import main
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_eval', action='store_true', help='Enable multi-subject evaluation for high-performing trials')
    parser.add_argument('--multi_eval_subjects', nargs='+', type=str, default=[], help='List of subjects for multi-subject evaluation')
    parser.add_argument('--robustness_study', action='store_true', help='Run robustness-focused study instead of general tuning')
    parser.add_argument('--robustness_trials', type=int, default=50, help='Number of trials for robustness study')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials for general tuning')
    args, _ = parser.parse_known_args()

    params = main(return_params=True)
    
    if args.robustness_study:
        print("🎯 Running ROBUSTNESS-FOCUSED study with noise injection")
        study_name = f"CBraMod-robustness-focused-{params.num_of_classes}class"
        run_robustness_study(params, n_trials=args.robustness_trials, study_name=study_name)
    else:
        print("🔍 Running GENERAL hyperparameter tuning study") 
        run_optuna_tuning(params, multi_eval=args.multi_eval, multi_eval_subjects=args.multi_eval_subjects, n_trials=args.n_trials)
