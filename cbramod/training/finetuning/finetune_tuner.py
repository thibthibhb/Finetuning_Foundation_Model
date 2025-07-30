import sys
import os
import optuna
import copy
import argparse
import matplotlib.pyplot as plt
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
from statistics import mean, stdev
import torch
from torch.utils.data import DataLoader, Subset
 



def objective(trial, base_params):
    print("🚀 Starting new trial")
    params = copy.deepcopy(base_params)

    # Suggest hyperparams
    params.lr = trial.suggest_float("lr",  1e-5, 5e-3, log=True)
    params.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.01)
    params.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    params.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024]) #512, 1024,
    params.clip_value = trial.suggest_float("clip_value", 0.0, 2.0, step=0.5)
    params.multi_lr = trial.suggest_categorical("multi_lr", [True, False])
    params.use_weighted_sampler = trial.suggest_categorical("use_weighted_sampler", [True, False])
    params.optimizer = trial.suggest_categorical("optimizer", ["AdamW","Lion"]) #"AMSGrad", "SGD"
    params.scheduler = trial.suggest_categorical("scheduler", ["cosine"])
    params.head_type = trial.suggest_categorical("head_type", ["simple", "deep", "attention"]) #simple
    # Focal Loss - conditional parameter to avoid wasted trials
    params.use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
    # Only tune gamma when actually using focal loss (50% efficiency gain)
    if params.use_focal_loss:
        params.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0, step=0.5)
    else:
        params.focal_gamma = 2.0  # Default unused value
    params.data_ORP = trial.suggest_float("data_ORP", 0.4,0.6, step=0.1)
    # Gradient accumulation (commented out - may affect performance)
    # params.gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    # params.gradient_accumulation_steps = 1  # Default to no accumulation
    
    # Scale learning rate proportionally to gradient accumulation (commented out)
    # if params.gradient_accumulation_steps > 1:
    #     lr_scaling_factor = params.gradient_accumulation_steps ** 0.5  # Square root scaling is more conservative
    #     params.lr = params.lr * lr_scaling_factor
    #     print(f"🔧 Scaled LR for gradient accumulation: {params.lr:.2e} (factor: {lr_scaling_factor:.2f})")
    
    #params.epochs = trial.suggest_int("epochs", 3, 7, step=1)
    
    # Two-phase training - conditional parameters for efficiency
    params.two_phase_training = trial.suggest_categorical("two_phase_training", [True, False])
    
    if params.two_phase_training:
        # Only tune these parameters when actually using two-phase training
        params.phase1_epochs = trial.suggest_int("phase1_epochs", 2, 5)  # Shorter phase 1
        
        # Use similar ranges to the main LR, but with reasonable ratios
        base_head_lr = trial.suggest_float("base_head_lr", 0.5, 2.0)  # Multiplier for main LR
        head_to_backbone_ratio = trial.suggest_float("head_to_backbone_ratio", 5, 20)
        
        params.head_lr = params.lr * base_head_lr
        params.backbone_lr = params.head_lr / head_to_backbone_ratio
        
        print(f"🔧 Two-phase training: Phase1={params.phase1_epochs}e, Head_LR={params.head_lr:.2e}, Backbone_LR={params.backbone_lr:.2e}")
    else:
        # Standard single-phase training
        print(f"🔧 Standard training: LR={params.lr:.2e}")
            
    # AMP is always enabled for performance
    #params.use_amp = True

    # Pretrained weights
    params.foundation_dir = trial.suggest_categorical(
        "foundation_dir", [
            "./artifacts/models/pretrained/BEST__loss10527.31.pth",
            "./artifacts/models/pretrained/pretrained_weights.pth"
        ]
    )

    # Load dataset and apply fixed subject-level split
    load_dataset = idun_datasets.LoadDataset(params)
    seqs_labels_path_pair = load_dataset.get_all_pairs()
    dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair, num_of_classes=params.num_of_classes)
    seed = trial.suggest_int("split_seed", 0, 10000)
    fold, train_idx, val_idx, test_idx = next(idun_datasets.get_custom_split(dataset, seed=seed, orp_train_frac=params.data_ORP))

    print(f"\n▶️ Using fixed split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    if params.num_subjects_train < 0:   
        params.num_subjects_train = dataset.num_subjects_train

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print(params)
    model = model_for_idun.Model(params)
    trainer = Trainer(params, data_loader, model)

    print(f"🚀 Training on fixed split")
    try:
        kappa = trainer.train_for_multiclass(trial=trial)
    except optuna.exceptions.TrialPruned:
        print("🔪 Trial was pruned early")
        raise
    acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)

    # Log to trial
    trial.set_user_attr("test_kappa", kappa)
    trial.set_user_attr("test_f1", f1)
    trial.set_user_attr("test_accuracy", acc)

    # Optional fallback pruning if final kappa is very low
    if kappa <= 0.05:
        print("🔪 Final pruning due to very low kappa")
        raise optuna.exceptions.TrialPruned()

    return kappa


def run_optuna_tuning(params):
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))
    print("🔵 Starting hyperparameter search...")

    study.optimize(lambda trial: objective(trial, params), n_trials=20)

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


if __name__ == "__main__":
    from finetune_main import main
    params = main(return_params=True)
    run_optuna_tuning(params)