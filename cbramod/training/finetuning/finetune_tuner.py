import sys
import os
import optuna
import copy
import argparse
import json
from statistics import mean, stdev
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
            sfreq=getattr(config, 'sample_rate', 200.0)
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
    # Process datasets after Optuna selection (this should happen here, not in main)
    params.dataset_names = [name.strip() for name in params.datasets.split(',')]
    params.num_datasets = len(params.dataset_names)
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

    params.preprocess = trial.suggest_categorical("preprocess", [True, False])


    params.frozen = trial.suggest_categorical("frozen", [True, False])
    
    load_dataset = idun_datasets.LoadDataset(params)
    seqs_labels_path_pair = load_dataset.get_all_pairs()
    dataset = idun_datasets.MemoryEfficientKFoldDataset(
        seqs_labels_path_pair,
        num_of_classes=params.num_of_classes,
        label_mapping_version=getattr(params, 'label_mapping_version', 'v1'),
        do_preprocess=getattr(params, 'preprocess', False),
        sfreq=getattr(params, 'sample_rate', 200.0)
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


if __name__ == "__main__":
    from finetune_main import main
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_eval', action='store_true', help='Enable multi-subject evaluation for high-performing trials')
    parser.add_argument('--multi_eval_subjects', nargs='+', type=str, default=[], help='List of subjects for multi-subject evaluation')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials for general tuning')
    args, _ = parser.parse_known_args()

    params = main(return_params=True)
    
    print("🔍 Running GENERAL hyperparameter tuning study")
    run_optuna_tuning(params, multi_eval=args.multi_eval, multi_eval_subjects=args.multi_eval_subjects, n_trials=args.n_trials)
