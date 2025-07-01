import optuna
import copy
import argparse
import matplotlib.pyplot as plt
from datasets import idun_datasets
from models import model_for_idun
from finetune_trainer import Trainer
from statistics import mean, stdev
import torch
from torch.utils.data import DataLoader, Subset


def objective(trial, base_params):
    print("🚀 Starting new trial")
    params = copy.deepcopy(base_params)

    # Suggest hyperparams
    params.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    params.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.01)
    params.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    params.batch_size = trial.suggest_categorical("batch_size", [1024]) #512, 1024,
    params.clip_value = trial.suggest_float("clip_value", 0.1, 2.0)
    params.multi_lr = trial.suggest_categorical("multi_lr", [True, False])
    params.use_weighted_sampler = trial.suggest_categorical("use_weighted_sampler", [True, False])
    params.optimizer = trial.suggest_categorical("optimizer", ["AdamW","Lion"])
    params.scheduler = trial.suggest_categorical("scheduler", ["cosine"])
    #params.data_ORP = trial.suggest_float("data_ORP", 0.1,0.2, step=0.1)
    #params.noise = trial.suggest_categorical("noise", [0.0021196, 0.003154, 0.0034662, 0.068127, 0.083107, 0.12952, 0.21352, 0.29603]) 

    # Pretrained weights
    params.foundation_dir = trial.suggest_categorical(
        "foundation_dir", [
            #"optuna_ckpts/BEST__loss8698.99.pth",
            "optuna_ckpts/BEST__loss10527.31.pth",
            # "optuna_ckpts/BEST__loss11060.72.pth",
            "pretrained_weights/pretrained_weights.pth"
            ]
    )

    # Load dataset and apply fixed subject-level split
    load_dataset = idun_datasets.LoadDataset(params)
    seqs_labels_path_pair = load_dataset.get_all_pairs()
    dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair)
    seed = trial.suggest_int("split_seed", 0, 10000)
    fold, train_idx, val_idx, test_idx = next(idun_datasets.get_custom_split(dataset, seed=seed, orp_train_frac=params.data_ORP))

    print(f"\n▶️ Using fixed split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    if params.num_subjects_train < 0:    # leave CLI override untouched
        params.num_subjects_train = dataset.num_subjects_train
    # if params.num_nights_train < 0:
    #     params.num_nights_train   = dataset.num_nights_train
        
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
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
    if kappa <= 0.03:
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
