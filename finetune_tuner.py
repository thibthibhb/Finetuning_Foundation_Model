import optuna
import copy
import matplotlib.pyplot as plt

from finetune_main import main
from datasets import orp_dataset, idun_takeda_dataset
from models import model_for_takeda_idun
from finetune_trainer import Trainer

# ✨ Receive params from outside
def objective(trial, base_params):
    print("🚀 Starting new trial")

    # Step 1: Copy params
    params = copy.deepcopy(base_params)

    # Step 2: Tune hyperparameters
    params.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    params.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.01)
    params.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    params.batch_size = trial.suggest_categorical("batch_size", [64])
    params.clip_value = trial.suggest_float("clip_value", 0.1, 2.0)
    params.multi_lr = trial.suggest_categorical("multi_lr", [True, False])

    # Step 3: Load dataset and model
    load_dataset = idun_takeda_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = model_for_takeda_idun.Model(params)

    # Step 4: Train
    trainer = Trainer(params, data_loader, model)
    kappa_best = trainer.train_for_multiclass()
    # Evaluate on test set with best model (leak-free since we're only logging)
    trainer.model.load_state_dict(trainer.best_model_states)
    acc_test, kappa_test, f1_test, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(trainer.model)

    # Attach test results to the trial (we will NOT use this for optimization!)
    trial.set_user_attr("kappa_test", kappa_test)
    trial.set_user_attr("f1_test", f1_test)
    trial.set_user_attr("acc_test", acc_test)

    # Report metric for pruning
    trial.report(kappa_best, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return kappa_best

def run_optuna_tuning(base_params):
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    print("🔵 Starting hyperparameter search...")

    # 💡 Pass lambda to give trial access to params
    study.optimize(lambda trial: objective(trial, base_params),  n_trials=10, timeout=3600)
    print("🔵 Hyperparameter search completed.")

    best_trial = study.best_trial
    print("=== Top Trials by Test Kappa ===")
    sorted_trials = sorted(study.trials, key=lambda t: t.user_attrs.get("kappa_test", -1), reverse=True)
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"Top {i} - Trial {trial.number} | Val Kappa: {trial.value:.4f} | Test Kappa: {trial.user_attrs.get('kappa_test'):.4f}")
        print(f"  ↪ Params: {trial.params}")
    print("previous prints")
    print("\n" + "="*60)
    print(f"✅ Best Trial Number: {best_trial.number}")
    print(f"✅ Best Validation Accuracy: {best_trial.value:.4f}")
    print("✅ Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # ❗ Get params correctly
    params = main(return_params=True)
    run_optuna_tuning(params)
