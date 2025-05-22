import optuna
import copy
import matplotlib.pyplot as plt

from finetune_main import main
from datasets import orp_dataset, idun_takeda_dataset
from models import model_for_takeda_idun
from finetune_trainer import Trainer
import wandb

def objective(trial, base_params):
    print("🚀 Starting new trial")
    params = copy.deepcopy(base_params)

    # Suggest hyperparams
    params.lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    params.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.01)
    params.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
    params.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    params.clip_value = trial.suggest_float("clip_value", 0.1, 2.0)
    params.multi_lr = trial.suggest_categorical("multi_lr", [True, False])
    params.use_weighted_sampler = trial.suggest_categorical("use_weighted_sampler", [True, False])

    # NEW: Pretrained weights selection
    pretrained_weight_path = trial.suggest_categorical(
        "foundation_dir", [
            #"pretrained_weights/new_weights_unlabelled_batch128.pth",
            #"pretrained_weights/new_weights_unlabelled_batch64.pth",
            "pretrained_weights/new_weights_unlabelled_full_data+hyperparam.pth",
            #"pretrained_weights/pretrained_weights.pth"
        ]
    )
    params.foundation_dir = pretrained_weight_path

    # Log selected hyperparams to Optuna for later inspection
    trial.set_user_attr("lr", params.lr)
    trial.set_user_attr("weight_decay", params.weight_decay)
    trial.set_user_attr("label_smoothing", params.label_smoothing)
    trial.set_user_attr("dropout", params.dropout)
    trial.set_user_attr("batch_size", params.batch_size)
    trial.set_user_attr("clip_value", params.clip_value)
    trial.set_user_attr("multi_lr", params.multi_lr)
    trial.set_user_attr("foundation_dir", params.foundation_dir)

        # 👇 Start a W&B run
    with wandb.init(
        project="CBraMod-earEEG-tuning",
        config=params.__dict__,
        reinit=True,
        name=params.run_name):
            # Load dataset & model
            load_dataset = idun_takeda_dataset.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
            model = model_for_takeda_idun.Model(params)

            # Train
            trainer = Trainer(params, data_loader, model)
            kappa_best = trainer.train_for_multiclass()

            # Evaluate
            trainer.model.load_state_dict(trainer.best_model_states)
            acc_test, kappa_test, f1_test, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(trainer.model)

            # Attach to trial
            trial.set_user_attr("kappa_test", kappa_test)
            trial.set_user_attr("f1_test", f1_test)
            trial.set_user_attr("acc_test", acc_test)

            # Log metrics to wandb
            wandb.log({
                "val_kappa": kappa_best,
                "test_kappa": kappa_test,
                "test_accuracy": acc_test,
                "test_f1": f1_test,
                # Extended metadata
                "hours_of_data": trainer.hours_of_data,
                "inference_latency_ms": getattr(trainer.model, 'inference_latency_ms', None),
                "inference_throughput_samples_per_sec": getattr(trainer.model, 'throughput', None),
                "num_subjects": getattr(params, 'num_subjects', None),
                "est_cost_per_night_usd": trainer.est_cost_per_night_usd
            })

            # Pruning
            trial.report(kappa_best, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return kappa_best

def run_optuna_tuning(base_params):
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    print("🔵 Starting hyperparameter search...")

    study.optimize(lambda trial: objective(trial, base_params),  n_trials=2, timeout=144000)

    print("🔵 Hyperparameter search completed.")
    best_trial = study.best_trial

    # Show top trials
    print("=== Top Trials by Test Kappa ===")
    sorted_trials = sorted(study.trials, key=lambda t: t.user_attrs.get("kappa_test", -1), reverse=True)
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"Top {i} - Trial {trial.number} | Val Kappa: {trial.value:.4f} | Test Kappa: {trial.user_attrs.get('kappa_test'):.4f}")
        print(f"  ↪ Params: {trial.params}")

    print("\n" + "="*60)
    print(f"✅ Best Trial Number: {best_trial.number}")
    print(f"✅ Best Validation Kappa: {best_trial.value:.4f}")
    print("✅ Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")
    print("="*60 + "\n")

    # 🎯 Retrain best config from scratch
    print("🎯 Re-training best config on full training set...")

    # Step 1: Create new params
    final_params = copy.deepcopy(base_params)
    for k, v in best_trial.params.items():
        setattr(final_params, k, v)

    # Step 2: Setup new dataset, model, trainer
    load_dataset = idun_takeda_dataset.LoadDataset(final_params)
    data_loader = load_dataset.get_data_loader()
    model = model_for_takeda_idun.Model(final_params)
    trainer = Trainer(final_params, data_loader, model)

    # Step 3: Train and evaluate
    final_kappa = trainer.train_for_multiclass()
    trainer.model.load_state_dict(trainer.best_model_states)
    acc_test, kappa_test, f1_test, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(trainer.model)

    # Step 4: Log final results
    print("🏁 Final Test Evaluation After Retraining:")
    print(f"  - Test Accuracy: {acc_test:.4f}")
    print(f"  - Test Kappa:    {kappa_test:.4f}")
    print(f"  - Test F1:       {f1_test:.4f}")

    # Step 5: Log to W&B
    wandb.init(project="CBraMod-earEEG-tuning", name="final_best_model", job_type="retrain", config=final_params.__dict__)
    wandb.log({
        "final_test_accuracy": acc_test,
        "final_test_kappa": kappa_test,
        "final_test_f1": f1_test,
        # Extended metadata
        "hours_of_data": trainer.hours_of_data,
        "inference_latency_ms": getattr(trainer.model, 'inference_latency_ms', None),
        "inference_throughput_samples_per_sec": getattr(trainer.model, 'throughput', None),
        "num_subjects": getattr(final_params, 'num_subjects', None),
        "est_cost_per_night_usd": trainer.est_cost_per_night_usd
    })
    wandb.finish()


if __name__ == "__main__":
    # ❗ Get params correctly
    params = main(return_params=True)
    run_optuna_tuning(params)
