import copy
from finetune_tuner import run_optuna_tuning
from finetune_main import main
from datasets import idun_datasets

TRAIN_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5]   # 10% to 50%

for train_frac in TRAIN_FRACS:
        params = main(return_params=True)

        # === Customize params ===
        params.orp_train_frac = train_frac
        params.run_name = f"ORP_frac{int(train_frac*100)}"
        params.use_wandb = True
        params.dataset_names = ["ORP"]

        print(f"\n🚀 Training with {int(train_frac*100)}% of ORP data")
        run_optuna_tuning(params)
