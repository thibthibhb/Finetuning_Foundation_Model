import sys
import argparse
import random

import numpy as np
import torch
import os
# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)
from cbramod.load_datasets import idun_datasets
try:
    from .finetune_trainer import Trainer
except ImportError:
    from finetune_trainer import Trainer
from cbramod.models import model_for_idun
import pdb
from statistics import mean, stdev
from torch.utils.data import DataLoader
import torch
try:
    from .finetune_tuner import run_optuna_tuning
except ImportError:
    from finetune_tuner import run_optuna_tuning

# 
def main(return_params=False):
    parser = argparse.ArgumentParser(description='Big model downstream')
    
    # Foundation model architecture info (for scaling plots)
    parser.add_argument('--model_name', type=str, default='EEG-Foundation-v1')
    parser.add_argument('--model_size', type=int, default=4000000)  # total params
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=512)

    # Data-related info
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of training data used')
    parser.add_argument('--num_subjects_train', type=int, default=-1, help='(auto-filled if -1) Number of unique subjects in the dataset')
    #parser.add_argument('--num_nights_train', type=int, default=-1, help='(auto-filled if -1) Number of unique subjects in the dataset')

    # Experiment tracking
    parser.add_argument('--baseline', action='store_true', help='Is this a non-foundation baseline model?')
    parser.add_argument('--comment', type=str, default='', help='Optional note for logging')

    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (default: 1e-2)') #5e-2
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout') # 0.1
    parser.add_argument('--sample_rate', type=float, default=200, help='sample_rate') # 200
    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/Faced/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='artifacts/models/pretrained/', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=False,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        # default='/data/wjq/models_weights/Big/0.4/epoch40_loss0.001386052928864956.pth',
                        default='./artifacts/models/pretrained/pretrained_weights.pth',
                        help='foundation_dir')
    parser.add_argument("--run_name", type=str, default="test", help="WandB run name prefix")
    parser.add_argument('--use_weighted_sampler', type=bool, default=False,
                        help='Use weighted sampler to balance class distributions')
    # ADD BY THIBAUT
    parser.add_argument('--weight_class', type=str, default=None,
                    help='Path to the .npy file of class weights (optional)')
    
    # Two-phase training parameters
    parser.add_argument('--two_phase_training', type=bool, default=False,
                        help='Enable two-phase training: phase 1 (frozen backbone) -> phase 2 (unfrozen)')
    parser.add_argument('--phase1_epochs', type=int, default=3,
                        help='Number of epochs for phase 1 (frozen backbone)')
    parser.add_argument('--head_lr', type=float, default=1e-3,
                        help='Learning rate for head/classifier in two-phase training')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                        help='Learning rate for backbone in phase 2 of two-phase training')
    # Gradient accumulation (commented out - may affect performance)
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help='Number of steps to accumulate gradients before updating')
    parser.add_argument('--tune', action='store_true', help="Use Optuna to tune hyperparameters")
    parser.add_argument('--datasets', type=str, default='ORP', help='Comma-separated dataset names, e.g., ORP,2023_Open_N')
    parser.add_argument('--scheduler', type=str, default='cosine', help='["cosine", "step", "none"]')
    parser.add_argument('--head_type', type=str, default='simple', 
                        help='Classification head architecture: ["simple", "deep", "attention"]')
    parser.add_argument('--use_focal_loss', action='store_true', 
                        help='Use Focal Loss instead of CrossEntropy for imbalanced EEG data (improves N1/REM recall)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (higher = more focus on hard examples)')
    parser.add_argument('--data_ORP', type=float, default=0.6,
                    help='Fraction of ORP data to use in training set (e.g., 0.1, 0.3, 0.5, 0.6)')
    # Add plot directory
    parser.add_argument('--plot_dir', type=str,
                        default='./artifacts/results/figures',
                        help='Directory for plots and figures')    
    # Performance optimization arguments
    parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Use Automatic Mixed Precision training for faster training and reduced memory usage')
    parser.add_argument('--results_dir', type=str, default='./artifacts/results', help='results directory')
    params = parser.parse_args()
    # Automatically compute number of datasets
    params.dataset_names = [name.strip() for name in params.datasets.split(',')]
    params.num_datasets = len(params.dataset_names)

    # ✅ Load class weights if provided
    if params.weight_class is not None and os.path.exists(params.weight_class + ".npy"):
        weights_array = np.load(params.weight_class + ".npy")
        params.class_weights = torch.tensor(weights_array, dtype=torch.float32).cuda()
    else:
        params.class_weights = None
    # Set gradient accumulation to 1 (no accumulation) for simplicity
    # params.gradient_accumulation_steps = 1
    
    
    # Scale learning rate for gradient accumulation (commented out)
    # if params.gradient_accumulation_steps > 1:
    #     lr_scaling_factor = params.gradient_accumulation_steps ** 0.5
    #     params.lr = params.lr * lr_scaling_factor
    #     print(f"🔧 Scaled LR for gradient accumulation: {params.lr:.2e} (factor: {lr_scaling_factor:.2f})")
    #     
    #     # Also scale two-phase learning rates if enabled
    #     if params.two_phase_training:
    #         params.head_lr = params.head_lr * lr_scaling_factor
    #         params.backbone_lr = params.backbone_lr * lr_scaling_factor
    #         print(f"🔧 Scaled two-phase LRs: Head={params.head_lr:.2e}, Backbone={params.backbone_lr:.2e}")


    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    
    # ✨ NEW: move tuning here
    if params.tune:
        run_optuna_tuning(params)
        print("Tuning completed. Exiting.")
        return
    
    print('The downstream dataset is {}'.format(params.downstream_dataset))

    if params.downstream_dataset == 'IDUN_EEG':
        load_dataset = idun_datasets.LoadDataset(params)
        seqs_labels_path_pair = load_dataset.get_all_pairs()

        # Load once!
        dataset = idun_datasets.MemoryEfficientKFoldDataset(seqs_labels_path_pair, num_of_classes=params.num_of_classes)

        # Use single subject-level split
        fold, train_idx, val_idx, test_idx = next(idun_datasets.get_custom_split(dataset, seed=42, orp_train_frac=params.data_ORP))

        print(f"\n▶️ Using split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        if params.num_subjects_train < 0:      # leave CLI override untouched
            params.num_subjects_train = dataset.num_subjects_train
        # if params.num_nights_train < 0:
        #     params.num_nights_train   = dataset.num_nights_train
            
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        val_loader = DataLoader(val_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
        test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

        data_loader = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
        }

        model = model_for_idun.Model(params)
        trainer = Trainer(params, data_loader, model)

        print(f"🚀 Training model on fixed split")
        kappa = trainer.train_for_multiclass()
        acc, _, f1, _, _, _ = trainer.test_eval.get_metrics_for_multiclass(model)

        print("\n📊 Evaluation Results:")
        print(f"  Kappa: {kappa:.4f}")
        print(f"  Acc:   {acc:.4f}")
        print(f"  F1:    {f1:.4f}")

        return kappa, acc, f1
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
