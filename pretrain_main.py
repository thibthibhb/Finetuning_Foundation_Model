# pretrain_main.py
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod
from pretrain_trainer import Trainer
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def freeze_early_layers(model, num_layers_to_freeze):
    """
    Freezes the first `num_layers_to_freeze` transformer encoder layers in CBraMod.
    """
    for idx, layer in enumerate(model.encoder.layers):
        if idx < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--in_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=200)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--need_mask', type=bool, default=True)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--dataset_dir', type=str, default='dataset_dir')
    parser.add_argument('--model_dir', type=str, default='model_dir')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_weights/pretrained_weights.pth')
    parser.add_argument('--save_path', type=str, default='pretrained_weights/new_weights_unlabelled_batch128.pth')
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Number of early transformer layers to freeze (0 to train all)')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use WeightedRandomSampler to balance training batches')

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)

    # ✅ Ensure model_dir exists before trying to save to it
    os.makedirs(params.model_dir, exist_ok=True)

    dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, num_workers=8, shuffle=True)

    model = CBraMod(params.in_dim, params.out_dim, params.d_model, params.dim_feedforward,
                    params.seq_len, params.n_layer, params.nhead)


    if os.path.exists(params.pretrained_path):
        print(f"Loading pretrained weights from {params.pretrained_path}")
        model.load_state_dict(torch.load(params.pretrained_path, map_location='cpu'))

        # Apply layer freezing from argument
        if params.freeze_layers > 0:
            print(f"Freezing the first {params.freeze_layers} transformer layers...")
            freeze_early_layers(model, num_layers_to_freeze=params.freeze_layers)


    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")


    trainer = Trainer(params, dataloader, model)
    trainer.train()

    torch.save(model.state_dict(), params.save_path)
    print(f"Final model saved at {params.save_path}")

if __name__ == '__main__':
    main()
