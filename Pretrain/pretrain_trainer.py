import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm
import os
from utils.util import generate_mask
import optuna
from torch.utils.data import DataLoader
from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        summary(self.model, input_size=(1, 19, 30, 200))

        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )

    def train(self):
        self.train_loss_history = []
        self.best_loss = 10000000
        self.best_epoch = -1
        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100
                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    y = self.model(x, mask=mask)
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    loss = self.criterion(masked_y, masked_x)

                    # non_masked_x = x[mask == 0]
                    # non_masked_y = y[mask == 0]
                    # non_masked_loss = self.criterion(non_masked_y, non_masked_x)
                    # loss = 0.8 * masked_loss + 0.2 * non_masked_loss
                else:
                    y = self.model(x)
                    loss = self.criterion(y, x)
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
            mean_loss = np.mean(losses)
            self.train_loss_history.append(mean_loss)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            # ✅ track best epoch
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.best_epoch = epoch + 1
        print(f"\n🟢 Best Training Loss: {self.best_loss:.4f} at Epoch {self.best_epoch}")


def objective(trial):
    # 🔧 Sample hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.6)
    clip_value = trial.suggest_float("clip_value", 0.0, 1.0)
    lr_scheduler = trial.suggest_categorical("lr_scheduler", [
        "CosineAnnealingLR", "ExponentialLR", "StepLR", "MultiStepLR", "CyclicLR"
    ])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    epochs = trial.suggest_int("epochs", 10, 30)

    # ⚙️ Params object
    class Params:
        def __init__(self):
            self.lr = lr
            self.weight_decay = weight_decay
            self.mask_ratio = mask_ratio
            self.clip_value = clip_value
            self.lr_scheduler = lr_scheduler
            self.cuda = 0
            self.parallel = False
            self.epochs = epochs
            self.need_mask = True
            self.model_dir = "./optuna_ckpts"
            self.batch_size = batch_size
            self.in_dim = 200
            self.out_dim = 200
            self.d_model = 200
            self.dim_feedforward = 800
            self.seq_len = 30
            self.n_layer = 12
            self.nhead = 8
            self.dropout = dropout
            self.dataset_dir = "Unlabelled/Sleep"

    params = Params()

    # 📦 Load dataset and model
    dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, num_workers=8, shuffle=True)
    model = CBraMod(params.in_dim, params.out_dim, params.d_model, params.dim_feedforward,
                    params.seq_len, params.n_layer, params.nhead)

    # 🚀 Train
    trainer = Trainer(params, dataloader, model)
    trainer.train()

    # 🧠 Store trial metadata
    trial.set_user_attr("best_loss", trainer.best_loss)
    trial.set_user_attr("best_epoch", trainer.best_epoch)

    # 🌍 Save global best model only
    if not hasattr(objective, "global_best_loss"):
        objective.global_best_loss = float("inf")

    if trainer.best_loss < objective.global_best_loss:
        objective.global_best_loss = trainer.best_loss
        best_model_path = os.path.join(
            params.model_dir, f"BEST__loss{trainer.best_loss:.2f}.pth"
        )
        torch.save(model.state_dict(), best_model_path)
        trial.set_user_attr("model_path", best_model_path)
        print(f"🌟 Saved new best model: {best_model_path}")

    return trainer.best_loss