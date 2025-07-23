import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from .finetune_evaluator import Evaluator
except ImportError:
    from finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
import umap
from sklearn.decomposition import PCA
import copy
import os
import pandas as pd
import seaborn as sns
import csv
import time
import subprocess
import json
from datetime import timedelta
from torchinfo import summary
from sklearn.metrics import classification_report
from collections import Counter
from csv import QUOTE_MINIMAL
import wandb
import optuna
import gc
import logging
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 1.10
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 1.10
# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Import memory management utilities
try:
    from cbramod.utils.memory_manager import MemoryManager, TrainingMemoryTracker
except ImportError:
    # Fallback if memory manager is not available
    class MemoryManager:
        def __init__(self, *args, **kwargs): pass
        def cleanup_between_trials(self, *args, **kwargs): pass
        def manage_checkpoints(self, *args, **kwargs): return True
        def start_memory_monitoring(self): pass
        def get_checkpoint_summary(self): return {}
    
    class TrainingMemoryTracker:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args, **kwargs): pass
class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        # Move model to appropriate device
        device = torch.device(f"cuda:{params.cuda}" if params.cuda >= 0 and torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.device = device
        
        if getattr(self.params, 'downstream_dataset', 'IDUN_EEG') in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'IDUN_EEG']:
            self.criterion = CrossEntropyLoss(label_smoothing=getattr(self.params, 'label_smoothing', 0.0)).to(device)
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().to(device)
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().to(device)

        self.best_model_states = None
        self.best_val_cm = None
        
        # Initialize mixed precision training
        self.use_amp = getattr(params, 'use_amp', True)
        if self.use_amp:
            try:
                self.scaler = GradScaler('cuda')  # New API
            except TypeError:
                self.scaler = GradScaler()  # Fallback for older PyTorch
        else:
            self.scaler = None
        self.gradient_accumulation_steps = getattr(params, 'gradient_accumulation_steps', 1)
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            checkpoint_dir=getattr(params, 'model_dir', './artifacts/models/finetuned'),
            max_checkpoints=getattr(params, 'max_checkpoints', 5),
            cleanup_older_than_days=getattr(params, 'cleanup_older_than_days', 7),
            memory_threshold_mb=getattr(params, 'memory_threshold_mb', 1000.0)
        )
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    for p in self.model.backbone.parameters():
                        p.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)


        def _get_lion_cls():
            if hasattr(torch.optim, "Lion"):            # PyTorch ≥ 2.3
                return torch.optim.Lion
            try:                                        # community wheel
                from lion_pytorch import Lion as LionCls
                return LionCls
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Requested optimizer 'Lion' but neither torch.optim.Lion "
                    "nor the 'lion-pytorch' package is available."
                ) from exc

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)

        elif self.params.optimizer == "Lion":                       # ⬅️ NEW
            LionCls = _get_lion_cls()
            if self.params.multi_lr:
                self.optimizer = LionCls(
                    [
                        {"params": backbone_params, "lr": self.params.lr},
                        {"params": other_params,  "lr": self.params.lr * 5},
                    ],
                    lr=self.params.lr,
                    weight_decay=self.params.weight_decay,
                    betas=(0.9, 0.99),              # authors’ defaults
                )
            else:
                self.optimizer = LionCls(
                    self.model.parameters(),
                    lr=self.params.lr,
                    weight_decay=self.params.weight_decay,
                    betas=(0.9, 0.99),
                )

        else:  # fallback / legacy: SGD
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD(
                    [
                        {"params": backbone_params, "lr": self.params.lr},
                        {"params": other_params,  "lr": self.params.lr * 5},
                    ],
                    momentum=0.9,
                    weight_decay=self.params.weight_decay,
                )
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.params.lr, momentum=0.9,
                                weight_decay=self.params.weight_decay,)


        self.data_length = len(self.data_loader['train'])
        
        # Learning rate warmup settings
        self.warmup_epochs = getattr(params, 'warmup_epochs', 3)
        self.total_steps = self.params.epochs * self.data_length // self.gradient_accumulation_steps
        
        # Ensure warmup doesn't exceed total training steps
        max_warmup_steps = max(1, self.total_steps // 4)  # Max 25% of training for warmup
        self.warmup_steps = min(
            self.warmup_epochs * self.data_length // self.gradient_accumulation_steps,
            max_warmup_steps
        )
        
        # Create scheduler with warmup
        if params.scheduler == "cosine":
            import warnings
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(step):
                if step < self.warmup_steps:
                    # Linear warmup - avoid division by zero
                    if self.warmup_steps == 0:
                        return 1.0
                    return step / self.warmup_steps
                else:
                    # Cosine annealing after warmup
                    cosine_steps = self.total_steps - self.warmup_steps
                    if cosine_steps <= 0:
                        return 1.0  # No cosine annealing if no steps left
                    progress = (step - self.warmup_steps) / cosine_steps
                    progress = min(progress, 1.0)  # Clamp to [0, 1]
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            self.optimizer_scheduler = LambdaLR(self.optimizer, lr_lambda)
            
        elif params.scheduler == "step":
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.params.epochs // 3), gamma=0.5)
        elif params.scheduler == "none":
            self.optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        else:
            # Default fallback for any unrecognized scheduler types
            print(f"⚠️ Unknown scheduler '{params.scheduler}', defaulting to cosine with warmup")
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            self.optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.n_params_total = sum(p.numel() for p in self.model.parameters())
        self.n_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.frozen_layers = len([m for m in getattr(self.model, "encoder", self.model).children()
                                   if all(not p.requires_grad for p in m.parameters())])

        self.lora_rank = getattr(self.model, "lora_rank", 0)
        train_dl = self.data_loader['train']
        self.train_steps_per_epoch = len(train_dl)
        n_batches = len(train_dl)
        samples_per_batch = self.params.batch_size
        total_epochs = n_batches * samples_per_batch
        self.total_epochs = total_epochs
        self.hours_of_data = (total_epochs * 30) / 3600
        all_labels = [lab for _, y in train_dl for lab in y.tolist()]
        counts = Counter(all_labels)
        self.class_balance_ratio = {cls: c / len(all_labels) for cls, c in counts.items()}
        self.eval_split = getattr(self.params, "eval_split", "same-night")
        self.gpu_cost_per_hour = getattr(params, "gpu_cost_per_hour", 0.6)
        self.num_gpus = torch.cuda.device_count()

        # Defaults before measurement
        self.model.inference_latency_ms = None
        self.model.throughput = None
        self.throughput = None
        self.cost_per_inference_usd = None
        self.cost_per_night_usd = None
        # Skip inference latency measurement to avoid shape issues
        # self.model.inference_latency_ms = None
        # self.model.throughput = None

        # Use updated model throughput
        self.throughput = self.model.throughput
        samples_per_night = 8 * 3600 // 30  # 8 hours; 960
        if self.throughput and self.throughput > 0:
            self.cost_per_inference_usd = self.gpu_cost_per_hour / (3600 * self.throughput)
        else:
            self.cost_per_inference_usd = None
        
        #self.cost_per_night_usd = self.cost_per_inference_usd * (self.hours_of_data * 3600) / 30
        if self.cost_per_inference_usd is not None:
            self.cost_per_night_usd = self.cost_per_inference_usd * samples_per_night
        else:
            self.cost_per_night_usd = 0.0

        # Log class balance
        self.log_class_distribution(self.data_loader['train'], "train")
        self.log_class_distribution(self.data_loader['val'], "val")
        self.log_class_distribution(self.data_loader['test'], "test")

        # Initialize WandB
        self.wandb_run = None
        if getattr(self.params, 'use_wandb', True):
            self.wandb_run = wandb.init(
                project="CBraMod-earEEG-tuning",
                config=self.params.__dict__,
                reinit=True,
                name=self.params.run_name,
                dir='./artifacts/experiments/wandb'
            )

        #print(self.model)

    def log_class_distribution(self, loader, name="train"):
        labels = [lab for _, y in loader for lab in y.tolist()]
        dist = Counter(labels)
        print(f"{name} class distribution: {dict(dist)}")

    def print_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, digits=4)
        print("Classification Report:\n" + report)

    def early_stopping_check(self, patience_counter, kappa_best, kappa):
        if kappa > kappa_best:
            return 0
        else:
            return patience_counter + 1

    def log_wandb_metrics(self, metrics: dict):
        if self.wandb_run:
            wandb.log(metrics)

    def close_wandb(self):
        if self.wandb_run:
            wandb.finish()


    def train_for_multiclass(self, trial=None):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        best_f1_epoch = 0
        score_history = []
        #log_history = []
        cm_best = None
        patience_counter = 0
        
        # Start memory monitoring for this training session
        trial_name = f"trial_{trial.number}" if trial else "multiclass_training"
        self.memory_manager.start_memory_monitoring()
        
        for epoch in range(self.params.epochs):
            total_train_start = timer()
            self.model.train()
            start_time = timer()
            losses = []
            
            # Reset gradient accumulation
            self.optimizer.zero_grad()

            for batch_idx, (x, y) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast('cuda'):
                        pred = self.model(x)
                        if self.params.downstream_dataset == 'ISRUC':
                            loss = self.criterion(pred.transpose(1, 2), y)
                        else:
                            loss = self.criterion(pred, y)
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                else:
                    pred = self.model(x)
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = self.criterion(pred.transpose(1, 2), y)
                    else:
                        loss = self.criterion(pred, y)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                losses.append(loss.data.cpu().numpy() * self.gradient_accumulation_steps)
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Apply gradient clipping and optimization
                    if self.use_amp:
                        if self.params.clip_value > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.params.clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                        self.optimizer.step()
                    
                    # Step scheduler after each optimizer update (per accumulated gradient step)
                    self.optimizer_scheduler.step()
                    self.optimizer.zero_grad()

            optim_state = self.optimizer.state_dict()

            if epoch % getattr(self.params, "eval_every", 1) == 0:
                with torch.no_grad():
                    acc, kappa, f1, cm, y_true, y_pred = self.val_eval.get_metrics_for_multiclass(self.model)
                    if trial is not None:
                        trial.report(kappa, step=epoch)
                        if kappa <= 0.05:
                            print(f"🔪 Pruning trial at epoch {epoch} due to kappa={kappa:.4f} <= 0.05")
                            raise optuna.exceptions.TrialPruned()

                    val_losses = []
                    for x_val, y_val in self.data_loader['val']:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        
                        if self.use_amp:
                            with autocast('cuda'):
                                pred_val = self.model(x_val)
                                if self.params.downstream_dataset == 'ISRUC':
                                    val_loss = self.criterion(pred_val.transpose(1, 2), y_val)
                                else:
                                    val_loss = self.criterion(pred_val, y_val)
                        else:
                            pred_val = self.model(x_val)
                            if self.params.downstream_dataset == 'ISRUC':
                                val_loss = self.criterion(pred_val.transpose(1, 2), y_val)
                            else:
                                val_loss = self.criterion(pred_val, y_val)

                        val_losses.append(val_loss.item())

                    val_loss_mean = np.mean(val_losses)
                    
                    print("Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.6f}, Time: {:.2f} mins".format(
                        epoch + 1, np.mean(losses), acc, kappa, f1,
                        optim_state['param_groups'][0]['lr'], (timer() - start_time) / 60
                    ))
                    print(cm)

                    if kappa > kappa_best:
                        print("kappa increasing....saving weights !!")
                        print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(acc, kappa, f1))
                        best_f1_epoch = epoch + 1
                        acc_best = acc
                        kappa_best = kappa
                        f1_best = f1
                        cm_best = cm
                        self.best_val_cm = cm
                        self.best_model_states = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter = self.early_stopping_check(patience_counter, kappa_best, kappa)

                    if patience_counter >= getattr(self.params, "early_stop_patience", 5):
                        print("Early stopping triggered.")
                        break

                    # log_history.append({
                    #     'epoch': epoch + 1,
                    #     'train_loss': np.mean(losses),
                    #     'val_loss': val_loss_mean,
                    #     'val_acc': acc,
                    #     'val_kappa': kappa,
                    #     'val_f1': f1
                    # })

                score_history.append((epoch + 1, acc, kappa, f1))
                
                # Periodic memory cleanup every 10 epochs
                if (epoch + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        if self.best_model_states is None:
            print("⚠️ Warning: No improvement in validation. Saving current model as best.")
            self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)

        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm, y_true, y_pred = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print("Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(acc, kappa, f1))
            print(cm)
            
            # === Sleep Recording Detection Logic ===
            # Define sleep stages based on number of classes
            if self.params.num_of_classes == 5:
                sleep_stages = [1, 2, 3, 4]  # Light, Deep, REM (excluding Wake=0, Movement=1)
            elif self.params.num_of_classes == 4:
                sleep_stages = [1, 2, 3]  # Light, Deep, REM (excluding merged Wake=0)
            else:
                raise ValueError(f"Unsupported num_of_classes: {self.params.num_of_classes}")
            y_pred_np = np.array(y_pred)
            total_preds = len(y_pred_np)
            sleep_preds = np.isin(y_pred_np, sleep_stages).sum()
            sleep_ratio = sleep_preds / total_preds

            is_sleep_recording = sleep_ratio > 0.70
            print(f"🧠 Sleep Stage Prediction Ratio: {sleep_ratio:.2%}")
            print("🛌 This is a sleep recording." if is_sleep_recording else "⚠️ This is NOT a sleep recording.")

            self.log_wandb_metrics({
                "val_kappa": kappa_best,
                "test_kappa": kappa,
                "test_accuracy": acc,
                "test_f1": f1,
                "scheduler": type(self.optimizer_scheduler).__name__,
                "hours_of_data": self.hours_of_data,
                "inference_latency_ms": getattr(self.model, 'inference_latency_ms', None),
                "inference_throughput_samples_per_sec": getattr(self.model, 'throughput', None),
                "num_subjects_train": getattr(self.params, 'num_subjects_train', None),
                "cost_per_inference_usd": self.cost_per_inference_usd, 
                "cost_per_night_usd": self.cost_per_night_usd,
                "num_datasets": getattr(self.params, 'num_datasets', None),
                "dataset_names": ','.join(getattr(self.params, 'dataset_names', [])),
                "epochs": self.params.epochs,
                "frac_data_ORP_train": getattr(self.params, 'orp_train_frac', None),
                "use_amp": self.use_amp,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "warmup_epochs": self.warmup_epochs,
            })
            api = wandb.Api()
            ENTITY = "thibaut_hasle-epfl"
            PROJECT = "CBraMod-earEEG-tuning"
            PLOT_DIR = getattr(self.params, 'plot_dir', './artifacts/results/figures')
            os.makedirs(PLOT_DIR, exist_ok=True)

            runs = api.runs(f"{ENTITY}/{PROJECT}")
            records = []
            for run in runs:
                summary = run.summary
                config = run.config
                name = run.name
                records.append({
                    "name": name,
                    "test_kappa": summary.get("test_kappa"),
                    "test_accuracy": summary.get("test_accuracy"),
                    "test_f1": summary.get("test_f1"),
                    "val_kappa": summary.get("val_kappa"),
                    "hours_of_data": summary.get("hours_of_data"),
                    "dataset_names": summary.get("dataset_names"),
                    "num_subjects_train": summary.get("num_subjects_train"),
                    "epochs": summary.get("epochs"),
                    "scheduler": summary.get("scheduler"),
                })

            df = pd.DataFrame(records)
            results_dir = getattr(self.params, 'results_dir', './artifacts/results')
            os.makedirs(results_dir, exist_ok=True)
            df.to_csv(f"{results_dir}/experiment_summary.csv", index=False)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(
                best_f1_epoch, acc, kappa, f1
            )
            
            # Use memory manager for checkpoint management
            if self.memory_manager.manage_checkpoints(model_path):
                torch.save(self.model.state_dict(), model_path)
                print("✅ model saved in " + model_path)
            else:
                print("⚠️ Checkpoint limit reached, skipping save")
            
            # Log checkpoint summary
            checkpoint_summary = self.memory_manager.get_checkpoint_summary()
            print(f"📊 Checkpoint Summary: {checkpoint_summary['total_files']} files, "
                  f"{checkpoint_summary['total_size_mb']:.1f} MB total")

        self.close_wandb()
        
        # Cleanup memory after training
        self.memory_manager.cleanup_between_trials(trial_name)

        return kappa_best


    # def train_for_binaryclass(self):
    #     acc_best = 0
    #     roc_auc_best = 0
    #     pr_auc_best = 0
    #     cm_best = None
    #     for epoch in range(self.params.epochs):
    #         self.model.train()
    #         start_time = timer()
    #         losses = []
    #         for x, y in tqdm(self.data_loader['train'], mininterval=10):
    #             self.optimizer.zero_grad()
    #             x = x.cuda()
    #             y = y.cuda()
    #             pred = self.model(x)

    #             loss = self.criterion(pred, y)

    #             loss.backward()
    #             losses.append(loss.data.cpu().numpy())
    #             if self.params.clip_value > 0:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
    #                 # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
    #             self.optimizer.step()
    #             self.optimizer_scheduler.step()

    #         optim_state = self.optimizer.state_dict()

    #         with torch.no_grad():
    #             acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
    #             print(
    #                 "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
    #                     epoch + 1,
    #                     np.mean(losses),
    #                     acc,
    #                     pr_auc,
    #                     roc_auc,
    #                     optim_state['param_groups'][0]['lr'],
    #                     (timer() - start_time) / 60
    #                 )
    #             )
    #             print(cm)
    #             if roc_auc > roc_auc_best:
    #                 print("kappa increasing....saving weights !! ")
    #                 print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
    #                     acc,
    #                     pr_auc,
    #                     roc_auc,
    #                 ))
    #                 best_f1_epoch = epoch + 1
    #                 acc_best = acc
    #                 pr_auc_best = pr_auc
    #                 roc_auc_best = roc_auc
    #                 cm_best = cm
    #                 self.best_model_states = copy.deepcopy(self.model.state_dict())
    #     self.model.load_state_dict(self.best_model_states)
    #     with torch.no_grad():
    #         print("***************************Test************************")
    #         acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
    #         print("***************************Test results************************")
    #         print(
    #             "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
    #                 acc,
    #                 pr_auc,
    #                 roc_auc,
    #             )
    #         )
    #         print(cm)
    #         if not os.path.isdir(self.params.model_dir):
    #             os.makedirs(self.params.model_dir)
    #         model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
    #         torch.save(self.model.state_dict(), model_path)
    #         print("model save in " + model_path)

    # def train_for_regression(self):
    #     corrcoef_best = 0
    #     r2_best = 0
    #     rmse_best = 0
    #     for epoch in range(self.params.epochs):
    #         self.model.train()
    #         start_time = timer()
    #         losses = []
    #         for i, (x, y) in tqdm(self.data_loader['train']):
    #             print(f"Batch {i}: label distribution:", y.unique(return_counts=True))
    #             if i > 2:  # Only print a few batches
    #                 break

    #         for x, y in tqdm(self.data_loader['train'], mininterval=10):
    #             self.optimizer.zero_grad()
    #             x = x.cuda()
    #             y = y.cuda()
    #             pred = self.model(x)
    #             loss = self.criterion(pred, y)

    #             loss.backward()
    #             losses.append(loss.data.cpu().numpy())
    #             if self.params.clip_value > 0:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
    #                 # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
    #             self.optimizer.step()
    #             self.optimizer_scheduler.step()

    #         optim_state = self.optimizer.state_dict()

    #         with torch.no_grad():
    #             corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
    #             print(
    #                 "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
    #                     epoch + 1,
    #                     np.mean(losses),
    #                     corrcoef,
    #                     r2,
    #                     rmse,
    #                     optim_state['param_groups'][0]['lr'],
    #                     (timer() - start_time) / 60
    #                 )
    #             )
    #             if r2 > r2_best:
    #                 print("kappa increasing....saving weights !! ")
    #                 print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
    #                     corrcoef,
    #                     r2,
    #                     rmse,
    #                 ))
    #                 best_r2_epoch = epoch + 1
    #                 corrcoef_best = corrcoef
    #                 r2_best = r2
    #                 rmse_best = rmse
    #                 self.best_model_states = copy.deepcopy(self.model.state_dict())

    #     self.model.load_state_dict(self.best_model_states)
    #     with torch.no_grad():
    #         print("***************************Test************************")
    #         corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
    #         print("***************************Test results************************")
    #         print(
    #             "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
    #                 corrcoef,
    #                 r2,
    #                 rmse,
    #             )
    #         )

    #         if not os.path.isdir(self.params.model_dir):
    #             os.makedirs(self.params.model_dir)
    #         model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
    #         torch.save(self.model.state_dict(), model_path)
    #         print("model save in " + model_path)