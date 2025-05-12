import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
from Plot import plot_metrics
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

class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'IDUN_EEG']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None
        self.best_val_cm = None
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )

        self.n_params_total = sum(p.numel() for p in self.model.parameters())
        self.n_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.frozen_layers = len([m for m in getattr(self.model, "encoder", self.model).children()
                                   if all(not p.requires_grad for p in m.parameters())])

        self.lora_rank = getattr(self.model, "lora_rank", 0)

        train_dl = self.data_loader['train']
        self.train_steps_per_epoch = len(train_dl)
        _tmp = next(iter(train_dl))[0]
        self.seq_len = _tmp.shape[-1]
        del _tmp

        sample_window_sec = self.seq_len / self.params.sample_rate
        self.hours_of_data = (self.train_steps_per_epoch *
                              self.params.batch_size *
                              sample_window_sec) / 3600
        all_labels = [lab for _, y in train_dl for lab in y.tolist()]
        counts = Counter(all_labels)
        self.class_balance_ratio = {cls: c / len(all_labels) for cls, c in counts.items()}
        self.eval_split = getattr(self.params, "eval_split", "same-night")

        self.log_class_distribution(self.data_loader['train'], "train")
        self.log_class_distribution(self.data_loader['val'], "val")
        self.log_class_distribution(self.data_loader['test'], "test")

        print(self.model)

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


    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        best_f1_epoch = 0
        score_history = []
        log_history = []
        cm_best = None
        patience_counter = 0

        for epoch in range(self.params.epochs):
            total_train_start = timer()
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            if epoch % getattr(self.params, "eval_every", 1) == 0:
                with torch.no_grad():
                    acc, kappa, f1, cm, y_true, y_pred = self.val_eval.get_metrics_for_multiclass(self.model)
                    #self.print_classification_report(y_true, y_pred)

                    val_losses = []
                    for x_val, y_val in self.data_loader['val']:
                        x_val = x_val.cuda()
                        y_val = y_val.cuda()
                        pred_val = self.model(x_val)

                        if self.params.downstream_dataset == 'ISRUC':
                            val_loss = self.criterion(pred_val.transpose(1, 2), y_val)
                        else:
                            val_loss = self.criterion(pred_val, y_val)

                        val_losses.append(val_loss.item())

                    val_loss_mean = np.mean(val_losses)
                    print("Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time: {:.2f} mins".format(
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

                    if patience_counter >= getattr(self.params, "early_stop_patience", 10):
                        print("Early stopping triggered.")
                        break

                    log_history.append({
                        'epoch': epoch + 1,
                        'train_loss': np.mean(losses),
                        'val_loss': val_loss_mean,
                        'val_acc': acc,
                        'val_kappa': kappa,
                        'val_f1': f1
                    })

                score_history.append((epoch + 1, acc, kappa, f1))
                
        total_walltime_min = (timer() - total_train_start) / 60               # ➕ NEW
        train_steps = self.train_steps_per_epoch * self.params.epochs         # ➕ NEW
        tokens_seen = (train_steps *
                       self.params.batch_size *
                       self.seq_len)                                          # ➕ NEW
        petaFLOPs = 6 * self.n_params_total * tokens_seen / 1e15              # ➕ NEW
        gpu_hours = total_walltime_min / 60 * torch.cuda.device_count()       # ➕ NEW


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

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(
                best_f1_epoch, acc, kappa, f1
            )
            torch.save(self.model.state_dict(), model_path)
            print("✅ model saved in " + model_path)



            log_path = "scaling_laws.csv"
            fieldnames = [
                'model_name', 'model_size', 'layers', 'heads', 'embedding_dim',
                'data_fraction', 'num_subjects', 'seed',
                'accuracy', 'f1', 'kappa', 'baseline', 'comment'
                ,'n_params','n_trainable_params','frozen_layers','lora_rank',
                'train_steps','tokens_seen','petaFLOPs','walltime_min','gpu_hours',
                'lr_schedule','max_lr','weight_decay','dropout',
                'aug_noise_dB','mixup_p','time_mask_pct',
                'hours_of_data','n_sleep_epochs','class_balance_ratio',
                'eval_split'
            ]

            new_row = {
                'model_name': self.params.model_name,
                'model_size': self.params.model_size,
                'layers': self.params.layers,
                'heads': self.params.heads,
                'embedding_dim': self.params.embedding_dim,
                'data_fraction': self.params.data_fraction,
                'num_subjects': self.params.num_subjects,
                'seed': self.params.seed,
                'accuracy': acc,
                'f1': f1,
                'kappa': kappa,
                'baseline': self.params.baseline if hasattr(self.params, 'baseline') else False,
                'comment': self.params.comment if hasattr(self.params, 'comment') else "",
                # ➕ NEW  ------------  scaling & env fields  -------------
                'n_params': self.n_params_total,
                'n_trainable_params': self.n_params_trainable,
                'frozen_layers': self.frozen_layers,
                'lora_rank': self.lora_rank,
                'train_steps': train_steps,
                'tokens_seen': tokens_seen,
                'petaFLOPs': petaFLOPs,
                'walltime_min': total_walltime_min,
                'gpu_hours': gpu_hours,
                'lr_schedule': type(self.optimizer_scheduler).__name__,
                'max_lr': max([g['lr'] for g in optim_state['param_groups']]),
                'weight_decay': self.params.weight_decay,
                'dropout': self.params.dropout,
                'aug_noise_dB': getattr(self.params, 'aug_noise_dB', None),
                'mixup_p': getattr(self.params, 'mixup_p', None),
                'time_mask_pct': getattr(self.params, 'time_mask_pct', None),
                'hours_of_data': self.hours_of_data,
                'n_sleep_epochs': round(self.hours_of_data / 1.5, 1),
                'class_balance_ratio': json.dumps(self.class_balance_ratio),
                'eval_split': self.eval_split
            }

            # Append or create CSV
            file_exists = os.path.isfile(log_path)
            with open(log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(new_row)

            df = pd.read_csv("scaling_laws.csv")
            print(df.columns)
            print(df.head())
            #plot_metrics.plot_data_scaling(df, metric='accuracy')
            #plot_metrics.plot_model_scaling(df, metric='accuracy')
            #plot_metrics.plot_pareto(df, metric='accuracy')
            #plot_metrics.plot_scaling_fit(df)

            top_scores = sorted(score_history, key=lambda x: x[2], reverse=True)[:3]
            print("\n=== Top 3 Epochs by Kappa Score ===")
            for i, (epoch, acc, kappa, f1) in enumerate(top_scores, 1):
                print(f"Top {i}: Epoch {epoch}, Acc: {acc:.5f}, Kappa: {kappa:.5f}, F1: {f1:.5f}")

        # ✅ Return required values to make Optuna work
        return kappa_best


    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if roc_auc > roc_auc_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for i, (x, y) in tqdm(self.data_loader['train']):
                print(f"Batch {i}: label distribution:", y.unique(return_counts=True))
                if i > 2:  # Only print a few batches
                    break

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)