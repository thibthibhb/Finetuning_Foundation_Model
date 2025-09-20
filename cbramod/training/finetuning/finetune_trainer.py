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
import torch.nn.functional as F
from timeit import default_timer as timer
import numpy as np
import matplotlib as mpl
import copy
import os
import pandas as pd
import csv
import time
import subprocess
import json
from datetime import timedelta
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
from collections import Counter
from csv import QUOTE_MINIMAL
try:
    import wandb
except ImportError:
    wandb = None
import optuna
import gc
import logging
# CLAUDE-ENHANCEMENT: Unified AMP import for consistency
from torch import amp

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


class FocalLoss(nn.Module):
    """Focal Loss implementation for imbalanced classification
    
    Focal Loss down-weights easy examples and focuses on hard examples,
    helping with class imbalance issues in EEG sleep staging.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    Application: Improved N1 and REM recall for sleep staging
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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
        
        
        # Loss function will be initialized after class distribution is computed

        self.best_model_states = None
        self.best_val_cm = None
        
        # CLAUDE-ENHANCEMENT: Initialize mixed precision training with unified AMP
        self.use_amp = getattr(params, 'use_amp', True)
        if self.use_amp:
            self.scaler = amp.GradScaler('cuda')
            
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            checkpoint_dir=getattr(params, 'model_dir', './saved_models/finetuned')
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
                param_groups = [
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ]
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)

        elif self.params.optimizer == "Lion":                       # ⬅️ NEW
            LionCls = _get_lion_cls()
            if self.params.multi_lr:
                param_groups = [
                    {"params": backbone_params, "lr": self.params.lr},
                    {"params": other_params,  "lr": self.params.lr * 5},
                ]
                # if icl_params:
                #     param_groups.append({"params": icl_params, "lr": self.params.lr * 1})
                self.optimizer = LionCls(
                    param_groups,
                    lr=self.params.lr,
                    weight_decay=self.params.weight_decay,
                    betas=(0.9, 0.99),              # authors' defaults
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
                param_groups = [
                    {"params": backbone_params, "lr": self.params.lr},
                    {"params": other_params,  "lr": self.params.lr * 5},
                ]
                # if icl_params:
                #     param_groups.append({"params": icl_params, "lr": self.params.lr * 1})
                self.optimizer = torch.optim.SGD(
                    param_groups,
                    momentum=0.9,
                    weight_decay=self.params.weight_decay,
                )
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.params.lr, momentum=0.9,
                                weight_decay=self.params.weight_decay,)


        self.data_length = len(self.data_loader['train'])
        
        # Simple learning rate scheduling (removed warmup for simplicity)
        #self.total_steps = self.params.epochs * self.data_length // self.gradient_accumulation_steps
        
        # Create simple scheduler without warmup
        if params.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.optimizer_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.params.epochs,  # Cosine cycle over all epochs
                eta_min=0  # Minimum LR is 0
            )
            
        elif params.scheduler == "step":
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.params.epochs // 3), gamma=0.5)
        elif params.scheduler == "none":
            self.optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)

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
        all_labels = []
        for batch in train_dl:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, yb, _ = batch
            else:
                _, yb = batch
            all_labels.extend(yb.tolist())
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

        # Initialize loss function with real class weights computed from training data
        self._initialize_loss_function(device)

        # Initialize WandB with label space versioning
        self.wandb_run = None
        if getattr(self.params, 'use_wandb', True):
            # Prepare W&B config with label mapping metadata
            wandb_config = self.params.__dict__.copy()
            
            # Add label space metadata to config
            wandb_config['label_mapping_version'] = getattr(self.params, 'label_mapping_version', 'v1')
            wandb_config['label_space_description'] = getattr(self.params, 'label_space_description', '')
            
            # Get tags for label space versioning
            wandb_tags = getattr(self.params, 'label_space_tags', [])
            
            self.wandb_run = wandb.init(
                project="CBraMod-earEEG-tuning",
                config=wandb_config,
                tags=wandb_tags,
                reinit=True,
                name=self.params.run_name,
                dir='./experiments/wandb'
            )
            
            # Log label mapping info as a summary
            if self.wandb_run:
                self.wandb_run.summary['label_space_version'] = wandb_config['label_mapping_version']
                self.wandb_run.summary['label_space_description'] = wandb_config['label_space_description']
                print(f"📊 W&B Run initialized with tags: {wandb_tags}")

        



    # CLAUDE-ENHANCEMENT: Compute real class weights from training data
    def _compute_class_weights(self, device):
        """
        Compute class weights using inverse frequency weighting from actual training data.
        Uses the class_balance_ratio computed during initialization.
        
        Returns:
            class_weights: torch.Tensor of shape [num_classes] on specified device, or None if disabled
        """
        # Check if class weighting is enabled
        use_class_weights = getattr(self.params, 'use_class_weights', False)
        if not use_class_weights:
            print("📊 Class weighting disabled (use --use_class_weights to enable)")
            return None
            
        # Check if external class weights were provided
        external_weights = getattr(self.params, 'class_weights', None)
        if external_weights is not None:
            print(f"📊 Using provided class weights: {external_weights.cpu().numpy()}")
            return external_weights.to(device)
        
        # Compute inverse frequency weights from real training data
        if hasattr(self, 'class_balance_ratio') and self.class_balance_ratio:
            num_classes = len(self.class_balance_ratio)
            weights = torch.zeros(num_classes)
            
            # Inverse frequency weighting: weight = 1 / frequency
            for class_id, frequency in self.class_balance_ratio.items():
                weights[class_id] = 1.0 / frequency
            
            # Normalize weights so they sum to num_classes (standard practice)
            weights = weights * num_classes / weights.sum()
            
            print(f"📊 Computed real class weights from training data:")
            for class_id, (freq, weight) in enumerate(zip(self.class_balance_ratio.values(), weights)):
                print(f"   Class {class_id}: freq={freq:.3f} -> weight={weight:.3f}")
            
            return weights.to(device)
        else:
            print("⚠️ No class distribution available, using equal weights")
            return None

    def _initialize_loss_function(self, device):
        """Initialize loss function with proper class weighting based on dataset type."""
        # Loss function selection - Focal Loss for imbalanced EEG data or standard losses
        if getattr(self.params, 'downstream_dataset', 'IDUN_EEG') in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'IDUN_EEG']:
            if getattr(self.params, 'use_focal_loss', False):
                # Focal Loss for imbalanced sleep staging - helps with N1 and REM recall
                focal_gamma = getattr(self.params, 'focal_gamma', 2.0)
                self.criterion = FocalLoss(alpha=1.0, gamma=focal_gamma, reduction='mean').to(device)
                print(f"📊 Using Focal Loss with gamma={focal_gamma}")
            else:
                # Compute real class weights from training data distribution
                class_weights = self._compute_class_weights(device)
                self.criterion = CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=getattr(self.params, 'label_smoothing', 0.0)
                ).to(device)
                
                if class_weights is not None:
                    print(f"📊 Using CrossEntropyLoss with real class weights and label_smoothing={getattr(self.params, 'label_smoothing', 0.0)}")
                else:
                    print(f"📊 Using standard CrossEntropyLoss (no class weights) with label_smoothing={getattr(self.params, 'label_smoothing', 0.0)}")
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().to(device)
            print("📊 Using BCEWithLogitsLoss for binary classification")
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().to(device)
            print("📊 Using MSELoss for regression")

    # CLAUDE-ENHANCEMENT: Extract features method for Proto ICL enhancement
    def _extract_features(self, x):
        """Extract features using forward_features if available, otherwise fallback to logits."""
        if hasattr(self.model, 'forward_features'):
            return self.model.forward_features(x)
        else:
            # Fallback to logits if forward_features not available
            return self.model(x)

    # CLAUDE-ENHANCEMENT: Enhanced central support selection with global fallback
    def _select_support_idxs_central(self, z_cls_cpu, K: int, class_id=None):
        """
        Select K support samples closest to class center (most representative).
        If class has fewer than K samples, returns all available and flags for global fallback.
        """
        if len(z_cls_cpu) == 0:
            return torch.tensor([], dtype=torch.long), True  # empty, needs global fallback
        
        center = z_cls_cpu.mean(0, keepdim=True)  # [1, D]
        d = torch.cdist(z_cls_cpu, center).squeeze(1)  # [Nc]
        
        available_samples = len(z_cls_cpu)
        topk = min(K, available_samples)
        selected_idxs = torch.topk(-d, k=topk).indices  # closest first
        
        # Flag if we need global fallback (insufficient samples)
        needs_global_fallback = available_samples < K
        
        return selected_idxs, needs_global_fallback


    # CLAUDE-ENHANCEMENT: Subject alignment and shrinkage regularization
    def _apply_subject_alignment(self, features, subject_mean=None):
        """Center features per subject by subtracting subject mean."""
        if subject_mean is None:
            subject_mean = features.mean(0, keepdim=True)
        return features - subject_mean, subject_mean


    # CLAUDE-ENHANCEMENT: Temporal smoothing for EEG sleep staging sequences
    def _apply_temporal_smoothing(self, predictions, method='median', window_size=3):
        """
        Apply temporal smoothing to prediction sequences using median or majority voting.
        
        Args:
            predictions: [N] sequence of predictions (numpy array or torch tensor)
            method: 'median' or 'majority' filtering
            window_size: size of smoothing window (should be odd)
            
        Returns:
            smoothed_preds: [N] temporally smoothed predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        predictions = np.array(predictions)
        smoothed = predictions.copy()
        
        if len(predictions) <= window_size:
            return smoothed  # Too short for smoothing
        
        half_window = window_size // 2
        
        if method == 'median':
            # Apply median filter
            from scipy.ndimage import median_filter
            smoothed = median_filter(predictions.astype(float), size=window_size, mode='reflect')
            smoothed = smoothed.astype(predictions.dtype)
            
        elif method == 'majority':
            # Apply majority voting in sliding window
            for i in range(half_window, len(predictions) - half_window):
                window = predictions[i-half_window:i+half_window+1]
                # Find most frequent class in window
                unique, counts = np.unique(window, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                smoothed[i] = majority_class
        
        return smoothed

    def _apply_sleep_stage_hmm(self, predictions, transition_prior_strength=0.8):
        """
        Apply simple HMM-like temporal smoothing with sleep stage transition priors.
        
        Sleep stages typically follow patterns:
        - Wake (0) -> N1 (1) -> N2 (2) -> N3 (3) -> REM (4)
        - Transitions between distant stages are rare
        
        Args:
            predictions: [N] sequence of sleep stage predictions
            transition_prior_strength: how much to weight transition priors vs observations
            
        Returns:
            smoothed_preds: [N] HMM-smoothed predictions
        """
        if len(predictions) <= 1:
            return predictions
            
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        predictions = np.array(predictions)
        num_stages = 5  # Assume 5-class sleep staging (Wake, N1, N2, N3, REM)
        
        # Define sleep stage transition matrix (prior knowledge)
        # Rows = from, Columns = to
        transition_priors = np.array([
            [0.7, 0.2, 0.05, 0.03, 0.02],  # Wake -> [Wake, N1, N2, N3, REM]
            [0.1, 0.4, 0.4, 0.08, 0.02],   # N1 -> 
            [0.05, 0.15, 0.5, 0.25, 0.05], # N2 -> 
            [0.02, 0.05, 0.3, 0.6, 0.03],  # N3 -> 
            [0.1, 0.1, 0.2, 0.1, 0.5]      # REM -> 
        ])
        
        # Ensure valid predictions (clip to valid range)
        predictions = np.clip(predictions, 0, num_stages - 1)
        
        # Simple Viterbi-like smoothing
        smoothed = predictions.copy()
        
        for i in range(1, len(predictions)):
            current_pred = predictions[i]
            prev_pred = smoothed[i-1]  # Use previous smoothed prediction
            
            # Compute transition probability from previous to current
            transition_prob = transition_priors[prev_pred, current_pred]
            
            # If transition is unlikely, consider alternatives
            if transition_prob < (1 - transition_prior_strength):
                # Find most likely next stage given previous
                next_probs = transition_priors[prev_pred]
                most_likely_next = np.argmax(next_probs)
                
                # Blend with original prediction based on prior strength
                if np.random.random() < transition_prior_strength:
                    smoothed[i] = most_likely_next
                else:
                    smoothed[i] = current_pred
        
        return smoothed

    def _apply_full_temporal_smoothing(self, predictions, subject_ids=None, 
                                     enable_median=True, enable_hmm=True,
                                     median_window=3, hmm_strength=0.6):
        """
        Apply comprehensive temporal smoothing combining median filtering and HMM.
        
        Args:
            predictions: [N] predictions to smooth
            subject_ids: [N] subject identifiers (smooth within subjects)
            enable_median: whether to apply median filtering
            enable_hmm: whether to apply HMM smoothing
            median_window: window size for median filtering
            hmm_strength: strength of HMM transition priors
            
        Returns:
            smoothed_preds: [N] fully smoothed predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        predictions = np.array(predictions)
        smoothed = predictions.copy()
        
        if subject_ids is None:
            # Treat as single sequence
            if enable_median:
                smoothed = self._apply_temporal_smoothing(smoothed, 'median', median_window)
            if enable_hmm:
                smoothed = self._apply_sleep_stage_hmm(smoothed, hmm_strength)
        else:
            # Apply smoothing per subject
            unique_subjects = np.unique(subject_ids)
            for subj in unique_subjects:
                if subj == 'UNK':  # Skip unknown subjects
                    continue
                    
                subj_mask = (subject_ids == subj)
                subj_preds = smoothed[subj_mask]
                
                if len(subj_preds) > 1:  # Need at least 2 samples for smoothing
                    if enable_median:
                        subj_preds = self._apply_temporal_smoothing(subj_preds, 'median', median_window)
                    if enable_hmm:
                        subj_preds = self._apply_sleep_stage_hmm(subj_preds, hmm_strength)
                    
                    smoothed[subj_mask] = subj_preds
        
        return smoothed


    # CLAUDE-ENHANCEMENT: Metric-friendly training with contrastive and prototypical losses
    def _compute_supervised_contrastive_loss(self, features, labels, temperature=0.1):
        """
        Compute supervised contrastive loss to make embeddings more cluster-friendly.
        
        Args:
            features: [N, D] L2-normalized embedding features
            labels: [N] class labels
            temperature: temperature parameter for contrastive loss
            
        Returns:
            contrastive_loss: scalar loss value
        """
        device = features.device
        batch_size = features.size(0)
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Create label mask - positive pairs have same label
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), temperature
        )  # [N, N]
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)  # Number of positive pairs per sample
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # Avoid division by zero
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


    def log_class_distribution(self, loader, name="train"):
        labels = []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, yb, _ = batch
            else:
                _, yb = batch
            labels.extend(yb.tolist())
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
        """Log metrics to W&B with label space version prefixes."""
        if self.wandb_run:
            # Add label space version prefix to metric names for clarity
            label_version = getattr(self.params, 'label_mapping_version', 'v1')
            num_classes = getattr(self.params, 'num_of_classes', 5)
            
            # Create version-aware metric names
            versioned_metrics = {}
            for key, value in metrics.items():
                if num_classes == 4:
                    # Add version prefix for 4-class metrics to distinguish v0 vs v1
                    versioned_key = f"{key}_4c{label_version}"
                else:
                    # 5-class doesn't need version prefix (always v1)
                    versioned_key = key
                    
                versioned_metrics[versioned_key] = value
                # Also log original key for backwards compatibility
                versioned_metrics[key] = value
            
            wandb.log(versioned_metrics)

    def close_wandb(self):
        if self.wandb_run:
            wandb.finish()

    def _update_optimizer_for_two_phase(self, phase):
        """Update optimizer for two-phase training"""
        if not hasattr(self.params, 'head_lr') or not hasattr(self.params, 'backbone_lr'):
            print("⚠️ head_lr and backbone_lr not found, using params.lr as fallback")
            # Use the main learning rate as head_lr, and much smaller for backbone  
            head_lr = self.params.lr
            backbone_lr = self.params.lr / 10
        else:
            head_lr = self.params.head_lr
            backbone_lr = self.params.backbone_lr
        
        # Get parameter groups with current state
        param_groups = self.model.get_param_groups(head_lr=head_lr, backbone_lr=backbone_lr)
        
        if not param_groups:
            print("⚠️ No trainable parameters found")
            return
            
        # Create new optimizer with updated parameter groups (using same logic as original)
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
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.params.weight_decay)
        elif self.params.optimizer == "Lion":
            LionCls = _get_lion_cls()
            self.optimizer = LionCls(param_groups, weight_decay=self.params.weight_decay, betas=(0.9, 0.99))
        elif self.params.optimizer == "AMSGrad":
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.params.weight_decay, amsgrad=True)
        else:  # SGD fallback
            self.optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.params.weight_decay)
        
        # Print current state
        total_params = sum(len(group['params']) for group in param_groups)
        print(f"📊 Phase {phase}: Updated optimizer with {len(param_groups)} parameter groups, {total_params} total trainable params")
        for i, group in enumerate(param_groups):
            print(f"   Group {i+1} ({group['name']}): {len(group['params'])} params, lr={group['lr']:.2e}")


    @torch.no_grad()
    def _extract_features(self, x):
        try:
            if hasattr(self.model, 'extract_features'):
                return self.model.extract_features(x)
            if hasattr(self.model, 'forward_features'):
                return self.model.forward_features(x)
        except Exception:
            pass
        # TEMP fallback:
        out = self.model(x)
        if out.ndim > 2:
            out = out.mean(dim=tuple(range(2, out.ndim)))
        return out


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
        
        # Initialize two-phase training before the epoch loop if needed
        if hasattr(self.params, 'two_phase_training') and self.params.two_phase_training:
            print(f"🚀 Initializing two-phase training")
            self.model.set_progressive_unfreezing_mode(phase=1)
            self._update_optimizer_for_two_phase(phase=1)
            
            # Log phase 1 initialization to W&B
            if hasattr(self, 'wandb_run') and self.wandb_run:
                wandb.log({
                    "training_phase": 1,
                    "phase1_epochs": self.params.phase1_epochs,
                    "backbone_frozen": True,
                    "two_phase_training_enabled": True
                }, step=0)

        for epoch in range(self.params.epochs):
            # Two-phase training logic - only switch at phase transition
            if hasattr(self.params, 'two_phase_training') and self.params.two_phase_training:
                if epoch == self.params.phase1_epochs:
                    # Phase 2: Switch to unfrozen backbone  
                    print(f"🔄 Switching to Phase 2 at epoch {epoch}")
                    self.model.set_progressive_unfreezing_mode(phase=2)
                    self._update_optimizer_for_two_phase(phase=2)
                    
                    # Log phase transition to W&B
                    if hasattr(self, 'wandb_run') and self.wandb_run:
                        wandb.log({
                            "training_phase": 2,
                            "phase_transition_epoch": epoch,
                            "backbone_unfrozen": True
                        }, step=epoch)
            
            total_train_start = timer()
            self.model.train()  # Set model to train mode
            
            # In two-phase training, ensure backbone stays in correct mode
            if hasattr(self.params, 'two_phase_training') and self.params.two_phase_training:
                if epoch < self.params.phase1_epochs:
                    # Phase 1: Keep backbone in eval mode
                    self.model.backbone.eval()
                else:
                    # Phase 2: Backbone should be in train mode  
                    self.model.backbone.train()
            
            start_time = timer()
            losses = []
            
            # Reset gradient accumulation
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, sid = batch
                else:
                    x, y = batch
                    sid = None
                x = x.to(self.device)
                y = y.to(self.device)
                
                # CLAUDE-ENHANCEMENT: Forward pass with ICL episodic training option
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                loss_components = {}
                
                # Standard forward pass (default path since ICL/proto methods are commented out)
                with amp.autocast('cuda', enabled=self.use_amp):
                    pred = self.model(x)
                    
                    # Ensure labels are LongTensor for CrossEntropyLoss  
                    y = y.to(device=pred.device, dtype=torch.long)
                    
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = self.criterion(pred.float().transpose(1, 2), y)
                    else:
                        loss = self.criterion(pred.float(), y)
                        
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                losses.append(loss.data.cpu().numpy())  # No accumulation scaling

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
                    for batch in self.data_loader['val']:
                        if isinstance(batch, (list, tuple)) and len(batch) == 3:
                            x_val, y_val, _ = batch
                        else:
                            x_val, y_val = batch

                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)

                        # CLAUDE-ENHANCEMENT: Validation with unified AMP
                        with amp.autocast('cuda', enabled=self.use_amp):
                            pred_val = self.model(x_val)
                        
                        # Ensure labels are LongTensor for CrossEntropyLoss
                        y_val = y_val.to(device=pred_val.device, dtype=torch.long)
                        
                        if self.params.downstream_dataset == 'ISRUC':
                            val_loss = self.criterion(pred_val.float().transpose(1, 2), y_val)
                        else:
                            val_loss = self.criterion(pred_val.float(), y_val)
                            

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
                #                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                #"warmup_epochs": self.warmup_epochs,
            })
            api = wandb.Api()
            ENTITY = "thibaut_hasle-epfl"
            PROJECT = "CBraMod-earEEG-tuning"
            PLOT_DIR = getattr(self.params, 'plot_dir', './experiments/results/figures')
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
            results_dir = getattr(self.params, 'results_dir', './experiments/results')
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


