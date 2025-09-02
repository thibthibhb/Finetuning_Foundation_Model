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

# CLAUDE-COMMENTED-OUT: Old AMP import with try/except
# try:
#     from torch.amp import GradScaler, autocast  # PyTorch >= 1.10
# except ImportError:
#     from torch.cuda.amp import GradScaler, autocast  # PyTorch < 1.10
# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# """
# CLAUDE-ENHANCED TIER-A PROTO ICL IMPLEMENTATION - COMMENTED OUT
# 
# This file has been enhanced with comprehensive Tier-A Proto ICL improvements to achieve:
# • Stable positive Δκ gains across all K values
# • Robust prototypical learning with minimal negative cases
# • EEG-specific temporal smoothing and transition modeling
# 
# === ENHANCEMENT SUMMARY ===
# 
# A. ✅ ENCODER FEATURES
#    - Added forward_features() to Model class for pre-classifier embeddings
#    - Updated _extract_features() to prefer encoder features over logits
#    - Reason: Encoder embeddings form better metric spaces than logits
# 
# B. ✅ BETTER SUPPORT SELECTION  
#    - Central selection: picks K samples closest to class centroid (not random)
#    - Global fallback: uses global prototypes when classes have insufficient samples
#    - Enhanced _select_support_idxs_central() with fallback detection
# 
# C. ✅ SUBJECT ALIGNMENT & SHRINKAGE
#    - Subject centering: subtracts per-subject feature mean for alignment
#    - Shrinkage regularization: blends subject prototypes with global prototypes (λ=5.0)
#    - Stabilizes rare/missing classes with global knowledge
# 
# D. ✅ CALIBRATION
#    - LOO temperature tuning: optimizes τ using leave-one-out validation on support set
#    - Confidence fallback: uses baseline predictions when proto confidence margin < 0.02
#    - Prevents low-confidence prototypical predictions
# 
# E. ✅ TEMPORAL SMOOTHING (EEG-specific)
#    - Median/majority filtering over adjacent epochs (window=3-5)
#    - Sleep stage HMM: applies transition priors for realistic stage sequences
#    - Subject-specific smoothing preserves individual sleep patterns
# 
# F. ✅ METRIC-FRIENDLY TRAINING
#    - Supervised contrastive loss: makes embeddings cluster by class & subject
#    - Prototypical loss: episodic training teaches metric-space properties
#    - Configurable loss weights for balanced training
# 
# === USAGE ===
# 
# 1. Standard Proto ICL (existing behavior):
#    --icl_mode proto --k_support 5
# 
# 2. Enhanced Proto ICL with all features:
#    --icl_mode proto --k_support 5 --use_metric_friendly_training True
#    --use_temporal_smoothing True --contrastive_weight 0.1
# 
# 3. Configuration parameters:
#    - use_metric_friendly_training: Enable contrastive + prototypical losses
#    - contrastive_weight: Weight for supervised contrastive loss (default: 0.1)
#    - prototypical_weight: Weight for episodic prototypical loss (default: 0.1)
#    - use_temporal_smoothing: Enable EEG temporal smoothing
#    - temporal_smoothing_window: Median filter window size (default: 3)
# 
# === BACKWARD COMPATIBILITY ===
# All original code preserved with "CLAUDE-COMMENTED-OUT" markers.
# Existing APIs unchanged - enhancements activate automatically based on configuration.
# """

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
        
        # In-Context Learning (ICL) config - COMMENTED OUT
        # self.icl_mode   = getattr(self.params, 'icl_mode', 'none')
        # self.k_support  = getattr(self.params, 'k_support', 0)
        # self.proto_temp = getattr(self.params, 'proto_temp', 0.1)
        # self.icl_eval_Ks = [int(x) for x in str(getattr(self.params, 'icl_eval_Ks', '0')).split(',')]
        # self.icl_hidden = getattr(self.params, 'icl_hidden', 256)
        # self.icl_layers = getattr(self.params, 'icl_layers', 2)
        
        # Initialize ICL head if needed - COMMENTED OUT
        # self.icl_head = None
        # if self.icl_mode in ['cnp', 'set']:
        #     from cbramod.models.icl_heads import create_icl_head
        #     z_dim = 512  # Feature dimension from model.forward_features()
        #     self.icl_head = create_icl_head(
        #         icl_mode=self.icl_mode,
        #         z_dim=z_dim,
        #         num_classes=self.params.num_of_classes,
        #         icl_hidden=self.icl_hidden,
        #         icl_layers=self.icl_layers
        #     )
        #     if self.icl_head is not None:
        #         self.icl_head = self.icl_head.to(device)
        #         print(f"✅ Initialized ICL head ({self.icl_mode}) with z_dim={z_dim}, hidden={self.icl_hidden}")
        
        # Loss function will be initialized after class distribution is computed

        self.best_model_states = None
        self.best_val_cm = None
        
        # CLAUDE-ENHANCEMENT: Initialize mixed precision training with unified AMP
        self.use_amp = getattr(params, 'use_amp', True)
        if self.use_amp:
            self.scaler = amp.GradScaler('cuda')
            
        # CLAUDE-COMMENTED-OUT: Old AMP initialization
        # # Initialize mixed precision training
        # self.use_amp = getattr(params, 'use_amp', True)
        # if self.use_amp:
        #     try:
        #         self.scaler = GradScaler('cuda')  # New API
        #     except TypeError:
        #         self.scaler = GradScaler()  # Fallback for older PyTorch
        #else:
        #     self.scaler = None
        # self.gradient_accumulation_steps = getattr(params, 'gradient_accumulation_steps', 1)
        
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
        
        # Separate ICL head parameters for different learning rate - COMMENTED OUT
        # icl_params = []
        # if self.icl_head is not None:
        #     for param in self.icl_head.parameters():
        #         icl_params.append(param)


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
                # Add ICL head with conservative learning rate - COMMENTED OUT
                # if icl_params:
                #     param_groups.append({'params': icl_params, 'lr': self.params.lr * 1})
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
        #else:
            # Default fallback for any unrecognized scheduler types
            #print(f"⚠️ Unknown scheduler '{params.scheduler}', defaulting to cosine with warmup")
            # def lr_lambda(step):
            #     if step < self.warmup_steps:
            #         return step / self.warmup_steps
            #     else:
            #         progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            #         return 0.5 * (1 + np.cos(np.pi * progress))
            # self.optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

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

        #print(self.model)
        
        # CLAUDE-ENHANCEMENT: Compute global prototypes for Proto ICL enhancement - COMMENTED OUT
        # self.global_protos = None
        # if hasattr(params, 'icl_mode') and params.icl_mode == 'proto':
        #     self._compute_global_protos()
            
        # CLAUDE-ENHANCEMENT: Metric-friendly training parameters - COMMENTED OUT
        # self.use_metric_friendly_training = getattr(params, 'use_metric_friendly_training', False)
        # self.contrastive_weight = getattr(params, 'contrastive_weight', 0.1)
        # self.prototypical_weight = getattr(params, 'prototypical_weight', 0.1)
        # self.use_temporal_smoothing = getattr(params, 'use_temporal_smoothing', False)
        
        # CLAUDE-ENHANCEMENT: Strengthened ICL training parameters - COMMENTED OUT
        # self.icl_loss_weight = getattr(params, 'icl_loss_weight', 0.15)  # Increased from 0.05
        # self.icl_contrastive_weight = getattr(params, 'icl_contrastive_weight', 0.1)  # New: contrastive loss for ICL features
        # self.temporal_smoothing_window = getattr(params, 'temporal_smoothing_window', 3)

    # @torch.no_grad() - COMMENTED OUT
    # def _compute_global_protos(self):
    #     """Compute global prototypes from training data using a safe (num_workers=0) loader."""
    #     C = self.params.num_of_classes
    #     dev = self.device
    #     feats_list, labs_list = [], []
    # 
    #     print("🔄 Computing global prototypes from training data (safe loader)...")
    # 
    #     # Build a temporary, zero-worker loader over the same dataset
    #     train_loader = self.data_loader['train']
    #     ds = getattr(train_loader, 'dataset', None)
    #     bs = getattr(train_loader, 'batch_size', 64)
    #     collate_fn = getattr(train_loader, 'collate_fn', None)
    # 
    #     if ds is None:
    #         # Fallback: iterate the current loader (rare), but it may still fail if it has many workers
    #         tmp_loader = train_loader
    #     else:
    #         from torch.utils.data import DataLoader
    #         tmp_loader = DataLoader(
    #             ds,
    #             batch_size=bs,
    #             shuffle=False,
    #             num_workers=0,             # ✅ key change
    #             pin_memory=False,          # minimize extra resources here
    #             persistent_workers=False,  # ✅ ensure no long-lived workers
    #             drop_last=False,
    #             collate_fn=collate_fn,
    #             prefetch_factor=None       # ignored when num_workers==0, but explicit for clarity
    #         )
    # 
    #     for batch in tmp_loader:
    #         if isinstance(batch, (list, tuple)) and len(batch) == 3:
    #             x, y, _ = batch
    #         else:
    #             x, y = batch
    #         x = x.to(dev)
    # 
    #         with amp.autocast('cuda', enabled=self.use_amp):
    #             f = self._extract_features(x).detach()
    #         feats_list.append(f.cpu())
    #         labs_list.append(y.cpu())
    # 
    #     F = torch.cat(feats_list)
    #     L = torch.cat(labs_list)
    # 
    #     protos = []
    #     for c in range(C):
    #         m = (L == c)
    #         if m.any():
    #             protos.append(F[m].mean(0))
    #         else:
    #             protos.append(torch.zeros(F.size(1)))
    #     self.global_protos = torch.stack(protos)  # keep on CPU
    #     print(f"✅ Computed global prototypes for {C} classes, shape: {self.global_protos.shape}")
    pass


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

    # def _get_global_class_prototype(self, class_id, feature_dim, device, subject_mean=None): - COMMENTED OUT
    pass

    #     """Get global class prototype with optional subject centering."""
    #     if self.global_protos is not None and class_id < len(self.global_protos):
    #         proto = self.global_protos[class_id].to(device)
    #         if subject_mean is not None:
    #             # Assume subject_mean is already on correct device and properly shaped
    #             if subject_mean.dim() > 1:
    #                 subject_mean = subject_mean.squeeze(0)
    #             # Center global prototype in subject's feature space
    #             proto = proto - subject_mean
    #         return proto
    #     else:
    #         # Fallback to zero vector if no global prototypes available
    #         return torch.zeros(feature_dim, device=device)
    pass

    # CLAUDE-ENHANCEMENT: Subject alignment and shrinkage regularization
    def _apply_subject_alignment(self, features, subject_mean=None):
        """Center features per subject by subtracting subject mean."""
        if subject_mean is None:
            subject_mean = features.mean(0, keepdim=True)
        return features - subject_mean, subject_mean

    # def _compute_shrunk_prototypes(self, support_features, support_labels, num_classes, - COMMENTED OUT 
    pass

    #                              subject_mean, lambda_shrinkage=5.0):
    #     """
    #     Compute class prototypes with shrinkage toward global prototypes.
    #     
    #     Args:
    #         support_features: [N_sup, D] subject-centered features
    #         support_labels: [N_sup] class labels  
    #         num_classes: number of classes
    #         subject_mean: [1, D] subject mean for centering global prototypes
    #         lambda_shrinkage: shrinkage strength toward global prototypes
    #     
    #     Returns:
    #         prototypes: [C, D] shrunk prototypes on device
    #     """
    #     device = support_features.device
    #     protos_list = []
    #     
    #     for c in range(num_classes):
    #         class_mask = (support_labels == c)
    #         
    #         if class_mask.any():
    #             # Compute subject-specific class prototype
    #             n_samples = int(class_mask.sum())
    #             subject_proto = support_features[class_mask].mean(0)  # [D]
    #             
    #             # Get global prototype (centered in subject space)
    #             global_proto = self._get_global_class_prototype(
    #                 c, support_features.size(1), device, subject_mean
    #             )
    #             
    #             # Shrinkage: combine subject and global prototypes
    #             shrunk_proto = (n_samples * subject_proto + lambda_shrinkage * global_proto) / (n_samples + lambda_shrinkage)
    #             protos_list.append(shrunk_proto)
    #         else:
    #             # No samples for this class - use global prototype only
    #             global_proto = self._get_global_class_prototype(
    #                 c, support_features.size(1), device, subject_mean
    #             )
    #             protos_list.append(global_proto)
    #     
    #     prototypes = torch.stack(protos_list)  # [C, D]
    #     return _l2norm(prototypes, dim=-1)  # L2 normalize for cosine similarity
    pass

    # CLAUDE-ENHANCEMENT: Temperature calibration and confidence fallback
    # def _tune_temperature_loo(self, support_features, support_labels, prototypes, - COMMENTED OUT 
    pass

    #                          temp_grid=(0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)):
    #     """
    #     Tune temperature using leave-one-out validation on support set.
    #     
    #     Args:
    #         support_features: [N_sup, D] L2-normalized features
    #         support_labels: [N_sup] class labels
    #         prototypes: [C, D] L2-normalized prototypes
    #         temp_grid: candidate temperatures to try
    #         
    #     Returns:
    #         best_temperature: optimal temperature
    #     """
    #     if len(support_features) < 4:  # Too few samples for LOO
    #         return self.proto_temp  # fallback to default
    #     
    #     device = support_features.device
    #     num_classes = len(prototypes)
    #     best_temp, best_acc = temp_grid[0], -1.0
    #     
    #     for temp in temp_grid:
    #         correct = 0
    #         total = 0
    #         
    #         for i in range(len(support_features)):
    #             # Leave-one-out: exclude sample i
    #             loo_mask = torch.ones(len(support_features), dtype=torch.bool, device=device)
    #             loo_mask[i] = False
    #             
    #             if loo_mask.sum() == 0:  # Skip if no samples left
    #                 continue
    #                 
    #             # Recompute prototypes without sample i
    #             loo_features = support_features[loo_mask]
    #             loo_labels = support_labels[loo_mask]
    #             loo_protos = _compute_prototypes(loo_features, loo_labels, num_classes)
    #             loo_protos = _l2norm(loo_protos, dim=-1)
    #             
    #             # Predict on left-out sample
    #             query_feat = support_features[i:i+1]  # [1, D]
    #             logits = _proto_logits(query_feat, loo_protos, temperature=temp)
    #             pred = logits.argmax(-1).item()
    #             true_label = support_labels[i].item()
    #             
    #             correct += int(pred == true_label)
    #             total += 1
    #         
    #         if total > 0:
    #             acc = correct / total
    #             if acc > best_acc:
    #                 best_acc, best_temp = acc, temp
    #     
    #     return best_temp
    pass

    # def _compute_prediction_confidence(self, query_features, prototypes, temperature): - COMMENTED OUT
    pass

    #     """
    #     Compute prediction confidence based on margin between top-2 similarities.
    #     
    #     Args:
    #         query_features: [N_query, D] L2-normalized features
    #         prototypes: [C, D] L2-normalized prototypes
    #         temperature: temperature parameter
    #         
    #     Returns:
    #         confidences: [N_query] confidence scores (top1 - top2 margin)
    #         predictions: [N_query] predicted class indices
    #     """
    #     # Compute similarities (cosine since both are L2-normalized)
    #     similarities = query_features @ prototypes.T  # [N_query, C]
    #     
    #     # Get top-2 similarities for confidence margin
    #     top2_sims, top2_indices = torch.topk(similarities, k=min(2, similarities.size(1)), dim=1)
    #     
    #     if similarities.size(1) >= 2:
    #         # Confidence = margin between top-2 similarities
    #         confidences = top2_sims[:, 0] - top2_sims[:, 1]  # [N_query]
    #     else:
    #         # Only one class - use top similarity as confidence
    #         confidences = top2_sims[:, 0]
    #     
    #     # Apply temperature and get predictions
    #     logits = similarities / temperature
    #     predictions = logits.argmax(dim=1)
    #     
    #     return confidences, predictions
    pass

    # def _apply_confidence_fallback(self, proto_preds, baseline_preds, confidences, - COMMENTED OUT 
    pass

    #                              confidence_threshold=0.02):
    #     """
    #     Apply confidence-based fallback to baseline predictions.
    #     
    #     Args:
    #         proto_preds: [N] prototypical predictions (CUDA tensor)
    #         baseline_preds: [N] baseline model predictions (CPU tensor)
    #         confidences: [N] confidence scores (CUDA tensor)
    #         confidence_threshold: minimum confidence for using proto predictions
    #         
    #     Returns:
    #         final_preds: [N] final predictions with fallback applied (CUDA tensor)
    #         fallback_mask: [N] boolean mask indicating which samples used fallback (CUDA tensor)
    #     """
    #     # Ensure all tensors are on the same device
    #     device = proto_preds.device
    #     
    #     # Move baseline_preds to same device as proto_preds
    #     if isinstance(baseline_preds, torch.Tensor):
    #         baseline_preds = baseline_preds.to(device)
    #     else:
    #         baseline_preds = torch.tensor(baseline_preds, device=device)
    #     
    #     # Compute fallback mask
    #     fallback_mask = confidences < confidence_threshold
    #     
    #     # Apply fallback
    #     final_preds = proto_preds.clone()
    #     final_preds[fallback_mask] = baseline_preds[fallback_mask]
    #     
    #     return final_preds, fallback_mask
    pass

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

    # CLAUDE-ENHANCEMENT: Episodic training methods for ICL heads
    # def _create_episodic_batch_cnp(self, batch, n_way=None, k_shot=5, q_query=10): - COMMENTED OUT
#        """
#        Create episodic batches for CNP/Set Transformer ICL training.
#        
#        Args:
#            batch: original batch (x, y) or (x, y, sid)
#            n_way: number of classes per episode (None = use all available)
#            k_shot: number of support samples per class
#            q_query: number of query samples per class
#            
#        Returns:
#            support_x, support_y, query_x, query_y or None if not enough samples
#        """
#        if isinstance(batch, (list, tuple)) and len(batch) == 3:
#            x, y, sid = batch
#        else:
#            x, y = batch
#            
#        x, y = x.to(self.device), y.to(self.device)
#        
#        # Get unique classes in batch
#        unique_classes = torch.unique(y)
#        
#        if n_way is None:
#            n_way = len(unique_classes)
#        elif len(unique_classes) < n_way:
#            return None  # Not enough classes for episode
#        
#        # Randomly select n_way classes
#        if n_way < len(unique_classes):
#            selected_classes = unique_classes[torch.randperm(len(unique_classes))[:n_way]]
#        else:
#            selected_classes = unique_classes
#        
#        support_x_list, support_y_list = [], []
#        query_x_list, query_y_list = [], []
#        
#        for class_idx, cls in enumerate(selected_classes):
#            class_mask = (y == cls)
#            class_samples = x[class_mask]
#            
#            if len(class_samples) < k_shot + q_query:
#                return None  # Not enough samples for this class
#            
#            # Randomly split into support and query
#            indices = torch.randperm(len(class_samples))
#            support_indices = indices[:k_shot]
#            query_indices = indices[k_shot:k_shot + q_query]
#            
#            support_x_list.append(class_samples[support_indices])
#            # CLAUDE-ENHANCEMENT: Use episode-local class IDs (0, 1, 2, ...) for better ICL learning
#            support_y_list.append(torch.full((k_shot,), class_idx, dtype=torch.long, device=self.device))
#            
#            query_x_list.append(class_samples[query_indices])
#            query_y_list.append(torch.full((q_query,), class_idx, dtype=torch.long, device=self.device))
#        
#        # Concatenate all support and query samples
#        support_x = torch.cat(support_x_list, dim=0)
#        support_y = torch.cat(support_y_list, dim=0)
#        query_x = torch.cat(query_x_list, dim=0)
#        query_y = torch.cat(query_y_list, dim=0)
#        
#        return support_x, support_y, query_x, query_y
#
    # def _compute_icl_loss(self, support_x, support_y, query_x, query_y): - COMMENTED OUT
#        """
#        Compute enhanced ICL loss with supervised contrastive learning for Set-ICL training.
#        
#        Args:
#            support_x: [N_sup, ...] support samples
#            support_y: [N_sup] support labels  
#            query_x: [N_query, ...] query samples
#            query_y: [N_query] query labels
#            
#        Returns:
#            icl_loss: scalar loss value (CE + supervised contrastive)
#        """
#        if self.icl_head is None:
#            return torch.tensor(0.0, device=self.device, requires_grad=True)
#        
#        with amp.autocast('cuda', enabled=self.use_amp):
#            # Extract features using model backbone
#            support_features = self.model.forward_features(support_x)  # [N_sup, z_dim]
#            query_features = self.model.forward_features(query_x)      # [N_query, z_dim]
#            
#            # ICL head forward pass
#            icl_logits = self.icl_head(support_features, support_y, query_features)  # [N_query, num_classes]
#            
#            # Primary cross-entropy loss
#            icl_ce_loss = F.cross_entropy(icl_logits, query_y)
#            
#            # CLAUDE-ENHANCEMENT: Add supervised contrastive loss on ICL episode features
#            icl_contrastive_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
#            
#            if self.icl_contrastive_weight > 0 and len(support_features) > 1:
#                # Combine support and query features for contrastive learning
#                all_features = torch.cat([support_features, query_features], dim=0)
#                all_labels = torch.cat([support_y, query_y], dim=0)
#                
#                # Apply L2 normalization for contrastive learning
#                all_features_normalized = F.normalize(all_features, dim=-1)
#                
#                # Compute supervised contrastive loss on ICL episode features
#                icl_contrastive_loss = self._compute_supervised_contrastive_loss(
#                    all_features_normalized, all_labels, temperature=0.07  # Lower temp for ICL
#                )
#            
#            # Combine losses
#            total_icl_loss = icl_ce_loss + self.icl_contrastive_weight * icl_contrastive_loss
#        
#        return total_icl_loss
#
    pass

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

    # def _compute_prototypical_loss(self, support_features, support_labels, - COMMENTED OUT 
    pass

    #                              query_features, query_labels, num_classes):
    #     """
    #     Compute prototypical loss for episodic training.
    #     
    #     Args:
    #         support_features: [N_sup, D] support set features
    #         support_labels: [N_sup] support set labels
    #         query_features: [N_query, D] query set features  
    #         query_labels: [N_query] query set labels
    #         num_classes: number of classes
    #         
    #     Returns:
    #         proto_loss: scalar loss value
    #     """
    #     # Compute prototypes from support set
    #     prototypes = _compute_prototypes(support_features, support_labels, num_classes)
    #     prototypes = _l2norm(prototypes, dim=-1)
    #     
    #     # Compute distances from queries to prototypes
    #     query_features = _l2norm(query_features, dim=-1)
    #     distances = torch.cdist(query_features, prototypes)  # [N_query, C]
    #     
    #     # Convert distances to logits (negative distance)
    #     logits = -distances
    #     
    #     # Compute cross-entropy loss
    #     proto_loss = F.cross_entropy(logits, query_labels)
    #     
    #     return proto_loss
    pass

    # def _create_episodic_batch(self, batch, n_way=5, k_shot=5, q_query=10): - COMMENTED OUT
#        """
#        Create episodic batches for prototypical training.
#        
#        Args:
#            batch: original batch (x, y) or (x, y, sid)
#            n_way: number of classes per episode
#            k_shot: number of support samples per class
#            q_query: number of query samples per class
#            
#        Returns:
#            support_x, support_y, query_x, query_y or None if not enough samples
#        """
#        if isinstance(batch, (list, tuple)) and len(batch) == 3:
#            x, y, sid = batch
#        else:
#            x, y = batch
#            
#        x, y = x.cuda(), y.cuda()
#
#        # Get unique classes in batch
#        unique_classes = torch.unique(y)
#        
#        if len(unique_classes) < n_way:
#            return None  # Not enough classes for episode
#        
#        # Randomly select n_way classes
#        selected_classes = unique_classes[torch.randperm(len(unique_classes))[:n_way]]
#        
#        support_x_list, support_y_list = [], []
#        query_x_list, query_y_list = [], []
#        
#        for class_idx, cls in enumerate(selected_classes):
#            class_mask = (y == cls)
#            class_samples = x[class_mask]
#            
#            if len(class_samples) < k_shot + q_query:
#                return None  # Not enough samples for this class
#            
#            # Randomly split into support and query
#            indices = torch.randperm(len(class_samples))
#            support_indices = indices[:k_shot]
#            query_indices = indices[k_shot:k_shot + q_query]
#            
#            support_x_list.append(class_samples[support_indices])
#            support_y_list.append(torch.full((k_shot,), class_idx, dtype=torch.long, device=y.device))
#            
#            query_x_list.append(class_samples[query_indices])
#            query_y_list.append(torch.full((q_query,), class_idx, dtype=torch.long, device=y.device))
#        
#        # Concatenate all support and query samples
#        support_x = torch.cat(support_x_list, dim=0)
#        support_y = torch.cat(support_y_list, dim=0)
#        query_x = torch.cat(query_x_list, dim=0)
#        query_y = torch.cat(query_y_list, dim=0)
#        
#        return support_x, support_y, query_x, query_y
#
    # def _compute_metric_friendly_loss(self, x, y, use_contrastive=True, use_prototypical=True, - COMMENTED OUT
#                                    contrastive_weight=0.1, prototypical_weight=0.1):
#        """
#        Compute combined metric-friendly loss with CE, contrastive, and prototypical terms.
#        
#        Args:
#            x: [N, ...] input batch
#            y: [N] labels
#            use_contrastive: whether to add supervised contrastive loss
#            use_prototypical: whether to add prototypical loss (if episodic batch possible)
#            contrastive_weight: weight for contrastive loss term
#            prototypical_weight: weight for prototypical loss term
#            
#        Returns:
#            total_loss: combined loss
#            loss_components: dict with individual loss components
#        """
#        device = x.device
#        
#        # Forward pass to get features and logits
#        with amp.autocast('cuda', enabled=self.use_amp):
#            features = self.model.forward_features(x)  # [N, D] pre-classifier features
#            logits = self.model.classifier(features)   # [N, C] classification logits
#        
#        # Standard cross-entropy loss
#        # Ensure labels are LongTensor for CrossEntropyLoss
#        y = y.to(device=logits.device, dtype=torch.long)
#        
#        if self.params.downstream_dataset == 'ISRUC':
#            ce_loss = self.criterion(logits.float().transpose(1, 2), y)
#        else:
#            ce_loss = self.criterion(logits.float(), y)
#        
#        loss_components = {'ce_loss': ce_loss.item()}
#        total_loss = ce_loss
#        
#        # Add supervised contrastive loss
#        if use_contrastive and features.size(0) > 1:
#            features_normalized = _l2norm(features, dim=-1)
#            contrastive_loss = self._compute_supervised_contrastive_loss(features_normalized, y)
#            total_loss = total_loss + contrastive_weight * contrastive_loss
#            loss_components['contrastive_loss'] = contrastive_loss.item()
#        
#        # Add prototypical loss (episodic)
#        if use_prototypical:
#            # Try to create episodic batch
#            batch_data = (x, y) if not hasattr(x, '__len__') or len(x) == 2 else (x, y, None)
#            episode = self._create_episodic_batch(batch_data, n_way=min(5, len(torch.unique(y))), 
#                                                k_shot=3, q_query=5)
#            if episode is not None:
#                support_x, support_y, query_x, query_y = episode
#                
#                with amp.autocast('cuda', enabled=self.use_amp):
#                    support_features = self.model.forward_features(support_x)
#                    query_features = self.model.forward_features(query_x)
#                
#                support_features = _l2norm(support_features, dim=-1)
#                query_features = _l2norm(query_features, dim=-1)
#                
#                proto_loss = self._compute_prototypical_loss(
#                    support_features, support_y, query_features, query_y, 
#                    len(torch.unique(support_y))
#                )
#                
#                total_loss = total_loss + prototypical_weight * proto_loss
#                loss_components['prototypical_loss'] = proto_loss.item()
#        
#        return total_loss, loss_components
#
    pass

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

    #@torch.no_grad()
    # CLAUDE-ENHANCEMENT: Upgraded _proto_eval_Ks with support for all ICL modes
    # def _proto_eval_Ks(self, loader, K_list, split_name="test"): - COMMENTED OUT
#        """
#        CLAUDE-ENHANCED: Build per-subject support/query splits with:
#        (a) encoder features, (b) central support selection, (c) subject centering + 
#        shrinkage to global prototypes, (d) optional τ autotune, (e) low-confidence 
#        fallback to baseline to strengthen Proto ICL and avoid Δκ<0.
#        """
#        # CLAUDE-ENHANCEMENT: Local imports and helper functions
#        from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
#        import numpy as np
#        import torch
#
#        device = self.device
#        num_classes = getattr(self.params, 'num_of_classes', 5)
#        
#        # Helper function for optional temperature auto-tuning
    #     pass  # [LARGE METHOD BODY COMMENTED OUT]

        #def _choose_tau(zs, ys, protos, grid=(0.05, 0.1, 0.2, 0.5)):
            #if len(zs) < 4: return self.proto_temp
            #best_tau, best_acc = grid[0], -1
            #for tau in grid:
                #correct = 0
                #for i in range(len(zs)):
                    #m = torch.ones(len(zs), dtype=torch.bool)
                    #m[i] = False
                    #p_loo = _compute_prototypes(zs[m].to(device), ys[m].to(device), num_classes)
                    #pr = _proto_logits(zs[i:i+1].to(device), p_loo, tau).argmax(-1).item()
                    #correct += int(pr == ys[i].item())
                #acc = correct / len(zs)
                #if acc > best_acc:
                    #best_acc, best_tau = acc, tau
            #return best_tau

        # 1) Collect features, labels, subject ids, and baseline predictions
        #all_feats, all_labels, all_sids = [], [], []
        #baseline_preds_all = []

        #for batch in loader:
            #if isinstance(batch, (list, tuple)) and len(batch) == 3:
                #xb, yb, sidb = batch
            #else:
                #xb, yb = batch
                #sidb = ['UNK'] * len(yb)

            #xb = xb.to(device)

            # CLAUDE-ENHANCEMENT: Updated AMP usage for consistency
            # features + baseline logits
            #with amp.autocast('cuda', enabled=self.use_amp):
                #feats     = self._extract_features(xb)   # prefer encoder features if available
                #logits_bl = self.model(xb)               # baseline logits
                
            # CLAUDE-COMMENTED-OUT: Old AMP usage
            # # features + baseline logits
            # with (autocast('cuda') if self.use_amp else torch.cuda.amp.autocast(enabled=False)):
            #     feats     = self._extract_features(xb)   # prefer encoder features if available
            #     logits_bl = self.model(xb)               # baseline logits

            #all_feats.append(feats.detach().float().cpu())
            #all_labels.append(yb.detach().cpu())
            #all_sids.extend(list(sidb))
            #baseline_preds_all.append(logits_bl.argmax(-1).detach().cpu())

        #feats = torch.cat(all_feats, dim=0)         # [N, D] (CPU)
        #labels = torch.cat(all_labels, dim=0)       # [N]
        #sids = np.array(all_sids)
        #baseline_preds_all = torch.cat(baseline_preds_all, dim=0)  # [N]

        # 2) Index by subject
        #subj_to_idx = {}
        #for i, s in enumerate(sids):
            #subj_to_idx.setdefault(s, []).append(i)

        #results = {}
        #for K in K_list:
            #if K == 0:
                # K=0 is your baseline (already printed elsewhere). We store None for consistency.
                #results[K] = None
                #continue

            # accumulators for paired comparison (same query subset)
            #proto_preds_all, proto_gts_all = [], []
            #base_preds_all,  base_gts_all  = [], []

            #for s, idxs in subj_to_idx.items():
                # CLAUDE-ENHANCEMENT: Enhanced per-subject processing with all features
                #if 'UNK' in sids:
                    #print(f"⚠️ Warning: Some subject IDs are 'UNK' - this may affect per-subject prototyping quality")
                
                #idxs = np.array(idxs)
                #z = feats[idxs]    # [Ns, D] CPU features
                #y = labels[idxs]   # [Ns] CPU labels

                # Apply subject alignment (centering)
                #z_centered, subject_mean = self._apply_subject_alignment(z)

                # CLAUDE-ENHANCEMENT: Enhanced central support selection with global fallback
                #support_mask = torch.zeros(len(idxs), dtype=torch.bool)
                #y_np = y.numpy()
                
                #for c in range(num_classes):
                    #cls_idx = np.where(y_np == c)[0]
                    #if cls_idx.size == 0: 
                        #continue
                    #z_cls = z_centered[cls_idx]  # Use centered features
                    #selected_idxs, needs_fallback = self._select_support_idxs_central(z_cls, K, class_id=c)
                    
                    #if len(selected_idxs) > 0:
                        #sel = torch.as_tensor(cls_idx)[selected_idxs].long()
                        #support_mask[sel] = True

                #query_mask = ~support_mask
                #if query_mask.sum().item() == 0 or support_mask.sum().item() == 0:
                    #continue

                #z_sup, z_que = z_centered[support_mask], z_centered[~support_mask]
                #y_sup, y_que = y[support_mask], y[~support_mask]

                # Move to device for computation
                #z_sup_device = z_sup.to(device)
                #z_que_device = z_que.to(device)
                #y_sup_device = y_sup.to(device)

                # CLAUDE-ENHANCEMENT: Compute shrunk prototypes with global regularization
                #prototypes = self._compute_shrunk_prototypes(
                    #z_sup_device, y_sup_device, num_classes, subject_mean.to(device), lambda_shrinkage=5.0
                #)

                # CLAUDE-ENHANCEMENT: Tune temperature using LOO validation
                #optimal_temp = self._tune_temperature_loo(
                    #_l2norm(z_sup_device, dim=-1), y_sup_device, prototypes
                #)

                # CLAUDE-ENHANCEMENT: Compute predictions with confidence-based fallback
                #z_que_normalized = _l2norm(z_que_device, dim=-1)
                #confidences, proto_preds = self._compute_prediction_confidence(
                    #z_que_normalized, prototypes, optimal_temp
                #)
                
                # Map query rows back to GLOBAL indices for baseline preds
                #query_global_idxs = idxs[query_mask.numpy()]
                #baseline_q_preds = baseline_preds_all[torch.as_tensor(query_global_idxs)]
                
                # Apply confidence-based fallback
                #final_preds, fallback_mask = self._apply_confidence_fallback(
                    #proto_preds, baseline_q_preds, confidences, confidence_threshold=0.02
                #)

                # Accumulate final predictions (with fallback applied)
                #proto_preds_all.extend(final_preds.cpu().tolist())
                #proto_gts_all.extend(y_que.tolist())
                #base_preds_all.extend(baseline_q_preds.numpy().tolist())
                #base_gts_all.extend(y_que.tolist())

                # Optional: Print fallback statistics for debugging
                #if len(fallback_mask) > 0:
                    #fallback_rate = fallback_mask.sum().item() / len(fallback_mask)
                    #if fallback_rate > 0.3:  # Log if high fallback rate
                        #print(f"  Subject {s}: {fallback_rate:.1%} queries used baseline fallback")

            #if len(proto_gts_all) == 0:
                #print(f"[{split_name}] K={K}: no valid subjects for proto eval (skipping).")
                #continue

            # CLAUDE-ENHANCEMENT: Apply temporal smoothing if enabled
            #if self.use_temporal_smoothing and len(proto_preds_all) > self.temporal_smoothing_window:
                # Apply temporal smoothing to prototypical predictions
                #proto_preds_smoothed = self._apply_full_temporal_smoothing(
                    #np.array(proto_preds_all), 
                    #subject_ids=None,  # Could extract subject IDs if available
                    #enable_median=True, 
                    #enable_hmm=True,
                    #median_window=self.temporal_smoothing_window,
                    #hmm_strength=0.6
                #)
                #proto_preds_all = proto_preds_smoothed.tolist()
                #print(f"[{split_name}] K={K}: Applied temporal smoothing (window={self.temporal_smoothing_window})")

            # 3) Compute paired metrics (same query subset)
            #acc_bal_proto = balanced_accuracy_score(proto_gts_all, proto_preds_all)
            #kappa_proto   = cohen_kappa_score(proto_gts_all, proto_preds_all)
            #f1_w_proto    = f1_score(proto_gts_all, proto_preds_all, average='weighted')

            #acc_bal_base  = balanced_accuracy_score(base_gts_all, base_preds_all)
            #kappa_base    = cohen_kappa_score(base_gts_all, base_preds_all)
            #f1_w_base     = f1_score(base_gts_all, base_preds_all, average='weighted')

            #print(
                #f"[{split_name}] K={K} | "
                #f"BASE(acc_bal={acc_bal_base:.4f}, κ={kappa_base:.4f}, f1_w={f1_w_base:.4f})  "
                #f"→ PROTO(acc_bal={acc_bal_proto:.4f}, κ={kappa_proto:.4f}, f1_w={f1_w_proto:.4f})  "
                #f"Δκ={kappa_proto - kappa_base:+.4f}"
            #)

            #results[K] = dict(
                #base_acc_bal=acc_bal_base, base_kappa=kappa_base, base_f1_w=f1_w_base,
                #proto_acc_bal=acc_bal_proto, proto_kappa=kappa_proto, proto_f1_w=f1_w_proto,
            #)

        #return results

    #@torch.no_grad()
    # def _icl_eval_Ks(self, loader, K_list, split_name="test"): - COMMENTED OUT
#        """
#        Evaluation method for CNP and Set Transformer ICL modes.
#        
#        This method performs per-subject support/query splits and evaluates
#        the learned ICL head performance compared to baseline.
#        """
#        if self.icl_head is None:
#            return {}
#        
#        from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
#        import numpy as np
#        import torch
#
#        device = self.device
#        num_classes = getattr(self.params, 'num_of_classes', 5)
#        
#        # 1) Collect features, labels, subject ids, and baseline predictions
#        all_feats, all_labels, all_sids = [], [], []
#        baseline_preds_all = []
#
#        for batch in loader:
#            if isinstance(batch, (list, tuple)) and len(batch) == 3:
#                xb, yb, sidb = batch
#            else:
#                xb, yb = batch
#                sidb = ['UNK'] * len(yb)
#
#            xb = xb.to(device)
#
#            with amp.autocast('cuda', enabled=self.use_amp):
#                feats = self._extract_features(xb)   # Extract features for ICL
#                logits_bl = self.model(xb)          # Baseline logits
#
#            all_feats.append(feats.detach().float().cpu())
#            all_labels.append(yb.detach().cpu())
#            all_sids.extend(list(sidb))
#            baseline_preds_all.append(logits_bl.argmax(-1).detach().cpu())
#
#        feats = torch.cat(all_feats, dim=0)         # [N, D] (CPU)
#        labels = torch.cat(all_labels, dim=0)       # [N]
#        sids = np.array(all_sids)
#        baseline_preds_all = torch.cat(baseline_preds_all, dim=0)  # [N]
#
#        # 2) Index by subject
#        subj_to_idx = {}
#        for i, s in enumerate(sids):
#            subj_to_idx.setdefault(s, []).append(i)
#
#        results = {}
#        for K in K_list:
#            if K == 0:
#                results[K] = None
#                continue
#
#            # Accumulators for paired comparison
#            icl_preds_all, icl_gts_all = [], []
#            base_preds_all, base_gts_all = [], []
#
#            for s, idxs in subj_to_idx.items():
#                if 'UNK' in sids:
#                    print(f"⚠️ Warning: Some subject IDs are 'UNK' - this may affect per-subject ICL quality")
#                
#                idxs = np.array(idxs)
#                z = feats[idxs]    # [Ns, D] CPU features
#                y = labels[idxs]   # [Ns] CPU labels
#
#                # Create support/query split for this subject
#                support_mask = torch.zeros(len(idxs), dtype=torch.bool)
#                y_np = y.numpy()
#                
#                for c in range(num_classes):
#                    cls_idx = np.where(y_np == c)[0]
#                    if cls_idx.size == 0: 
#                        continue
#                    
#                    # Use central selection for better support quality
#                    selected_count = min(K, len(cls_idx))
#                    if selected_count > 0:
#                        z_cls = z[cls_idx]  # Features for this class
#                        selected_local_idxs, _ = self._select_support_idxs_central(z_cls, selected_count, class_id=c)
#                        selected_idxs = cls_idx[selected_local_idxs.numpy()]
#                        support_mask[selected_idxs] = True
#
#                query_mask = ~support_mask
#                if query_mask.sum().item() == 0 or support_mask.sum().item() == 0:
#                    continue
#
#                z_sup, z_que = z[support_mask], z[query_mask]
#                y_sup, y_que = y[support_mask], y[query_mask]
#
#                # Move to device for computation
#                z_sup_device = z_sup.to(device)
#                z_que_device = z_que.to(device)
#                y_sup_device = y_sup.to(device)
#
#                # ICL head forward pass
#                with amp.autocast('cuda', enabled=self.use_amp):
#                    # Use original features without normalization
#                    icl_logits = self.icl_head(z_sup_device, y_sup_device, z_que_device)
#                
#                icl_preds = icl_logits.argmax(dim=1).cpu()
#
#                # Map query rows back to global indices for baseline preds
#                query_global_idxs = idxs[query_mask.numpy()]
#                baseline_q_preds = baseline_preds_all[torch.as_tensor(query_global_idxs)]
#
#                # Accumulate predictions
#                icl_preds_all.extend(icl_preds.tolist())
#                icl_gts_all.extend(y_que.tolist())
#                base_preds_all.extend(baseline_q_preds.numpy().tolist())
#                base_gts_all.extend(y_que.tolist())
#
#            if len(icl_gts_all) == 0:
#                print(f"[{split_name}] K={K}: no valid subjects for ICL eval (skipping).")
#                continue
#
#            # 3) Compute paired metrics
#            acc_bal_icl = balanced_accuracy_score(icl_gts_all, icl_preds_all)
#            kappa_icl = cohen_kappa_score(icl_gts_all, icl_preds_all)
#            f1_w_icl = f1_score(icl_gts_all, icl_preds_all, average='weighted')
#
#            acc_bal_base = balanced_accuracy_score(base_gts_all, base_preds_all)
#            kappa_base = cohen_kappa_score(base_gts_all, base_preds_all)
#            f1_w_base = f1_score(base_gts_all, base_preds_all, average='weighted')
#
#            print(
#                f"[{split_name}] K={K} ({self.icl_mode.upper()}) | "
#                f"BASE(acc_bal={acc_bal_base:.4f}, κ={kappa_base:.4f}, f1_w={f1_w_base:.4f})  "
#                f"→ ICL(acc_bal={acc_bal_icl:.4f}, κ={kappa_icl:.4f}, f1_w={f1_w_icl:.4f})  "
#                f"Δκ={kappa_icl - kappa_base:+.4f}"
#            )
#
#            results[K] = dict(
#                base_acc_bal=acc_bal_base, base_kappa=kappa_base, base_f1_w=f1_w_base,
#                icl_acc_bal=acc_bal_icl, icl_kappa=kappa_icl, icl_f1_w=f1_w_icl,
#            )
#
#        return results
#        
#        # CLAUDE-COMMENTED-OUT: Original _proto_eval_Ks implementation (preserved for fallback)
#        # The above enhanced version includes:
#        # - Encoder features (via forward_features)
#        # - Central support selection (not random)
#        # - Subject centering + shrinkage to global prototypes
#        # - Optional temperature auto-tuning
#        # - Low-confidence fallback to baseline predictions
#
#
    #pass

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
                
                # if self.icl_mode in ['cnp', 'set'] and self.icl_head is not None: - COMMENTED OUT
                #     # Standard classification loss
                #     with amp.autocast('cuda', enabled=self.use_amp):
                #         pred = self.model(x)
                #     
                #     # Ensure labels are LongTensor for CrossEntropyLoss  
                #     y = y.to(device=pred.device, dtype=torch.long)
                #     
                #     if self.params.downstream_dataset == 'ISRUC':
                #         ce_loss = self.criterion(pred.float().transpose(1, 2), y)
                #     else:
                #         ce_loss = self.criterion(pred.float(), y)
                #     total_loss = total_loss + ce_loss
                #     loss_components['ce_loss'] = ce_loss.item()
                #     
                #     # CLAUDE-ENHANCEMENT: Strengthened episodic ICL training
                #     # Increase episode difficulty and balance
                #     unique_classes_in_batch = len(torch.unique(y))
                #     episode_data = self._create_episodic_batch_cnp(
                #         (x, y, sid) if sid is not None else (x, y),
                #         n_way=min(4, unique_classes_in_batch),  # Increased from 3 to 4
                #         k_shot=3,                               # Increased from 2 to 3  
                #         q_query=5                               # Increased from 3 to 5
                #     )
                #     
                #     if episode_data is not None:
                #         support_x, support_y, query_x, query_y = episode_data
                #         icl_loss = self._compute_icl_loss(support_x, support_y, query_x, query_y)
                #         
                #         # Weight ICL loss (conservative to avoid overpowering CE loss)
                #         total_loss = total_loss + self.icl_loss_weight * icl_loss
                #         loss_components['icl_loss'] = icl_loss.item()
                #         
                #     loss = total_loss
                #     
                #     # Log loss components periodically
                #     if batch_idx % 100 == 0 and loss_components:
                #         comp_str = ', '.join([f"{k}={v:.4f}" for k, v in loss_components.items()])
                #         print(f"Epoch {epoch}, Batch {batch_idx}: {comp_str}")
                
                # elif self.use_metric_friendly_training: - COMMENTED OUT
                #     # Use combined loss with contrastive and prototypical terms
                #     loss, loss_components = self._compute_metric_friendly_loss(
                #         x, y, 
                #         use_contrastive=True, 
                #         use_prototypical=True,
                #         contrastive_weight=self.contrastive_weight,
                #         prototypical_weight=self.prototypical_weight
                #     )
                #     
                #     # Log loss components periodically
                #     if batch_idx % 100 == 0:  # Log every 100 batches
                #         comp_str = ', '.join([f"{k}={v:.4f}" for k, v in loss_components.items()])
                #         print(f"Epoch {epoch}, Batch {batch_idx}: {comp_str}")
                        
                # Standard forward pass (default path since ICL/proto methods are commented out)
                with amp.autocast('cuda', enabled=self.use_amp):
                    pred = self.model(x)
                    
                    # Ensure labels are LongTensor for CrossEntropyLoss  
                    y = y.to(device=pred.device, dtype=torch.long)
                    
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = self.criterion(pred.float().transpose(1, 2), y)
                    else:
                        loss = self.criterion(pred.float(), y)
                        
                # CLAUDE-COMMENTED-OUT: Old AMP usage
                # # Forward pass with mixed precision
                # if self.use_amp:
                #     with autocast('cuda'):
                #         pred = self.model(x)
                #         if self.params.downstream_dataset == 'ISRUC':
                #             loss = self.criterion(pred.transpose(1, 2), y)
                #         else:
                #             loss = self.criterion(pred, y)
                # else:
                #     pred = self.model(x)
                #     if self.params.downstream_dataset == 'ISRUC':
                #         loss = self.criterion(pred.transpose(1, 2), y)
                #     else:
                #         loss = self.criterion(pred, y)
                # Scale loss for gradient accumulation (commented out - no accumulation)
                # loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                losses.append(loss.data.cpu().numpy())  # No accumulation scaling
                
                # Update weights every batch (no accumulation)
                # if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
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
                            
                        # CLAUDE-COMMENTED-OUT: Old AMP validation
                        # if self.use_amp:
                        #     with autocast('cuda'):
                        #         pred_val = self.model(x_val)
                        #         if self.params.downstream_dataset == 'ISRUC':
                        #             val_loss = self.criterion(pred_val.transpose(1, 2), y_val)
                        #         else:
                        #             val_loss = self.criterion(pred_val, y_val)
                        # else:
                        #     pred_val = self.model(x_val)
                        #     if self.params.downstream_dataset == 'ISRUC':
                        #         val_loss = self.criterion(pred_val.transpose(1, 2), y_val)
                        #     else:
                        #         val_loss = self.criterion(pred_val, y_val)

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
            
            # ==== Optional: K-shot ICL Evaluation ==== - COMMENTED OUT
            # if self.icl_mode != 'none' and any(k > 0 for k in self.icl_eval_Ks):
            #     if self.icl_mode == 'proto':
            #         print("\n🔎 Running prototypical (in-context) K-sweep on TEST split...")
            #         icl_results = self._proto_eval_Ks(self.data_loader['test'], self.icl_eval_Ks, split_name="test")
            #         metric_prefix = "proto_test"
            #         result_key_icl = "proto_acc_bal"  # for proto mode
            #         result_key_kappa_icl = "proto_kappa"
            #         result_key_f1_icl = "proto_f1_w"
            #     elif self.icl_mode in ['cnp', 'set']:
            #         print(f"\n🔎 Running {self.icl_mode.upper()} ICL K-sweep on TEST split...")
            #         icl_results = self._icl_eval_Ks(self.data_loader['test'], self.icl_eval_Ks, split_name="test")
            #         metric_prefix = f"{self.icl_mode}_test"
            #         result_key_icl = "icl_acc_bal"  # for cnp/set modes
            #         result_key_kappa_icl = "icl_kappa"
            #         result_key_f1_icl = "icl_f1_w"
            #     else:
            #         icl_results = {}
            #         
            #     # Log to W&B if available
            #     for K, res in (icl_results or {}).items():
            #         if res is None:
            #             continue
            #         self.log_wandb_metrics({
            #             f"{metric_prefix}_accbal_base_K{K}":  res['base_acc_bal'],
            #             f"{metric_prefix}_accbal_icl_K{K}":   res[result_key_icl],
            #             f"{metric_prefix}_kappa_base_K{K}":   res['base_kappa'],
            #             f"{metric_prefix}_kappa_icl_K{K}":    res[result_key_kappa_icl],
            #             f"{metric_prefix}_f1w_base_K{K}":     res['base_f1_w'],
            #             f"{metric_prefix}_f1w_icl_K{K}":      res[result_key_f1_icl],
            #             f"{metric_prefix}_delta_kappa_K{K}":  res[result_key_kappa_icl] - res['base_kappa'],
            #         })


        self.close_wandb()
        
        # Cleanup memory after training
        self.memory_manager.cleanup_between_trials(trial_name)

        return kappa_best




# def _l2norm(x, dim=-1, eps=1e-8): - COMMENTED OUT
    pass

#     return x / (x.norm(dim=dim, keepdim=True) + eps)

# @torch.no_grad() - COMMENTED OUT
# def _compute_prototypes(support_feats, support_labels, num_classes):
#     # support_feats: [Ns, D]; support_labels: [Ns]
#     protos = []
#     for k in range(num_classes):
#         mask = (support_labels == k)
#         if mask.any():
#             proto = support_feats[mask].mean(0)
#         else:
#             proto = torch.zeros(support_feats.size(1), device=support_feats.device)
#         protos.append(proto)
#     protos = torch.stack(protos, 0)  # [Kc, D]
#     return _l2norm(protos, dim=-1)

# def _proto_logits(query_feats, prototypes, temperature=0.1): - COMMENTED OUT
    pass

#     qn = _l2norm(query_feats, dim=-1)      # [Nq, D]
#     sims = qn @ prototypes.t()             # [Nq, Kc]
#     return sims / max(1e-8, temperature)

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