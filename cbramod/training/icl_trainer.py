"""
In-Context Learning (ICL) trainer for prototype-based classification.

Implements both standard prototypical networks (proto mode) and 
meta-learning with learnable projections (meta_proto mode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score


class ProtoICLHead(nn.Module):
    """
    Prototype-based ICL head for classification.
    
    Supports two modes:
    - Proto: Fixed identity projection, fixed temperature
    - Meta-proto: Learnable projection and temperature 
    """
    
    def __init__(self, in_dim: int = 512, proj_dim: int = 512, cosine: bool = False, learnable: bool = False, temperature: float = 2.0):
        """
        Args:
            in_dim: Input feature dimension
            proj_dim: Projection dimension
            cosine: Use cosine similarity vs L2 distance
            learnable: Whether to learn projection and temperature (meta_proto mode)
            temperature: Temperature scaling factor for logits
        """
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.cosine = cosine
        self.learnable = learnable
        
        if learnable:
            # Learnable projection for meta-proto mode
            self.projection = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim)
            )
        else:
            # Identity projection for proto mode
            self.projection = nn.Identity() if in_dim == proj_dim else nn.Linear(in_dim, proj_dim, bias=False)
        
        # Initialize temperature (can be modified after initialization)
        self.register_buffer('logit_scale', torch.tensor(temperature))
    
    def set_temperature(self, temperature: float):
        """Update the temperature scaling factor."""
        self.logit_scale.fill_(temperature)
    
    def forward(self, support_feats: torch.Tensor, support_labels: torch.Tensor, 
                query_feats: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Forward pass for prototype-based classification.
        
        Args:
            support_feats: [K, feat_dim] support features
            support_labels: [K] support labels  
            query_feats: [M, feat_dim] query features
            num_classes: Total number of classes in the dataset
            
        Returns:
            logits: [M, num_classes] classification logits
        """
        # Project features
        support_feats = self.projection(support_feats)  # [K, proj_dim]
        query_feats = self.projection(query_feats)  # [M, proj_dim]
        
        # Always L2 normalize for stability
        support_feats = F.normalize(support_feats, dim=-1, eps=1e-8)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-8)
        
        # Build class prototypes
        prototypes = {}
        present_classes = torch.unique(support_labels)
        
        for class_id in present_classes:
            class_mask = support_labels == class_id
            class_feats = support_feats[class_mask]  # [N_class, proj_dim]
            prototypes[int(class_id)] = class_feats.mean(dim=0)  # [proj_dim]
        
        # Compute similarities/distances
        logits = torch.full((query_feats.size(0), num_classes), float('-inf'), 
                           device=query_feats.device, dtype=query_feats.dtype)
        
        for class_id, prototype in prototypes.items():
            if self.cosine:
                # Cosine similarity (features are normalized)
                similarities = torch.mm(query_feats, prototype.unsqueeze(-1)).squeeze(-1)  # [M]
            else:
                # Negative squared L2 distance (more stable than L2)
                distances_sq = torch.sum((query_feats - prototype.unsqueeze(0)) ** 2, dim=1)  # [M]
                similarities = -distances_sq  # [M]
            
            logits[:, class_id] = similarities
        
        # Scale by fixed temperature
        logits = logits * self.logit_scale
        
        # Replace -inf with large negative values for numerical stability
        logits = torch.where(torch.isinf(logits) & (logits < 0), 
                           torch.full_like(logits, -1e6), logits)
        
        return logits


class ICLTrainer:
    """
    Trainer for In-Context Learning with episodic training.
    
    Handles both proto (no training) and meta_proto (episodic training) modes.
    """
    
    def __init__(self, params, model, num_classes: int, proj_dim: int = 512, cosine: bool = False, temperature: float = 2.0):
        """
        Args:
            params: Training parameters
            model: Backbone model with forward_features() method
            num_classes: Number of output classes
            proj_dim: Projection dimension
            cosine: Use cosine similarity vs L2 distance
            temperature: Temperature scaling factor for logits
        """
        self.params = params
        self.model = model
        self.num_classes = num_classes
        self.device = next(model.parameters()).device
        
        # Freeze backbone if requested
        if getattr(params, 'freeze_backbone_for_icl', True):
            print("🔒 Freezing backbone parameters for ICL")
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Create ICL head
        learnable = (params.icl_mode == 'meta_proto')
        self.icl_head = ProtoICLHead(
            in_dim=512,  # Model outputs 512-dim features
            proj_dim=proj_dim,
            cosine=cosine,
            learnable=learnable,
            temperature=temperature
        ).to(self.device)
        
        # Setup optimizer for meta-proto mode
        self.optimizer = None
        if learnable:
            head_lr = getattr(params, 'head_lr', 1e-3)
            self.optimizer = torch.optim.AdamW(
                self.icl_head.parameters(),
                lr=head_lr,
                weight_decay=getattr(params, 'weight_decay', 0.0)
            )
            print(f"📊 Meta-ICL mode: Training ICL head with lr={head_lr}")
        else:
            print("📊 Proto-ICL mode: No parameter updates")
    
    def _encode(self, x: torch.Tensor, train_mode: bool = False) -> torch.Tensor:
        """
        Encode input using backbone model.
        
        Args:
            x: Input tensor [B, ch, 30, 200]
            train_mode: Whether to enable gradients
            
        Returns:
            features: [B, 512] encoded features
        """
        if train_mode and self.params.icl_mode == 'meta_proto':
            # Enable gradients only through ICL head in meta-proto mode
            with torch.set_grad_enabled(True):
                return self.model.forward_features(x)
        else:
            # No gradients in proto mode or during evaluation
            with torch.no_grad():
                return self.model.forward_features(x)
    
    def _episode_step(self, batch: Dict[str, Any], train: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Process a single episode and compute loss + accuracy.
        
        Args:
            batch: Episode batch dictionary
            train: Whether this is a training step
            
        Returns:
            (loss, accuracy) tuple
        """
        # Handle batched episodes (for training) or single episode (for testing)  
        if batch['support_x'].dim() == 5:  # Batched episodes [B, K, ch, 30, 200]
            # Take first episode in batch for now (can extend to handle multiple)
            support_x = batch['support_x'][0]  # [K, ch, 30, 200]
            support_y = batch['support_y'][0]  # [K]
            query_x = batch['query_x'][0]  # [M, ch, 30, 200]
            query_y = batch['query_y'][0]  # [M]
        else:  # Single episode
            support_x = batch['support_x']  # [K, ch, 30, 200]
            support_y = batch['support_y']  # [K]
            query_x = batch['query_x']  # [M, ch, 30, 200]
            query_y = batch['query_y']  # [M]
        
        # Move to device
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        query_x = query_x.to(self.device)
        query_y = query_y.to(self.device)
        
        # Encode support and query
        support_feats = self._encode(support_x, train_mode=train)  # [K, 512]
        query_feats = self._encode(query_x, train_mode=train)  # [M, 512]
        
        # ICL head forward pass
        logits = self.icl_head(support_feats, support_y, query_feats, self.num_classes)  # [M, num_classes]
        
        # Compute loss and accuracy
        loss = F.cross_entropy(logits, query_y)
        
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            accuracy = (pred == query_y).float().mean().item()
        
        return loss, accuracy
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int = 3):
        """
        Train the ICL head with episodic learning (meta_proto mode only).
        
        Args:
            train_loader: Training episode loader
            val_loader: Validation episode loader  
            epochs: Number of training epochs
            patience: Early stopping patience (epochs)
        """
        if self.params.icl_mode != 'meta_proto':
            print("⚠️ Skipping training - proto mode has no learnable parameters")
            return
        
        print(f"🚀 Starting meta-ICL training for {epochs} epochs (patience={patience})")
        
        best_val_kappa = -np.inf
        epochs_without_improvement = 0
        best_state_dict = None
        
        for epoch in range(epochs):
            # Training phase
            self.icl_head.train()
            train_losses, train_accs = [], []
            
            for batch in train_loader:
                loss, acc = self._episode_step(batch, train=True)
                
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.icl_head.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                train_accs.append(acc)
            
            # Validation phase - compute κ for early stopping
            self.icl_head.eval()
            val_losses, val_accs = [], []
            all_val_preds, all_val_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    loss, acc = self._episode_step(batch, train=False)
                    val_losses.append(loss.item())
                    val_accs.append(acc)
                    
                    # Collect predictions for κ calculation
                    support_x = batch['support_x'][0] if batch['support_x'].dim() == 5 else batch['support_x']
                    support_y = batch['support_y'][0] if batch['support_y'].dim() == 2 else batch['support_y']
                    query_x = batch['query_x'][0] if batch['query_x'].dim() == 5 else batch['query_x']
                    query_y = batch['query_y'][0] if batch['query_y'].dim() == 2 else batch['query_y']
                    
                    support_x = support_x.to(self.device)
                    support_y = support_y.to(self.device)
                    query_x = query_x.to(self.device)
                    
                    support_feats = self._encode(support_x)
                    query_feats = self._encode(query_x)
                    logits = self.icl_head(support_feats, support_y, query_feats, self.num_classes)
                    
                    preds = logits.argmax(dim=-1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(query_y.numpy())
            
            # Calculate validation κ
            val_kappa = cohen_kappa_score(all_val_labels, all_val_preds)
            
            # Early stopping check
            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
                epochs_without_improvement = 0
                best_state_dict = self.icl_head.state_dict().copy()
            else:
                epochs_without_improvement += 1
            
            # Log progress
            if epoch % max(1, epochs // 10) == 0 or epochs_without_improvement == 0:
                print(f"Epoch {epoch:3d}: Train Loss={np.mean(train_losses):.4f} Acc={np.mean(train_accs):.3f} | "
                      f"Val Loss={np.mean(val_losses):.4f} Acc={np.mean(val_accs):.3f} κ={val_kappa:.4f} {'🎯' if epochs_without_improvement == 0 else ''}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1} (patience={patience})")
                break
        
        # Restore best model
        if best_state_dict is not None:
            self.icl_head.load_state_dict(best_state_dict)
            print(f"📦 Restored best model (κ={best_val_kappa:.4f})")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate ICL model on test episodes.
        
        Args:
            test_loader: Test episode loader (batch_size=1)
            
        Returns:
            Dictionary with kappa, accuracy, and macro_f1 scores
        """
        print("🧪 Evaluating ICL model...")
        
        self.icl_head.eval()
        all_preds, all_labels = [], []
        episode_accs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Single episode evaluation
                loss, acc = self._episode_step(batch, train=False)
                episode_accs.append(acc)
                
                # Get predictions for metrics
                support_x = batch['support_x'].to(self.device)
                support_y = batch['support_y'].to(self.device)
                query_x = batch['query_x'].to(self.device)
                query_y = batch['query_y'].to(self.device)
                
                support_feats = self._encode(support_x)
                query_feats = self._encode(query_x)
                logits = self.icl_head(support_feats, support_y, query_feats, self.num_classes)
                
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(query_y.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        kappa = cohen_kappa_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        results = {
            'kappa': kappa,
            'acc': accuracy, 
            'macro_f1': macro_f1,
            'episode_acc_mean': np.mean(episode_accs),
            'episode_acc_std': np.std(episode_accs),
            'num_episodes': len(episode_accs)
        }
        
        print(f"📊 ICL Results:")
        print(f"   Kappa: {kappa:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")  
        print(f"   Macro F1: {macro_f1:.4f}")
        print(f"   Episodes: {len(episode_accs)} (acc: {np.mean(episode_accs):.3f}±{np.std(episode_accs):.3f})")
        
        return results