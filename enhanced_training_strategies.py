"""
Advanced Training Strategies for Enhanced Finetuning

This module implements cutting-edge training techniques for optimal
EEG model finetuning performance:

1. Layer-wise Learning Rate Decay (LLRD)
2. Gradient Accumulation with Mixed Precision
3. Progressive Layer Unfreezing
4. Advanced Data Augmentation
5. Curriculum Learning
6. Model Compression & Quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import math

class LayerwiseLearningRateDecay:
    """
    Implements layer-wise learning rate decay for better finetuning.
    Lower layers (closer to input) get smaller learning rates.
    """
    
    def __init__(self, model: nn.Module, base_lr: float, decay_rate: float = 0.9):
        self.model = model
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.logger = logging.getLogger(__name__)
    
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups with layer-wise learning rates"""
        
        param_groups = []
        
        # Backbone layers (deeper = higher LR)
        if hasattr(self.model, 'backbone'):
            backbone_layers = []
            
            # Get transformer layers
            if hasattr(self.model.backbone, 'transformer_encoder'):
                layers = self.model.backbone.transformer_encoder.layers
                n_layers = len(layers)
                
                for i, layer in enumerate(layers):
                    # Higher layers get higher learning rates
                    layer_lr = self.base_lr * (self.decay_rate ** (n_layers - i - 1))
                    param_groups.append({
                        'params': list(layer.parameters()),
                        'lr': layer_lr,
                        'name': f'backbone_layer_{i}'
                    })
                    self.logger.info(f"Layer {i}: LR = {layer_lr:.2e}")
            
            # Other backbone components
            other_backbone_params = []
            for name, param in self.model.backbone.named_parameters():
                if 'transformer_encoder.layers' not in name:
                    other_backbone_params.append(param)
            
            if other_backbone_params:
                param_groups.append({
                    'params': other_backbone_params,
                    'lr': self.base_lr * (self.decay_rate ** n_layers),
                    'name': 'backbone_other'
                })
        
        # Head layers (highest learning rate)
        head_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' not in name:
                head_params.append(param)
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': self.base_lr,
                'name': 'head'
            })
        
        return param_groups

class ProgressiveUnfreezing:
    """
    Progressive layer unfreezing for stable finetuning.
    Starts with frozen backbone, gradually unfreezes layers.
    """
    
    def __init__(self, model: nn.Module, unfreeze_schedule: Dict[int, List[str]]):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.logger = logging.getLogger(__name__)
        
        # Initially freeze all backbone parameters
        self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self.logger.info("🧊 Backbone frozen")
    
    def update_frozen_layers(self, epoch: int):
        """Update which layers are frozen based on epoch"""
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            
            for layer_name in layers_to_unfreeze:
                self.unfreeze_layer(layer_name)
            
            self.logger.info(f"🔓 Epoch {epoch}: Unfroze layers {layers_to_unfreeze}")
    
    def unfreeze_layer(self, layer_name: str):
        """Unfreeze specific layer by name pattern"""
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                unfrozen_count += 1
        
        self.logger.info(f"Unfroze {unfrozen_count} parameters in {layer_name}")

class EEGDataAugmentation:
    """
    Advanced data augmentation specifically designed for EEG signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def temporal_masking(self, x: torch.Tensor, mask_prob: float = 0.1) -> torch.Tensor:
        """Randomly mask temporal segments"""
        if not self.training or torch.rand(1) > mask_prob:
            return x
        
        batch_size, seq_len, features = x.shape
        mask_length = int(seq_len * 0.1)  # Mask 10% of sequence
        
        for i in range(batch_size):
            start_idx = torch.randint(0, seq_len - mask_length, (1,))
            x[i, start_idx:start_idx + mask_length] = 0
        
        return x
    
    def frequency_masking(self, x: torch.Tensor, mask_prob: float = 0.1) -> torch.Tensor:
        """Randomly mask frequency bands"""
        if not self.training or torch.rand(1) > mask_prob:
            return x
        
        batch_size, seq_len, features = x.shape
        mask_channels = int(features * 0.1)  # Mask 10% of channels
        
        for i in range(batch_size):
            start_ch = torch.randint(0, features - mask_channels, (1,))
            x[i, :, start_ch:start_ch + mask_channels] = 0
        
        return x
    
    def mixup(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixup augmentation for EEG data"""
        if not self.training:
            return x, y
        
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        
        indices = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        # For classification, mix labels
        if y.dim() == 1:  # Class labels
            y_a, y_b = y, y[indices]
            return mixed_x, (y_a, y_b, lam)
        else:
            mixed_y = lam * y + (1 - lam) * y[indices]
            return mixed_x, mixed_y
    
    def gaussian_noise(self, x: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise"""
        if not self.training:
            return x
        
        noise = torch.randn_like(x) * noise_level
        return x + noise

class CurriculumLearning:
    """
    Curriculum learning: start with easy samples, gradually add harder ones
    """
    
    def __init__(self, difficulty_scores: np.ndarray, curriculum_epochs: int = 20):
        self.difficulty_scores = difficulty_scores
        self.curriculum_epochs = curriculum_epochs
        self.logger = logging.getLogger(__name__)
    
    def get_curriculum_indices(self, epoch: int, total_samples: int) -> np.ndarray:
        """Get sample indices based on curriculum schedule"""
        
        if epoch >= self.curriculum_epochs:
            # Use all samples after curriculum period
            return np.arange(total_samples)
        
        # Gradually increase difficulty
        progress = epoch / self.curriculum_epochs
        
        # Start with easiest 20%, gradually include all samples
        easy_fraction = 0.2 + 0.8 * progress
        n_samples = int(total_samples * easy_fraction)
        
        # Sort by difficulty and take the easiest samples
        sorted_indices = np.argsort(self.difficulty_scores)
        curriculum_indices = sorted_indices[:n_samples]
        
        self.logger.info(f"Epoch {epoch}: Using {n_samples}/{total_samples} samples ({easy_fraction:.1%})")
        
        return curriculum_indices

class ModelCompression:
    """
    Model compression techniques for deployment
    """
    
    @staticmethod
    def quantize_model(model: nn.Module, calibration_loader) -> nn.Module:
        """Quantize model to INT8 for faster inference"""
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare and calibrate
        prepared_model = torch.quantization.prepare(model, inplace=False)
        
        # Calibration
        with torch.no_grad():
            for data, _ in calibration_loader:
                prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        return quantized_model
    
    @staticmethod
    def prune_model(model: nn.Module, pruning_ratio: float = 0.2) -> nn.Module:
        """Prune model weights for efficiency"""
        import torch.nn.utils.prune as prune
        
        # Global magnitude pruning
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    @staticmethod
    def export_onnx(model: nn.Module, sample_input: torch.Tensor, output_path: str):
        """Export model to ONNX format"""
        model.eval()
        
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

class AdvancedOptimizer:
    """
    Advanced optimization techniques
    """
    
    @staticmethod
    def create_sam_optimizer(model_params, base_optimizer, rho: float = 0.05):
        """Sharpness-Aware Minimization (SAM) optimizer"""
        
        class SAM:
            def __init__(self, params, base_optimizer, rho=0.05):
                self.params = list(params)
                self.base_optimizer = base_optimizer
                self.rho = rho
                
            def first_step(self, zero_grad=False):
                grad_norm = self._grad_norm()
                for group in self.base_optimizer.param_groups:
                    scale = self.rho / (grad_norm + 1e-12)
                    
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        e_w = p.grad * scale
                        p.add_(e_w)  # climb to the local maximum
                        
                if zero_grad:
                    self.zero_grad()
                    
            def second_step(self, zero_grad=False):
                for group in self.base_optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        p.sub_(p.grad)  # go back to the original point
                        
                self.base_optimizer.step()  # do the actual "sharpness-aware" update
                
                if zero_grad:
                    self.zero_grad()
                    
            def _grad_norm(self):
                shared_device = self.params[0].device
                norm = torch.norm(
                    torch.stack([
                        p.grad.norm(dtype=torch.float32).to(shared_device)
                        for p in self.params if p.grad is not None
                    ]),
                    dtype=torch.float32
                )
                return norm
                
            def zero_grad(self):
                self.base_optimizer.zero_grad()
        
        return SAM(model_params, base_optimizer, rho)

class MemoryOptimization:
    """
    Memory optimization techniques
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module):
        """Enable gradient checkpointing to save memory"""
        
        def checkpoint_wrapper(module):
            def forward(*args):
                return torch.utils.checkpoint.checkpoint(module.forward, *args)
            return forward
        
        # Apply to transformer layers
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'transformer_encoder'):
            for layer in model.backbone.transformer_encoder.layers:
                layer.forward = checkpoint_wrapper(layer)
    
    @staticmethod
    def optimize_memory_usage(model: nn.Module, input_size: Tuple[int, ...]):
        """Analyze and optimize memory usage"""
        
        def get_model_memory_usage(model, input_size):
            """Calculate model memory usage"""
            
            # Parameter memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Activation memory (rough estimate)
            dummy_input = torch.randn(1, *input_size)
            
            # Hook to capture activation sizes
            activation_memory = 0
            
            def hook_fn(module, input, output):
                nonlocal activation_memory
                if isinstance(output, torch.Tensor):
                    activation_memory += output.numel() * output.element_size()
                elif isinstance(output, (list, tuple)):
                    for o in output:
                        if isinstance(o, torch.Tensor):
                            activation_memory += o.numel() * o.element_size()
            
            hooks = []
            for module in model.modules():
                hooks.append(module.register_forward_hook(hook_fn))
            
            with torch.no_grad():
                model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return param_memory, activation_memory
        
        param_mem, activation_mem = get_model_memory_usage(model, input_size)
        
        print(f"💾 Memory Analysis:")
        print(f"  Parameters: {param_mem / 1024**2:.1f} MB")
        print(f"  Activations: {activation_mem / 1024**2:.1f} MB")
        print(f"  Total: {(param_mem + activation_mem) / 1024**2:.1f} MB")
        
        return param_mem, activation_mem

# Factory function to create enhanced training components
def create_enhanced_training_components(config: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """Create all enhanced training components based on configuration"""
    
    components = {}
    
    # Layer-wise learning rate decay
    if config.get('use_llrd', False):
        llrd = LayerwiseLearningRateDecay(
            model, 
            base_lr=config.get('learning_rate', 0.0005),
            decay_rate=config.get('llrd_decay_rate', 0.9)
        )
        components['llrd'] = llrd
    
    # Progressive unfreezing
    if config.get('use_progressive_unfreezing', False):
        unfreeze_schedule = config.get('unfreeze_schedule', {
            5: ['head'],
            10: ['backbone.transformer_encoder.layers.11'],
            15: ['backbone.transformer_encoder.layers.10'],
            20: ['backbone.transformer_encoder.layers.9'],
            25: ['backbone']  # Unfreeze all
        })
        progressive_unfreezing = ProgressiveUnfreezing(model, unfreeze_schedule)
        components['progressive_unfreezing'] = progressive_unfreezing
    
    # Data augmentation
    if config.get('use_advanced_augmentation', False):
        augmentation = EEGDataAugmentation(config.get('augmentation', {}))
        components['augmentation'] = augmentation
    
    # Memory optimization
    if config.get('use_gradient_checkpointing', False):
        MemoryOptimization.enable_gradient_checkpointing(model)
    
    return components