# Two-Phase Training for CBraMod

A simple implementation of progressive unfreezing for fine-tuning the CBraMod foundation model on sleep staging tasks.

## Overview

Two-phase training follows the approach recommended in the literature for fine-tuning foundation models:

1. **Phase 1**: Train only the classification head with the transformer backbone frozen
2. **Phase 2**: Unfreeze the backbone and continue training with a lower learning rate

This approach helps stabilize training and allows the model to adapt its features to the sleep staging task while avoiding catastrophic forgetting.

## Quick Start

### Basic Usage
```bash
python cbramod/training/finetuning/finetune_main.py \
    --two_phase_training True \
    --epochs 15
```

### Custom Configuration
```bash
python cbramod/training/finetuning/finetune_main.py \
    --two_phase_training True \
    --phase1_epochs 5 \
    --head_lr 2e-3 \
    --backbone_lr 5e-6 \
    --epochs 20
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--two_phase_training` | `False` | Enable two-phase training |
| `--phase1_epochs` | `3` | Number of epochs for phase 1 (frozen backbone) |
| `--head_lr` | `1e-3` | Learning rate for head/classifier |
| `--backbone_lr` | `1e-5` | Learning rate for backbone in phase 2 |

## How It Works

### Phase 1 (Frozen Backbone)
- Only trains the classification head (`head`, `sequence_encoder`, `classifier`)
- Backbone transformer parameters are frozen (`requires_grad=False`)
- Uses `head_lr` for all trainable parameters
- Initializes head weights to reasonable values

### Phase 2 (Unfrozen Backbone)
- Unfreezes all backbone parameters
- Uses different learning rates: `head_lr` for head, `backbone_lr` for backbone
- Allows backbone to adapt features to the specific task

### Training Flow Example
```
Epochs 0-2:  Phase 1 - Train head (lr=1e-3), backbone frozen
Epoch 3:     🔄 Switch to Phase 2 - Unfreeze backbone
Epochs 3-14: Phase 2 - Train head (lr=1e-3) + backbone (lr=1e-5)
```

## Implementation Details

### Model Methods Added
- `freeze_backbone()`: Freeze backbone parameters
- `unfreeze_backbone()`: Unfreeze backbone parameters  
- `set_progressive_unfreezing_mode(phase)`: Set training phase
- `get_param_groups(head_lr, backbone_lr)`: Get parameter groups with different LRs

### Trainer Integration
- Automatically switches phases based on epoch count
- Recreates optimizer with new parameter groups when switching
- Logs parameter counts and learning rates for each phase

## Best Practices

### Learning Rate Selection
- **Head LR**: 5-10x higher than backbone LR (e.g., 1e-3)
- **Backbone LR**: Small to prevent catastrophic forgetting (e.g., 1e-5)
- Start with defaults and tune based on validation performance

### Phase Duration
- **Phase 1**: 15-25% of total epochs (e.g., 3 out of 15 epochs)
- **Phase 2**: Remaining epochs for fine-tuning
- Longer phase 1 for more complex head architectures

### Monitoring
- Watch validation metrics at phase transition
- Expect initial performance drop when unfreezing backbone
- Look for improved final performance vs single-phase training

## Example Results

```
🚀 Starting two-phase training
📌 Phase 1: Training head with frozen backbone
📊 Phase 1: Updated optimizer with 1 parameter groups, 2M total trainable params
   Group 1 (head): 2M params, lr=1.00e-03

🔄 Switching to Phase 2 at epoch 3  
📌 Phase 2: Unfreezing backbone for full model training
📊 Phase 2: Updated optimizer with 2 parameter groups, 45M total trainable params
   Group 1 (head): 2M params, lr=1.00e-03
   Group 2 (backbone): 43M params, lr=1.00e-05
```

## Comparison with Standard Training

| Approach | Pros | Cons |
|----------|------|------|
| **Standard** | Simple, one set of hyperparameters | Risk of catastrophic forgetting |
| **Two-Phase** | More stable, better final performance | Slightly more complex setup |

## Future Extensions

This simple implementation can be extended to support:
- More gradual unfreezing (layer-by-layer)
- Adaptive unfreezing based on validation metrics  
- Different learning rate schedules per phase
- Custom unfreezing strategies

For more advanced unfreezing strategies, see the configuration examples in the codebase.