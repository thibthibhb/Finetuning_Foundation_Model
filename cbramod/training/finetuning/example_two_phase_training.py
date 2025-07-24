#!/usr/bin/env python3
"""
Example: Simple Two-Phase Training for CBraMod

This script demonstrates how to use the simple two-phase training approach:
- Phase 1: Train classification head with frozen backbone (3 epochs by default)
- Phase 2: Unfreeze backbone and continue training with lower learning rate

Usage examples:

# Basic two-phase training with default parameters
python finetune_main.py --two_phase_training True --epochs 15

# Custom two-phase training
python finetune_main.py \
    --two_phase_training True \
    --phase1_epochs 5 \
    --head_lr 2e-3 \
    --backbone_lr 5e-6 \
    --epochs 20

# With other training options
python finetune_main.py \
    --two_phase_training True \
    --phase1_epochs 3 \
    --head_lr 1e-3 \
    --backbone_lr 1e-5 \
    --epochs 15 \
    --batch_size 32 \
    --optimizer AdamW \
    --datasets ORP,2023_Open_N \
    --num_of_classes 5
"""

import subprocess
import sys

def run_two_phase_training():
    """Example function showing how to run two-phase training"""
    
    # Example 1: Basic two-phase training
    print("🚀 Example 1: Basic two-phase training")
    cmd1 = [
        "python", "finetune_main.py",
        "--two_phase_training", "True",
        "--epochs", "15",
        "--phase1_epochs", "3",
        "--head_lr", "1e-3", 
        "--backbone_lr", "1e-5"
    ]
    print("Command:", " ".join(cmd1))
    
    # Example 2: Custom two-phase training with more configuration
    print("\n🚀 Example 2: Custom two-phase training")
    cmd2 = [
        "python", "finetune_main.py",
        "--two_phase_training", "True",
        "--phase1_epochs", "5",        # Longer phase 1
        "--head_lr", "2e-3",           # Higher head learning rate
        "--backbone_lr", "5e-6",       # Lower backbone learning rate
        "--epochs", "20",              # More total epochs
        "--batch_size", "32",
        "--optimizer", "AdamW",
        "--datasets", "ORP,2023_Open_N",
        "--num_of_classes", "5"
    ]
    print("Command:", " ".join(cmd2))
    
    print("\n📋 Two-Phase Training Parameters:")
    print("- --two_phase_training: Enable two-phase training (True/False)")
    print("- --phase1_epochs: Number of epochs for phase 1 (frozen backbone)")
    print("- --head_lr: Learning rate for head/classifier")
    print("- --backbone_lr: Learning rate for backbone in phase 2")
    print("- --epochs: Total number of training epochs")
    
    print("\n🎯 Expected Training Flow:")
    print("Phase 1 (epochs 0-2): Train head with frozen backbone at head_lr")
    print("Phase 2 (epochs 3-14): Train full model with head_lr and backbone_lr") 
    
    print("\n💡 Tips:")
    print("- Use head_lr 5-10x higher than backbone_lr")
    print("- Phase 1 should be 15-25% of total epochs")
    print("- Start with default values and tune based on validation performance")

if __name__ == "__main__":
    run_two_phase_training()