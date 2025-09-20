# CBraMod Training Guide

This guide covers all training methods available in CBraMod for EEG sleep staging, from basic fine-tuning to advanced in-context learning and hyperparameter optimization.

## 🎯 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Verify data and model paths
ls data/datasets/final_dataset/  # Should show: 2017_Open_N, 2019_Open_N, 2023_Open_N, ORP, ORP_improved
ls saved_models/pretrained/      # Should contain pretrained_weights.pth
```

### Basic Fine-tuning (Recommended Starting Point)
```bash
# Simple fine-tuning with default parameters
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 20 \
    --run_name "basic_finetune"
```

---

## 🚀 Training Methods

### 1. Standard Supervised Fine-tuning

**Basic Fine-tuning**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --run_name "standard_finetune"
```

**Multi-dataset Training**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 30 \
    --batch_size 64 \
    --run_name "multi_dataset_finetune"
```

### 2. Two-Phase Training (Recommended)

Two-phase training provides better stability and performance by first training only the classification head, then unfreezing the backbone.

**Basic Two-Phase**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --two_phase_training True \
    --epochs 15 \
    --run_name "two_phase_basic"
```

**Advanced Two-Phase with Custom Parameters**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --two_phase_training True \
    --phase1_epochs 5 \
    --head_lr 2e-3 \
    --backbone_lr 5e-6 \
    --epochs 25 \
    --batch_size 128 \
    --run_name "two_phase_advanced"
```

### 3. Hyperparameter Optimization

**Automated Hyperparameter Tuning with Optuna**
```bash
# Basic hyperparameter optimization (most useful command)
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --tune \
    --run_name "hparam_optimization"
```

**Advanced Hyperparameter Search**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --tune \
    --num_of_classes 4 \
    --multi_eval \
    --label_mapping_version v1 \
    --run_name "comprehensive_hparam_search"
```

### 4. In-Context Learning (ICL)

**ICL-Proto (No Training, Fast)**
```bash
# Basic prototypical ICL
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto \
    --icl_k 16 \
    --icl_m 64 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --run_name "icl_proto_k16"
```

**ICL-Proto with Optimized Parameters**
```bash
# Proto ICL with balanced support and cosine similarity
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto \
    --icl_k 8 \
    --icl_m 64 \
    --icl_balance_support \
    --icl_cosine \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --run_name "icl_proto_optimized"
```

**Meta-ICL (With Training)**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode meta_proto \
    --icl_k 16 \
    --icl_m 64 \
    --epochs 20 \
    --head_lr 1e-3 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --run_name "meta_icl_k16"
```

### 5. Robustness Analysis with Noise Injection

**EMG Artifacts (Muscle Activity)**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 20 \
    --noise_level 0.10 \
    --noise_type emg \
    --noise_seed 42 \
    --run_name "robustness_emg_10pct"
```

**Movement Artifacts**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 20 \
    --noise_level 0.10 \
    --noise_type movement \
    --noise_seed 42 \
    --run_name "robustness_movement_10pct"
```

**Realistic Mixed Artifacts**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 20 \
    --noise_level 0.05 \
    --noise_type realistic \
    --noise_seed 42 \
    --run_name "robustness_realistic_5pct"
```

---

## 🔧 Key Parameters

### Essential Parameters
| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--downstream_dataset` | Target dataset | `IDUN_EEG` | `IDUN_EEG` |
| `--datasets_dir` | Path to datasets | Required | `data/datasets/final_dataset` |
| `--datasets` | Training datasets | Required | `ORP`, `ORP,2023_Open_N` |
| `--use_pretrained_weights` | Use foundation model | `False` | `True` |
| `--model_dir` | Model save directory | Required | `"./saved_models"` |
| `--epochs` | Training epochs | `100` | `15-30` |
| `--batch_size` | Batch size | `64` | `32,64,128` |
| `--run_name` | Experiment name | Auto-generated | `"my_experiment"` |

### Training Strategy Parameters
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--two_phase_training` | Enable two-phase training | `False` | `True/False` |
| `--phase1_epochs` | Phase 1 duration | `3` | `3-10` |
| `--head_lr` | Head learning rate | `1e-3` | `1e-4` to `1e-2` |
| `--backbone_lr` | Backbone learning rate | `1e-5` | `1e-6` to `1e-4` |
| `--tune` | Hyperparameter optimization | `False` | `True/False` |
| `--multi_eval` | Multi-subject evaluation | `False` | `True/False` |

### ICL Parameters
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--icl_mode` | ICL mode | `off` | `proto`, `meta_proto` |
| `--icl_k` | Support examples per class | `16` | `2,4,8,16,32` |
| `--icl_m` | Query examples per episode | `64` | `32,64,128` |
| `--icl_balance_support` | Balance support classes | `False` | `True/False` |
| `--icl_cosine` | Use cosine similarity | `False` | `True/False` |

### Robustness Parameters
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--noise_level` | Noise intensity | `0.0` | `0.05,0.10,0.20` |
| `--noise_type` | Artifact type | `clean` | `emg,movement,electrode,realistic,gaussian` |
| `--noise_seed` | Noise reproducibility | `42` | Any integer |

---

## 🎯 Most Useful Commands

### 1. **Best Overall Performance** (Two-Phase + Multi-Dataset)
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --two_phase_training True \
    --epochs 20 \
    --batch_size 64 \
    --run_name "best_performance"
```

### 2. **Fastest Training** (ICL-Proto, No Training Required)
```bash
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto \
    --icl_k 16 \
    --icl_m 64 \
    --icl_cosine \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --run_name "fastest_icl"
```

### 3. **Hyperparameter Optimization** (Find Best Settings)
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --tune \
    --run_name "find_best_hparams"
```

### 4. **Production Ready** (Robust + Multi-Subject Validation)
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --two_phase_training True \
    --multi_eval \
    --epochs 25 \
    --run_name "production_ready"
```

### 5. **Quick Debugging** (Fast, Small Dataset)
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --epochs 5 \
    --batch_size 32 \
    --no_wandb \
    --run_name "debug_run"
```

---

## 📊 Experiment Tracking

### Weights & Biases Integration
```bash
# Login to W&B (one-time setup)
wandb login

# Training with W&B logging (default)
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --wandb_project "CBraMod-Experiments" \
    --wandb_entity "your-entity" \
    --run_name "tracked_experiment"

# Disable W&B for quick testing
python cbramod/training/finetuning/finetune_main.py \
    [other parameters] \
    --no_wandb
```

---

## 🛠 Advanced Usage

### Custom Model Configuration
```bash
# Different model architectures
python cbramod/training/finetuning/finetune_main.py \
    [base parameters] \
    --head_type attention \
    --dropout 0.2 \
    --use_focal_loss True \
    --label_smoothing 0.1
```

### Memory Optimization
```bash
# For limited GPU memory
python cbramod/training/finetuning/finetune_main.py \
    [base parameters] \
    --batch_size 32 \
    --use_amp True \
    --num_workers 4
```

### Reproducibility
```bash
# Fully reproducible runs
python cbramod/training/finetuning/finetune_main.py \
    [base parameters] \
    --seed 42 \
    --noise_seed 42
```

---

## 🔍 Results and Analysis

### Generate Publication Plots
After training, generate analysis plots:

```bash
# Extract data from W&B (REQUIRED FIRST STEP)
python plots/scripts/load_and_structure_runs.py \
    --project CBraMod-earEEG-tuning \
    --entity your-entity \
    --output-dir plots/data/

# Generate specific analysis figures
python plots/scripts/plot_hyperparameter_sensitivity.py
python plots/scripts/plot_heads_vs_weights_comparison.py
python plots/scripts/plot_calibration_comparison.py
python plots/scripts/plot_subjects_vs_minutes.py
python plots/scripts/plot_task_granularity_combined.py
python plots/scripts/plot_dataset_composition.py
```

---

## 🚨 Troubleshooting

### Common Issues

**1. Missing datasets error**
```bash
# Check data paths
ls data/datasets/final_dataset/
# Should show: 2017_Open_N, 2019_Open_N, 2023_Open_N, ORP, ORP_improved
```

**2. CUDA out of memory**
```bash
# Reduce batch size and enable mixed precision
--batch_size 32 --use_amp True
```

**3. W&B login issues**
```bash
# Use offline mode
--wandb_offline
# Or disable completely
--no_wandb
```

**4. Slow training**
```bash
# Use fewer datasets, reduce epochs, or try ICL
--datasets ORP --epochs 10
# Or use ICL (no training)
--icl_mode proto --icl_k 16
```

### Performance Tips

1. **Start with ICL-Proto** for quick baselines
2. **Use two-phase training** for best performance
3. **Enable hyperparameter tuning** for production models
4. **Use multi-dataset training** for robustness
5. **Test with noise injection** for real-world deployment

---

## 📁 Data Acquisition

CBraMod requires labeled EEG datasets for fine-tuning:

### Required Datasets
- **ORP**: Data from IDUN -- You can select your own dataset - Target downstream dataset
- **OpenNeuro datasets**: 2017_Open_N, 2019_Open_N, 2023_Open_N [https://openneuro.org/datasets/ds005178/versions/1.0.0]

### Setup
1. Download datasets to `data/datasets/final_dataset/`
2. Ensure pretrained weights in `saved_models/pretrained/pretrained_weights.pth`
3. Verify paths: `ls data/datasets/final_dataset/`

---

## 📚 File Structure

```
cbramod/training/
├── finetuning/
│   ├── finetune_main.py          # Main training script
│   ├── finetune_trainer.py       # Core training logic
│   ├── finetune_tuner.py         # Optuna hyperparameter optimization
│   └── finetune_evaluator.py     # Evaluation metrics
├── icl_trainer.py                # In-context learning implementation
├── icl_data.py                   # Episodic data loading for ICL
└── pretraining/
    ├── pretrain_main.py          # Foundation model pretraining
    └── pretrain_trainer.py       # Pretraining logic
```

---

## ✅ Current System Status

### Verified Working Features
- **All training methods**: Standard fine-tuning, two-phase training, ICL, and hyperparameter optimization
- **Noise injection**: EMG, movement, electrode, realistic, and gaussian noise types
- **WandB integration**: Full experiment tracking with 254+ logged runs
- **Memory management**: Automated cleanup and checkpoint management
- **Plot generation**: Modern plotting system with structured data pipeline

### Current Data Status
- **Datasets**: 313GB of EEG data properly structured (2017-2023 OpenNeuro, ORP, ORP_improved)
- **Pretrained models**: Available in `saved_models/pretrained/` (including pretrained_weights.pth)
- **Plot data**: Generated CSV files available in `plots/data/`
- **Model checkpoints**: 16 well-managed checkpoint files (609MB total)

### Production Ready Components
- **Two-phase training**: Proven approach for best performance
- **ICL implementation**: Fast baseline generation without training
- **Robustness testing**: Comprehensive noise injection for real-world validation
- **Hyperparameter optimization**: Optuna integration for automated tuning

---

This comprehensive guide covers all implemented training methods in CBraMod. All commands have been verified against the current codebase. Start with the Quick Start section and use the Most Useful Commands for common scenarios.