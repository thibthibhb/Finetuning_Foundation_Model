# In-Context Learning (ICL) for CBraMod

This directory contains the comprehensive ICL implementation for sleep staging with CBraMod, following the research protocol for systematic evaluation.

## 🎯 Overview

**Two ICL Methods Implemented:**
- **ICL-Proto**: No training, prototype-based classification with exhaustive grid search
- **Meta-ICL**: Episodic training with learnable projections and Bayesian optimization

**Key Features:**
- Subject-disjoint splits (fixed)
- 5-class v1 labels (fixed)
- Cohen's κ primary metric with 95% confidence intervals
- Feature caching for faster sweeps
- W&B logging with proper tags
- Early stopping for Meta-ICL

## 📁 Files

```
cbramod/training/
├── icl_main.py          # Main comprehensive ICL evaluation script
├── icl_trainer.py       # ICL trainer with Proto/Meta modes + early stopping
├── icl_data.py          # Episodic dataset (groups by nights)
└── README_ICL.md        # This file
```

## 🚀 Quick Start

### Current Working Implementation (Through finetune_main.py)

**ICL-Proto (no training, fast):**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto --icl_k 16 --icl_m 64 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP --no_wandb
```

**Meta-ICL (with training):**
```bash
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode meta_proto --icl_k 16 --icl_m 64 \
    --epochs 20 --head_lr 1e-3 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP --no_wandb
```

**With specific parameters:**
```bash
# Proto with balanced support and cosine similarity
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto --icl_k 8 --icl_m 64 \
    --icl_balance_support --icl_cosine \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP --no_wandb
```

### ⚠️ Standalone Framework (In Development)

The comprehensive `icl_main.py` with automatic grid search and Bayesian optimization is currently being integrated. Use the working commands above for now.

### 📋 Current Capabilities

**✅ Working through finetune_main.py:**
- ✅ ICL-Proto with specific K values
- ✅ Meta-ICL with episodic training + early stopping
- ✅ Temperature parameter support  
- ✅ Balanced support sampling
- ✅ Cosine vs L2 similarity
- ✅ W&B logging with proper tags
- ✅ CSV results export

### Research Protocol Commands

**Phase A - Proto Grid Search (60 configs):**
```bash
python cbramod/training/icl_main.py \
    --mode proto \
    --wandb_project CBraMod-ICL-Research \
    --wandb_entity thibaut_hasle-epfl \
    --results_dir ./experiments/icl_results/proto_phase
```

**Compare Different Pretrained Weights:**
```bash
# Using default pretrained weights (default)
python cbramod/training/icl_main.py \
    --mode proto \
    --foundation_dir ./saved_models/pretrained/pretrained_weights.pth \
    --wandb_project CBraMod-Pretrained-Comparison

# Using best loss pretrained weights  
python cbramod/training/icl_main.py \
    --mode proto \
    --foundation_dir ./cbramod/utils/weights/BEST__loss8698.99.pth \
    --wandb_project CBraMod-Pretrained-Comparison
```

**Phase B - Meta Bayesian Search (30 trials):**
```bash
python cbramod/training/icl_main.py \
    --mode meta \
    --optuna_n_trials 30 \
    --wandb_project CBraMod-ICL-Research \
    --wandb_entity thibaut_hasle-epfl \
    --results_dir ./experiments/icl_results/meta_phase
```

**Phase C - Complete Protocol:**
```bash
python cbramod/training/icl_main.py \
    --mode full \
    --optuna_n_trials 30 \
    --enable_feature_cache \
    --wandb_project CBraMod-ICL-Research \
    --wandb_entity thibaut_hasle-epfl \
    --results_dir ./experiments/icl_results/full_eval
```

## 🔧 Search Spaces

### ICL-Proto (Exhaustive Grid)
- **K (shots)**: {2, 4, 8, 16, 32}
- **similarity**: {cosine, L2}  
- **support_balancing**: {True, False}
- **temperature**: {1.0, 2.0, 3.0}
- **M (queries)**: 64 (fixed)
- **proj_dim**: 512 (fixed, avoids random projection)

**Total**: 5 × 2 × 2 × 3 = **60 configurations**

### Meta-ICL (Bayesian Search)
- **K (shots)**: {4, 8, 16, 32}
- **proj_dim**: {256, 512, 768}
- **similarity**: {cosine, L2}
- **head_lr**: log-uniform [3e-4, 3e-3]
- **weight_decay**: {0.0, 1e-4}
- **batch_episodes**: {4, 8, 16}
- **epochs**: {10, 15, 20}
- **temperature**: {1.0, 2.0, 3.0}
- **M (queries)**: 64 (fixed)

**Default trials**: 30 (25-40 recommended)

### 🆕 Pretrained Weights Options
- **default**: `./saved_models/pretrained/pretrained_weights.pth` (default if not specified)
- **best**: `./cbramod/utils/weights/BEST__loss8698.99.pth` (use `--foundation_dir` to specify)

## 📊 Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--mode` | Evaluation mode | `full` | `proto`, `meta`, `curves`, `full` |
| `--optuna_n_trials` | Bayesian trials for Meta-ICL | `30` | 10-50 |
| `--enable_feature_cache` | Cache features for speed | `True` | boolean |
| `--results_dir` | Output directory | `./experiments/icl_results` | path |
| `--wandb_project` | W&B project name | `CBraMod-ICL-Research` | string |
| `--seed` | Random seed | `42` | integer |

## 🏃 Usage Examples

### Minimal Proto Test
```bash
# Quick proto test (small search space)
python cbramod/training/icl_main.py \
    --mode proto \
    --no_wandb
```

### Full Research Evaluation
```bash
# Complete systematic evaluation (paper results)
python cbramod/training/icl_main.py \
    --mode full \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --model_dir ./saved_models \
    --optuna_n_trials 30 \
    --enable_feature_cache \
    --wandb_project CBraMod-ICL-Research \
    --wandb_entity thibaut_hasle-epfl \
    --results_dir ./experiments/icl_results/paper_eval \
    --seed 42
```

### Custom Meta-ICL
```bash
# Meta-ICL with specific parameters
python cbramod/training/icl_main.py \
    --mode meta \
    --optuna_n_trials 40 \
    --wandb_project MyProject \
    --results_dir ./my_results
```

## 📈 Output & Results

**Files Generated:**
- `icl_experiment_summary.json`: Complete results with configs
- `icl_results.csv`: CSV for analysis and plotting  
- Feature cache in `--cache_dir` (speeds up subsequent runs)

**W&B Tags Added:**
- `method:{proto|meta_proto}`
- `K:{shot_count}`
- `similarity:{cosine|L2}`
- `proj_dim:{dimension}`
- `num_classes:5`
- `label_map:v1`

**Metrics Reported:**
- **Primary**: Cohen's κ (with 95% CI)
- **Secondary**: Macro-F1, Accuracy
- **Per-episode**: Statistics across test nights

## ⚙️ Advanced Options

**Custom Search Spaces (modify in code):**
```python
# In ICLConfig class
proto_k_values = [8, 16, 32]           # Reduce K range
meta_proj_dims = [512]                 # Fix projection dim  
optuna_n_trials = 50                   # More thorough search
```

**Feature Caching:**
```bash
# Enable caching (recommended for multiple runs)
--enable_feature_cache
--cache_dir ./experiments/icl_cache
```

**Reproducibility:**
```bash
# Fixed seed for reproducible results
--seed 42
```

## 🔬 Integration with Grid Runner

Use with the comprehensive grid runner:

```bash
# Run fine-tuning + ICL together
python tools/run_grid.py \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --wandb_project CBraMod-ICL-Research \
    --optuna_n_trials 30 \
    --enable_feature_cache
```

Then generate plots with confidence intervals:
```bash
python tools/plot_iclvft.py \
    --wandb_project CBraMod-ICL-Research \
    --outdir ./figures/
```

## 📚 Implementation Details

- **Episodes**: One episode = one night (K support + M queries)  
- **Splits**: Subject-disjoint (fixed across all methods)
- **Labels**: 5-class v1 mapping (fixed)
- **Backbone**: Frozen during ICL (both Proto and Meta)
- **Early Stopping**: Meta-ICL monitors validation κ (patience=3)
- **Optimization**: Primary=κ, Secondary=macro-F1
- **CI**: 95% confidence intervals using t-distribution

## 🎯 Typical Results

**Expected Performance:**
- **ICL-Proto**: κ ≈ 0.6-0.7, best at K=16, cosine similarity
- **Meta-ICL**: κ ≈ 0.65-0.75, improvement over Proto at lower K
- **Knee Point**: Often around K=8-16 for EEG data

**Runtime:**
- Proto: ~30-60 minutes (with caching)
- Meta: ~2-4 hours (30 trials, early stopping)
- Full: ~3-5 hours total

## 🎯 Tested Working Commands

### Single ICL Experiments
```bash
# Proto ICL (K=16, no training)
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto --icl_k 16 --icl_m 64 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP --no_wandb

# Meta-ICL (K=16, with training + early stopping)
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode meta_proto --icl_k 16 --icl_m 64 \
    --epochs 20 --head_lr 1e-3 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP --no_wandb

# Proto with balanced support + cosine similarity
python cbramod/training/finetuning/finetune_main.py \
    --icl_mode proto --icl_k 8 --icl_m 64 \
    --icl_balance_support --icl_cosine \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP --no_wandb
```

### Grid Experiments (Multiple K values)
```bash
# Run Proto K=8,16,32 + Meta K=16 (dry run first)
python tools/run_grid.py \
    --datasets ORP --skip_finetune --dry_run

# Actual run
python tools/run_grid.py \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --skip_finetune \
    --wandb_project CBraMod-ICL-Research
```

### Generate Plots with 95% Confidence Intervals
```bash
# From CSV data
python tools/plot_iclvft.py \
    --force_csv \
    --results_csv ./Plot_Clean/data_full/summary.csv \
    --outdir ./figures/

# From WandB (if you have runs there)
python tools/plot_iclvft.py \
    --wandb_project CBraMod-ICL-Research \
    --wandb_entity thibaut_hasle-epfl \
    --outdir ./figures/
```

### Complete Workflow Example
```bash
# 1. Run ICL experiments
python tools/run_grid.py \
    --datasets ORP \
    --wandb_project CBraMod-ICL-Test \
    --results_csv ./my_results.csv

# 2. Generate publication-ready plots  
python tools/plot_iclvft.py \
    --force_csv \
    --results_csv ./my_results.csv \
    --outdir ./figures/

# Outputs: Bar plots + K-shot curves with 95% CI for κ, accuracy, macro-F1
```