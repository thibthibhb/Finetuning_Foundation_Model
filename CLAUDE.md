# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installing Dependencies
```bash
pip install -r requirements.txt
```

## Common Development Commands

### Prerequisites Check
Before running any training commands, verify these paths exist:
```bash
ls data/datasets/final_dataset/  # Should show: 2017_Open_N, 2019_Open_N, 2023_Open_N, ORP, ORP_improved
ls saved_models/pretrained/      # Should contain pretrained_weights.pth
```

### Training and Fine-tuning

#### Pretraining
**Basic** (minimal arguments - may need dataset setup):
```bash
python cbramod/training/pretraining/pretrain_main.py
```

**Complex** (production-ready with full configuration):
```bash
python cbramod/training/pretraining/pretrain_main.py \
    --epochs 40 \
    --batch_size 128 \
    --lr 1e-4 \
    --model_dir "./saved_models" \
    --run_name "cbramod_pretrain_v1"
```

#### Fine-tuning
**Basic** (will fail without required arguments):
```bash
python cbramod/training/finetuning/finetune_main.py  # ❌ Missing required datasets
```

**Minimal working** (basic fine-tuning):
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models"
```

**Complex** (full production setup):
```bash
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --tune \
    --num_of_classes 4 \
    --multi_eval \
    --label_mapping_version v1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3
```

**Two-phase training** (recommended approach):
```bash
python cbramod/training/finetuning/finetune_main.py \
    --two_phase_training True \
    --epochs 15 \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models"
```

#### Inference
**Prerequisites**: Trained model must exist in saved_models/
```bash
python deploy_prod/code/inference.py
```

### Research and Plotting

#### In-Context Learning (ICL)
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

**Complete ICL Research Protocol:**
```bash
python cbramod/training/icl_main.py \
    --mode full \
    --optuna_n_trials 30 \
    --enable_feature_cache \
    --wandb_project CBraMod-ICL-Research
```

## High-Level Architecture

### Core Model Architecture
The repository implements **CBraMod** (Criss-Cross Brain Foundation Model), an EEG foundation model for sleep staging and brain signal analysis.

**Key Components:**
- `cbramod/models/cbramod.py`: Main CBraMod model with transformer encoder
- `cbramod/models/criss_cross_transformer.py`: Custom transformer implementation
- `cbramod/models/model_for_idun.py`: Task-specific model adaptations

### Training Pipeline
**Available Training Methods:**
- **Standard Fine-tuning**: End-to-end supervised learning
- **Two-Phase Training**: Progressive unfreezing (add `--two_phase_training True --phase1_epochs 3`)
- **In-Context Learning (ICL)**: Few-shot learning (`--icl_mode proto` or `--icl_mode meta_proto`)
- **Hyperparameter Optimization**: Automated search with Optuna

**Training Components:**
- `cbramod/training/finetuning/finetune_trainer.py`: Core training logic
- `cbramod/training/finetuning/finetune_tuner.py`: Hyperparameter optimization with Optuna
- `cbramod/training/finetuning/finetune_evaluator.py`: Model evaluation and metrics
- `cbramod/training/icl_main.py`: Comprehensive ICL evaluation script
- `cbramod/training/icl_trainer.py`: ICL trainer with Proto/Meta modes
- `cbramod/training/icl_data.py`: Episodic dataset for ICL

### Dataset Loading
- `cbramod/load_datasets/idun_datasets.py`: IDUN sleep staging dataset
- `cbramod/preprocessing/`: EEG signal preprocessing for OpenNeuro datasets
  - `process_openneuro_2017.py`: Process 2017 OpenNeuro dataset
  - `process_openneuro_2019.py`: Process 2019 OpenNeuro dataset
  - `process_openneuro_2023.py`: Process 2023 OpenNeuro dataset
  - `preprocess_idun.py`: IDUN dataset preprocessing

### Utilities
- `cbramod/utils/signaltools.py`: EEG signal processing utilities
- `cbramod/utils/memory_manager.py`: Memory management for large datasets
- `cbramod/utils/comprehensive_logging.py`: Logging infrastructure
- `cbramod/utils/noise_injection.py`: Data augmentation utilities
- `cbramod/utils/util.py`: General utility functions

## Key Files and Configuration

### Model Loading
```python
from cbramod.models.cbramod import CBraMod
model = CBraMod()
model.load_state_dict(torch.load('saved_models/pretrained/pretrained_weights.pth'))
```

### Training Parameters
- Default epochs: 100 for fine-tuning, 40 for pretraining
- Default batch size: 64 for fine-tuning, 128 for pretraining
- Learning rates: 1e-3 for head, 1e-5 for backbone in two-phase training
- Supports mixed precision training and gradient clipping

### Model Architecture Defaults
- Input dimension: 200 (EEG patch size)
- Model dimension: 200
- Transformer layers: 12
- Attention heads: 8
- Feedforward dimension: 800
- Sequence length: 30

### Experiment Tracking
- **Weights & Biases**: Integrated for experiment tracking - requires WandB login
- **WandB Logs**: Experiment data stored in `wandb/` (254+ experiment runs)
- **Optuna**: Hyperparameter optimization available via `finetune_tuner.py`

## Project Structure Notes
- `cbramod/`: Main CBraMod module
  - `cbramod/models/`: Model architectures (CBraMod, transformers)
  - `cbramod/training/`: All training methods (finetuning, ICL, pretraining)
  - `cbramod/load_datasets/`: Dataset loaders (IDUN_EEG, OpenNeuro)
  - `cbramod/preprocessing/`: EEG signal preprocessing
  - `cbramod/utils/`: Utilities (signal processing, memory management)
- `deploy_prod/`: Production deployment code and models
  - `deploy_prod/code/`: Deployment scripts including `inference.py`
- `saved_models/`: Model storage
  - `saved_models/pretrained/`: Pretrained weights and models
- `data/datasets/final_dataset/`: EEG datasets (not tracked in git)
  - Contains: 2017_Open_N, 2019_Open_N, 2023_Open_N, ORP, ORP_improved
- `Extra/`: Additional utilities and experimental scripts

## Development Environment
The codebase uses PyTorch with the following key dependencies:
- `torch`, `numpy`, `pandas` for ML infrastructure
- `mne`, `pyEDFlib` for EEG data processing
- `optuna`, `wandb` for experiment management
- `matplotlib`, `seaborn` for visualization
- `scikit_learn` for metrics and preprocessing

## Additional Development Resources

### Documentation Files
- `TWO_PHASE_TRAINING.md`: Detailed guide for progressive unfreezing training
- `docs/QUICK_START_GUIDE.md`: Quick start reference
- `docs/memory_management_guide.md`: Memory optimization strategies
- `docs/DATA_PIPELINE_README.md`: Data preprocessing documentation

### Common Troubleshooting

#### Training Issues
- **Missing datasets error**: Ensure `data/datasets/final_dataset/` contains required dataset folders
- **CUDA out of memory**: Reduce batch size or enable mixed precision training
- **WandB login required**: Run `wandb login` before training with experiment tracking

#### ICL Issues
- **Missing datasets error**: Ensure `data/datasets/final_dataset/` contains required dataset folders
- **ICL configuration errors**: Use correct K values (2, 4, 8, 16, 32) and modes (proto, meta_proto)
- **Feature cache issues**: Enable `--enable_feature_cache` for faster repeated runs

### Development Notes
- **Two-phase training** is the recommended and proven approach for CBraMod fine-tuning
- **Memory management** system implemented in `cbramod/utils/memory_manager.py`
  - Automated cleanup between trials
  - Intelligent checkpoint management (max 5 files by default)
  - Memory leak detection and monitoring
  - GPU cache management utilities
- **Training stability** improvements with comprehensive logging and error handling
- **Current codebase status**: All core functionality verified and operational
- **Inference deployment**: Production-ready inference available in `deploy_prod/code/`

## Current System Status (Updated)

### ✅ Verified Working Components
- **Training pipelines**: Pretraining, fine-tuning, and ICL scripts operational
- **Memory management**: Automated cleanup and monitoring implemented
- **ICL framework**: Proto and Meta-ICL with comprehensive evaluation
- **Model storage**: Pretrained and fine-tuned model checkpoints well-organized
- **Dataset access**: EEG datasets properly structured and accessible
- **WandB integration**: Experiment tracking for all training methods

### 📊 Current Data Status
- **Pretrained models**: Available in `saved_models/pretrained/`
- **Production models**: Deployment-ready files in `deploy_prod/`
- **Dataset completeness**: All required datasets (2017-2023 OpenNeuro, ORP) present
- **ICL capabilities**: Full Proto/Meta-ICL evaluation with Bayesian optimization

### 🔧 Maintenance Recommendations
- Regular cleanup of WandB runs and experiment logs
- Monitor checkpoint accumulation in `saved_models/`
- Archive experimental data if disk space becomes constrained