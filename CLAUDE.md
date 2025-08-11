# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Training and Fine-tuning
- **Pretraining**: `python cbramod/training/pretraining/pretrain_main.py`
- **Fine-tuning**: `python cbramod/training/finetuning/finetune_main.py`
- **Two-phase training**: `python cbramod/training/finetuning/finetune_main.py --two_phase_training True --epochs 15`
- **Hyperparameter tuning**: Uses Optuna for optimization - integrated in the training scripts
- **Inference**: `python scripts/inference/Inference_local.py`

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Research and Plotting
- **Generate plots**: `python Plot/generate_all_plots.py`
- **Research analysis**: `python Plot/research_plots.py`

## High-Level Architecture

### Core Model Architecture
The repository implements **CBraMod** (Criss-Cross Brain Foundation Model), an EEG foundation model for sleep staging and brain signal analysis.

**Key Components:**
- `cbramod/models/cbramod.py`: Main CBraMod model with transformer encoder
- `cbramod/models/criss_cross_transformer.py`: Custom transformer implementation
- `cbramod/models/model_for_idun.py`: Task-specific model adaptations

### Training Pipeline
**Two-Phase Training Strategy:**
1. **Phase 1**: Train classification head with frozen backbone transformer
2. **Phase 2**: Unfreeze backbone and continue training with lower learning rate

**Training Components:**
- `cbramod/training/finetuning/finetune_trainer.py`: Core training logic
- `cbramod/training/finetuning/finetune_tuner.py`: Hyperparameter optimization with Optuna
- `cbramod/training/finetuning/finetune_evaluator.py`: Model evaluation and metrics

### Dataset Loading
- `cbramod/load_datasets/idun_datasets.py`: IDUN sleep staging dataset
- `cbramod/load_datasets/enhanced_dataset.py`: Enhanced dataset with quality checks
- `cbramod/preprocessing/`: EEG signal preprocessing for OpenNeuro datasets

### Utilities
- `cbramod/utils/signaltools.py`: EEG signal processing utilities
- `cbramod/utils/memory_manager.py`: Memory management for large datasets
- `cbramod/utils/comprehensive_logging.py`: Logging infrastructure

## Key Files and Configuration

### Model Loading
```python
from cbramod.models.cbramod import CBraMod
model = CBraMod()
model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth'))
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
- **Weights & Biases**: Integrated for experiment tracking
- **Optuna**: Hyperparameter optimization studies stored in `artifacts/experiments/optuna_studies/`
- **Reproducibility**: Full experiment configs saved in `reproducibility/`

## Project Structure Notes
- `artifacts/`: Training artifacts, logs, and results
- `deploy_prod/`: Production deployment code and models  
- `saved_models/`: Trained model checkpoints
- `data/datasets/`: EEG datasets (not tracked in git)
- `Plot/`: Research plotting and analysis scripts
- `docs/`: Documentation including quick start guide and memory management

## Development Environment
The codebase uses PyTorch with the following key dependencies:
- `torch`, `numpy`, `pandas` for ML infrastructure
- `mne`, `pyEDFlib` for EEG data processing
- `optuna`, `wandb` for experiment management
- `matplotlib`, `seaborn` for visualization
- `scikit_learn` for metrics and preprocessing