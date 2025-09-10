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
python scripts/inference/Inference_local.py
```

### Research and Plotting

#### CRITICAL: Plot Data Generation Workflow
**Step 1**: Generate structured data from WandB (REQUIRED FIRST):
```bash
python plots/scripts/load_and_structure_runs.py \
    --project CBraMod-earEEG-tuning \
    --entity thibaut_hasle-epfl \
    --output-dir plots/data/
```
This creates the CSV files needed for all subsequent plots.

**Step 2**: Verify data was created:
```bash
ls plots/data/  # Should show: all_runs_flat.csv, cohort_4_class_flat.csv, etc.
```

#### Individual Plot Generation
**Hyperparameter Analysis**:
```bash
python plots/scripts/plot_hyperparameter_sensitivity.py
python plots/scripts/plot_heads_vs_weights_comparison.py
```

**Performance Analysis**:
```bash
python plots/scripts/plot_calibration_comparison.py
python plots/scripts/plot_subjects_vs_minutes.py
```

**Task Analysis**:
```bash
python plots/scripts/plot_task_granularity_combined.py
python plots/scripts/plot_dataset_composition.py
```

**Deployment Visualization**:
```bash
python plots/scripts/deployment/create_deployment_visualization.py
python plots/scripts/deployment/create_improved_deployment_plots.py
```

#### Legacy Plot Generation (requires WandB API access)
**Comprehensive research plots**:
```bash
python Plot/research_plots.py
```

**All plots with options**:
```bash
python Plot/generate_all_plots.py \
    --plots all \
    --output-dir ./experiments/results/figures/ \
    --entity thibaut_hasle-epfl \
    --project CBraMod-earEEG-tuning
```

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
- **Optuna**: Hyperparameter optimization studies stored in `experiments/optuna_studies/`
- **Reproducibility**: Full experiment configs saved in `experiments/configs/`

## Project Structure Notes
- `experiments/`: Training artifacts, logs, and results
  - `experiments/configs/`: Reproducibility configurations
  - `experiments/logs/`: Training and error logs
  - `experiments/results/`: Analysis results and figures
  - `experiments/wandb/`: WandB run data
- `deploy_prod/`: Production deployment code and models  
- `saved_models/`: Consolidated model storage (pretrained/, finetuned/, production/)
- `data/datasets/final_dataset/`: EEG datasets (not tracked in git)
- `analysis/`: All analysis scripts organized by category
  - `analysis/deployment/`: Deployment analysis
  - `analysis/subjects/`: Subject-specific analysis
  - `analysis/experiments/`: Experiment-level analysis
  - `analysis/results/`: All CSV analysis results
- `plots/`: All plotting functionality
  - `plots/scripts/`: Plotting scripts organized by type
  - `plots/figures/`: Generated figures (PDF/PNG)
  - `plots/data/`: Plot data files
  - `plots/style/`: Figure styling configuration
- `docs/`: Documentation including quick start guide and memory management
- `tools/`: Utility scripts
- `temp/`: Temporary files and screenshots

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

#### Plot Generation Issues
- **Missing CSV files**: Must run `load_and_structure_runs.py` first to generate data
- **WandB API errors**: Ensure correct project name and entity in plot commands
- **Empty plots**: Verify CSV files contain data after WandB data extraction

### Development Notes
- **Two-phase training** is the recommended approach for CBraMod fine-tuning
- **Mixed precision training** available via PyTorch's automatic mixed precision
- **Gradient clipping** enabled by default to prevent training instability
- **Memory management** utilities available in `cbramod/utils/memory_manager.py`