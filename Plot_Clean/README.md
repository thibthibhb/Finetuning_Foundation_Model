# CBraMod Publication-Ready Analysis & Plotting

Clean, structured approach to analyzing and visualizing CBraMod experiment results with descriptive naming and organized output structure.

## 📋 Overview

This system provides publication-ready plots for CBraMod research with:

1. **Descriptive naming** - Clear, meaningful names for both scripts and outputs
2. **Flat structure** - All figures in one location (`figures/`) for easy access
3. **Consistent styling** - Unified visual style across all plots
4. **Reproducible workflow** - CSV-based analysis independent of WandB API

## 🚀 Quick Start

### 1. Generate Structured Data (Required First)

Extract and structure experimental data from WandB:

```bash
python Plot_Clean/load_and_structure_runs.py \
    --project CBraMod-earEEG-tuning \
    --entity thibaut_hasle-epfl \
    --output-dir Plot_Clean/data/
```

This creates CSV files needed for all subsequent analyses.

### 2. Generate Individual Plots

All publication-ready plots with descriptive names:

```bash
# Calibration and performance comparison
python Plot_Clean/plot_calibration_comparison.py --csv Plot_Clean/data/all_runs_flat.csv

# Training efficiency analysis  
python Plot_Clean/plot_subjects_vs_minutes.py --csv Plot_Clean/data/all_runs_flat.csv

# Training stage progression analysis
python Plot_Clean/plot_training_stage_gains.py --csv Plot_Clean/data/all_runs_flat.csv

# Noise robustness analysis (paired subjects)
python Plot_Clean/plot_robustness_noise_paired.py \
    --csv Plot_Clean/data/all_runs_flat.csv \
    --test-subjects Plot_Clean/data/all_test_subjects_complete.csv

# Hyperparameter sensitivity analysis
python Plot_Clean/plot_hyperparameter_sensitivity.py --csv Plot_Clean/data/all_runs_flat.csv

# Task granularity analysis
python Plot_Clean/plot_task_granularity_combined.py --csv Plot_Clean/data/all_runs_flat.csv
```

## 📁 New Organized Structure

```
Plot_Clean/
├── 🐍 Python Scripts (Descriptive Names)
│   ├── plot_calibration_comparison.py        # Model calibration analysis
│   ├── plot_subjects_vs_minutes.py          # Training efficiency analysis
│   ├── plot_training_stage_gains.py         # Two-phase training progression
│   ├── plot_robustness_noise_paired.py      # Noise robustness (paired analysis)
│   ├── plot_robustness_noise.py             # Noise robustness (basic analysis)
│   ├── plot_hyperparameter_sensitivity.py   # Hyperparameter sensitivity
│   ├── plot_task_granularity_combined.py    # Task granularity analysis
│   ├── plot_freeze_comparison.py            # Freeze vs unfreeze comparison
│   └── figure_style.py                      # Consistent styling utilities
│
├── 📊 Generated Figures (Flat Structure)
│   ├── calibration_comparison.pdf/.png      # Model calibration plots
│   ├── subjects_vs_minutes_analysis.pdf/.png # Training efficiency plots
│   ├── training_stage_gains.pdf/.png        # Training progression plots
│   ├── robustness_noise_paired.pdf/.png     # Paired noise robustness
│   ├── hyperparameter_sensitivity.pdf/.png  # Hyperparameter analysis
│   └── task_granularity_combined.pdf/.png   # Task granularity analysis
│
└── 📁 Data & Utilities
    ├── data/                                 # Structured CSV files from WandB
    ├── load_and_structure_runs.py          # WandB data extraction
    └── simple_boxplot.py                   # Basic plotting utilities
```

## 🎯 Plot Descriptions

| Script | Output | Description |
|--------|--------|-------------|
| `plot_calibration_comparison.py` | `calibration_comparison.*` | Model performance calibration across conditions |
| `plot_subjects_vs_minutes.py` | `subjects_vs_minutes_analysis.*` | Training data efficiency analysis |
| `plot_training_stage_gains.py` | `training_stage_gains.*` | Two-phase training progression analysis |
| `plot_robustness_noise_paired.py` | `robustness_noise_paired.*` | Paired-subject noise robustness analysis |
| `plot_hyperparameter_sensitivity.py` | `hyperparameter_sensitivity.*` | Hyperparameter sensitivity analysis |
| `plot_task_granularity_combined.py` | `task_granularity_combined.*` | Task granularity impact analysis |

## 🔧 Key Features

### Consistent Naming Convention
- **Scripts**: `plot_[descriptive_name].py` format
- **Outputs**: `[descriptive_name].pdf/.png` format
- **No confusing figure numbers** - names describe content

### Simplified Structure
- **All figures** in single `figures/` directory
- **No nested subdirectories** - easy to find outputs
- **Clear separation** of scripts, figures, and data

### Publication Ready
- **High-quality outputs** in both PDF and PNG formats
- **Consistent styling** via `figure_style.py`
- **Statistical annotations** where appropriate
- **Clear legends and labels** for all plots

## 🚀 Advanced Usage

### Custom Output Locations
```bash
# Specify custom output directory
python Plot_Clean/plot_calibration_comparison.py \
    --csv Plot_Clean/data/all_runs_flat.csv \
    --out /path/to/custom/output
```

### Batch Generation
Create all plots at once:
```bash
#!/bin/bash
# Generate all publication plots
CSV_FILE="Plot_Clean/data/all_runs_flat.csv"
SUBJECTS_FILE="Plot_Clean/data/all_test_subjects_complete.csv"

python Plot_Clean/plot_calibration_comparison.py --csv $CSV_FILE
python Plot_Clean/plot_subjects_vs_minutes.py --csv $CSV_FILE  
python Plot_Clean/plot_training_stage_gains.py --csv $CSV_FILE
python Plot_Clean/plot_robustness_noise_paired.py --csv $CSV_FILE --test-subjects $SUBJECTS_FILE
python Plot_Clean/plot_hyperparameter_sensitivity.py --csv $CSV_FILE
python Plot_Clean/plot_task_granularity_combined.py --csv $CSV_FILE

echo "✅ All publication plots generated in Plot_Clean/figures/"
```

## 📚 Dependencies

All plots use the shared styling system:
- `figure_style.py` - Consistent colors, fonts, and layout
- Standard scientific Python stack: `matplotlib`, `seaborn`, `pandas`, `numpy`
- Statistical analysis: `scipy`, `scikit-learn`

## 🔗 Integration

- **Main README**: [`../README.md`](../README.md) - Project overview
- **CLAUDE.md**: [`../CLAUDE.md`](../CLAUDE.md) - Development commands
- **Training Guide**: [`../ReadMe_training.md`](../ReadMe_training.md) - Training workflows

---

> **Note**: This reorganized structure eliminates the old `fig1/`, `fig3/`, `fig4/` subdirectories in favor of descriptive names and flat organization for better usability and maintenance.