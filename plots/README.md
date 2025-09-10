# Plots

This directory contains all plotting functionality for CBraMod visualization and analysis.

## Structure

- `scripts/` - All plotting scripts
  - `deployment/` - Deployment-specific visualization scripts
  - `plot_*.py` - Main plotting scripts for figures
- `figures/` - Generated figure outputs (PDF/PNG)
- `data/` - Plot data files (CSV)
- `style/` - Figure styling configuration (`figure_style.py`)

## Usage

### Generate Plot Data (Run Once)
```bash
python plots/scripts/load_and_structure_runs.py \
    --project CBraMod-earEEG-tuning \
    --entity thibaut_hasle-epfl \
    --output-dir plots/data/
```

### Generate Individual Figures
```bash
# From repository root
python plots/scripts/plot_hyperparameter_sensitivity.py
python plots/scripts/plot_calibration_comparison.py
python plots/scripts/plot_heads_vs_weights_comparison.py

# Deployment plots
python plots/scripts/deployment/create_deployment_visualization.py
python plots/scripts/deployment/create_improved_deployment_plots.py
```

## Figure Output

All figures are saved to `plots/figures/` in both PDF and PNG formats.

## Style Configuration

The `plots/style/figure_style.py` module provides consistent styling across all plots including:
- Color palettes (CBraMod blue, YASA colors, etc.)
- Font settings optimized for print
- Statistical visualization helpers