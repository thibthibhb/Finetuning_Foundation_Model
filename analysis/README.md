# Analysis Scripts

This directory contains all analysis scripts organized by category.

## Structure

- `deployment/` - Deployment-related analysis scripts
- `subjects/` - Subject-specific analysis (kappa, extraction, investigation)  
- `experiments/` - Experiment-level analysis (class distributions, etc.)
- `results/` - All CSV output files with results prefix

## Usage

Run analysis scripts from the repository root:

```bash
python analysis/subjects/analyze_real_subject_kappa.py
python analysis/deployment/deployment_analysis.py  
python analysis/experiments/analyze_class_distributions.py
```

## Results

All analysis results are saved to `results/results_*.csv` with standardized naming.