# CBraMod Clean Plotting & Analysis System

A structured approach to loading, analyzing, and visualizing CBraMod experiment results following a fair comparison "contract".

## 📋 Overview

This system implements the 5-step process for clean, fair analysis:

1. **Define comparison contract** - Structured requirements for fair comparison
2. **Build analysis cohorts** - Group experiments by comparable conditions  
3. **Normalize run metadata** - Ensure all runs have consistent, joinable metadata
4. **Quality assurance** - Sanity checks and validation before plotting
5. **Generate clean plots** - Publication-ready visualizations

## 🚀 Quick Start

### 1. Load and Structure Runs

First, load all WandB runs and structure them according to the contract:

```bash
python load_and_structure_runs.py --project CBraMod-earEEG-tuning --entity thibaut_hasle-epfl
```

This will:
- Load ALL runs from WandB (with proper pagination)
- Structure metadata according to the comparison contract
- Build analysis cohorts (5-class, 4-class, ICL, scaling)
- Run quality assurance checks
- Save structured data to `Plot_Clean/data/`

### 2. Explore the Data

Explore and understand your loaded data:

```bash
python explore_data.py --data-file Plot_Clean/data/all_runs.csv --create-plots
```

This will:
- Generate comprehensive data summary
- Identify potential issues
- Create exploratory visualization plots
- Save results to `Plot_Clean/outputs/exploration/`

### 3. Create Analysis Plots

(Coming next - this is where we'll implement the clean plotting functions)

## 📁 File Structure

```
Plot_Clean/
├── README.md                    # This file
├── config.py                   # Central configuration
├── load_and_structure_runs.py  # Main data loading script
├── explore_data.py             # Data exploration utility
├── data/                       # Structured data storage
│   ├── all_runs.csv           # All structured runs
│   ├── cohort_5_class.csv     # 5-class sleep staging cohort
│   ├── cohort_4_class.csv     # 4-class sleep staging cohort
│   ├── cohort_icl_comparison.csv # ICL mode comparison cohort
│   └── qa_report.json         # Quality assurance report
├── outputs/                    # Analysis outputs
├── figures/                    # Generated plots
└── requirements.txt           # Python dependencies
```

## 🎯 Comparison Contract

The system enforces a strict comparison contract for fairness:

### Required Fields

**Dataset & Splits:**
- Dataset name, number of training subjects, data fraction
- Split policy (LOSO implied for ear-EEG)

**Preprocessing:**
- Sample rate, window length, preprocessing version string

**Model & Training:**
- Backbone architecture, optimizer, learning rate, epochs
- Head type, batch size, scheduler

**Results:**
- Primary metrics: Cohen's κ, macro F1-score
- Secondary metrics: balanced accuracy, per-stage F1s
- Test set evaluation only (no train/val contamination)

**Compute:**
- Effective tokens processed (windows × epochs)
- Training time, throughput estimates

### Analysis Cohorts

**Cohort A (5-class):** Wake, N1, N2, N3, REM classification
- Compare model architectures, training methods, ICL modes
- Same dataset/preprocessing/splits

**Cohort B (4-class):** Wake, Light, Deep, REM classification  
- Same as above but with 4-class label mapping

**ICL Cohort:** In-context learning comparison
- Compare proto, cnp (DeepSets), set (Set Transformer) modes
- Analyze K-sweep performance for K ∈ {1,5,10,20}

**Scaling Cohort:** Data scaling analysis
- Effect of training data size on performance
- Subject count vs. performance curves

## 🔍 Quality Assurance

The system runs comprehensive QA checks:

- **Pagination completeness** - Ensures all runs loaded
- **Data leakage detection** - Verifies subject split integrity  
- **Class balance consistency** - Checks label distributions
- **Metric consistency** - Validates calculation methods
- **Outlier detection** - Flags anomalous runs

## ⚙️ Configuration

All settings are centralized in `config.py`:

- **Plot styling** - Colors, fonts, figure sizes
- **Metric definitions** - Names, ranges, thresholds
- **Cohort specifications** - Filters and groupings
- **ICL configuration** - Modes, K values, comparisons

## 📊 Key Metrics

**Primary Metrics:**
- **Cohen's κ** - Inter-rater agreement, handles class imbalance
- **Macro F1** - Average per-class F1, equal class weighting

**Secondary Metrics:**
- Balanced accuracy, per-stage F1 scores
- Training efficiency (epochs/sec, GPU-hours)
- Cost estimates ($/inference, $/night)

**Statistical Reporting:**
- Mean ± 95% CI across subjects/seeds
- Bootstrap confidence intervals for per-stage metrics
- Significance testing for comparisons

## 🎨 Plotting Standards

**Figure Quality:**
- Publication-ready 300 DPI PNG output
- Consistent color schemes and styling
- Clear legends, labels, and titles
- Grid lines and error bars where appropriate

**Comparison Fairness:**
- Same query subsets for ICL comparisons
- Paired statistical tests
- Confidence intervals on all metrics
- No cherry-picking of results

## 🛠️ Dependencies

Key packages used:
- `wandb` - Experiment tracking API
- `pandas` - Data manipulation
- `matplotlib` / `seaborn` - Plotting
- `numpy` - Numerical computing
- `scikit-learn` - Statistics and metrics

Install with:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Example Usage

Complete workflow example:

```bash
# 1. Load and structure all runs
python load_and_structure_runs.py

# 2. Explore the data
python explore_data.py --create-plots

# 3. Check the quality report
cat Plot_Clean/data/qa_report.json

# 4. Create publication plots (coming next)
# python create_analysis_plots.py --cohort 5_class --metric test_kappa
```

## 📈 Next Steps

The foundation is now in place. Next implementations:

1. **ICL comparison plots** - κ(CNP) vs κ(proto) scatter plots
2. **Scaling analysis** - Performance vs. data size curves  
3. **Architecture comparison** - Head types, optimizers, etc.
4. **Statistical testing** - Significance tests and effect sizes
5. **Summary tables** - LaTeX-ready result tables

The system is designed to be **extensible** - new cohorts and metrics can be easily added through the configuration system.