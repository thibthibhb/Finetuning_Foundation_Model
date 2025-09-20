<div align="center">

# CBraMod


_A Criss-Cross Brain Foundation Model for EEG Decoding_


[![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![Paper](https://img.shields.io/badge/Paper-ICLR-008B8B)](https://openreview.net/forum?id=NPNUHgHF2w)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/CBraMod)

</div>


<!-- Logo image: figure/CBraMod_logo.png (not included in repository) -->


<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🎯&nbsp;<a href="#-training-methods">Training Methods</a>
    | 📊&nbsp;<a href="#-results--analysis">Results & Analysis</a>
    | 📁&nbsp;<a href="#-project-structure">Project Structure</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>

🔥 NEWS: The paper "_CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding_" has been accepted by ICLR 2025!

## 🔍 About
We propose **CBraMod**, a novel EEG foundation model, for EEG decoding on various clinical and BCI application.
The preprint version of our paper is available at [arXiv](https://arxiv.org/abs/2412.07236). 
The camera-ready version of the paper will be available at [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w).
<!-- Model architecture diagram: figure/model.png (not included in repository) -->



## 🔨 Setup
Install [Python](https://www.python.org/downloads/).

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:
```commandline
pip install -r requirements.txt
``` 


## 🎯 Training Methods

CBraMod supports multiple training approaches optimized for EEG sleep staging:

### 🏃‍♂️ Quick Training (Recommended)
```bash
# Best performance with automatic hyperparameter optimization
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --tune \
    --run_name "best_model"
```

### 🎓 Available Training Methods
- **Standard Fine-tuning**: End-to-end supervised learning
- **Two-Phase Training**: Progressive unfreezing (add `--two_phase_training True --phase1_epochs 3`)
- **In-Context Learning (ICL)**: Few-shot learning (`--icl_mode proto --icl_k 16` or `--icl_mode meta_proto`)
- **Hyperparameter Optimization**: Automated search with Optuna
- **Robustness Analysis**: Training with realistic EEG artifacts

**📚 Complete Training Guide**: See [`ReadMe_training.md`](ReadMe_training.md) for comprehensive examples

### 🚢 Foundation Model Pretraining
```bash
python cbramod/training/pretraining/pretrain_main.py \
    --epochs 40 --batch_size 128 --lr 1e-4
```
**Pre-trained weights available**: [🤗 Hugging Face](https://huggingface.co/weighting666/CBraMod)


## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:
```python
import torch
import torch.nn as nn
from cbramod.models.cbramod import CBraMod
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(torch.load('saved_models/pretrained/pretrained_weights.pth', map_location=device))
model.proj_out = nn.Identity()
classifier = nn.Sequential(
  Rearrange('b c s p -> b (c s p)'),
  nn.Linear(22*4*200, 4*200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(4 * 200, 200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(200, 4),
).to(device)

# mock_eeg.shape = (batch_size, num_of_channels, time_segments, points_per_patch)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)

# logits.shape = (batch_size, num_of_classes)
logits = classifier(model(mock_eeg))
```

## 📊 Results & Analysis

### Performance Metrics
CBraMod achieves state-of-the-art performance on ear-EEG sleep staging:
- **Cohen's κ**: 0.76 (4-class sleep staging)
- **Macro-F1**: 0.78 (balanced across all sleep stages)
- **Sample Efficiency**: 10x fewer labels needed vs. training from scratch

### Generate Analysis Plots
```bash
# Extract experimental data from W&B
python plots/scripts/load_and_structure_runs.py \
    --project CBraMod-earEEG-tuning \
    --entity your-entity \
    --output-dir plots/data/

# Generate publication figures
python plots/scripts/plot_calibration_comparison.py
python plots/scripts/plot_subjects_vs_minutes.py
python plots/scripts/plot_training_stage_gains.py
python plots/scripts/plot_dataset_composition.py --out plots/figures
python plots/scripts/plot_hyperparameter_sensitivity.py
```

**📈 Complete Analysis Guide**: See [`plots/README.md`](plots/README.md) for all visualization options

## 📁 Project Structure

```
CBraMod/
├── 📊 Core Implementation
│   └── cbramod/                # Main CBraMod module
│       ├── models/             # Model architectures (CBraMod, transformers)
│       ├── training/           # All training methods (finetuning, ICL, pretraining)
│       ├── load_datasets/      # Dataset loaders (IDUN_EEG, OpenNeuro)
│       └── utils/              # Utilities (signal processing, memory management)
│
├── 🎯 Training & Models
│   ├── saved_models/           # Pretrained and fine-tuned model weights
│   ├── data/datasets/          # EEG datasets (OpenNeuro, ORP, IDUN_EEG)
│   └── experiments/            # Experiment logs and configurations
│
├── 📈 Analysis & Deployment
│   ├── plots/                  # Publication-ready analysis and figures
│   │   ├── scripts/            # Plotting and analysis scripts
│   │   ├── figures/            # Generated figures (PDF/PNG)
│   │   └── data/               # Plot data files
│   ├── analysis/               # Analysis results and reports
│   ├── scripts/                # Inference and utility scripts
│   └── deploy_prod/            # Production deployment code
│
└── 📚 Documentation
    ├── ReadMe_training.md      # Complete training guide
    ├── CLAUDE.md              # Development commands reference
    └── docs/                  # Additional documentation
```

### Quick Navigation
- **🚀 Start Training**: [`ReadMe_training.md`](ReadMe_training.md)
- **💻 Development**: [`CLAUDE.md`](CLAUDE.md)
- **📊 Analysis**: [`plots/README.md`](plots/README.md)
- **🏗️ Architecture**: [`cbramod/models/`](cbramod/models/)
- **📁 Analysis Results**: [`analysis/README.md`](analysis/README.md)

## 🔗 Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
```bibtex
@inproceedings{wang2025cbramod,
    title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
    author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

## ⭐ Star History
<div align="center">
    <a href="https://star-history.com/#wjq-learning/CBraMod&Date">
        <img src="https://api.star-history.com/svg?repos=wjq-learning/CBraMod&type=Date" style="width: 80%;" />
    </a>
</div>