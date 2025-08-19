<div align="center">

# CBraMod


_A Criss-Cross Brain Foundation Model for EEG Decoding_


[![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![Paper](https://img.shields.io/badge/Paper-ICLR-008B8B)](https://openreview.net/forum?id=NPNUHgHF2w)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/CBraMod)

</div>


<div align="center">
<img src="figure/CBraMod_logo.png" style="width: 15%;" />
</div>


<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚢&nbsp;<a href="#-how-to-pretrain">How to Pretrain</a>
    | ⛵&nbsp;<a href="#-how-to-finetune">How to Finetune</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>

🔥 NEWS: The paper "_CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding_" has been accepted by ICLR 2025!

## 🔍 About
We propose **CBraMod**, a novel EEG foundation model, for EEG decoding on various clinical and BCI application.
The preprint version of our paper is available at [arXiv](https://arxiv.org/abs/2412.07236). 
The camera-ready version of the paper will be available at [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w).
<div align="center">
<img src="figure/model.png" style="width:100%;" />
</div>



## 🔨 Setup
Install [Python](https://www.python.org/downloads/).

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:
```commandline
pip install -r requirements.txt
``` 


## 🚢 How to Pretrain
You can pretrain CBraMod on our pretraining dataset or your custom pretraining dataset using the following code:
```commandline
python cbramod/training/pretraining/pretrain_main.py
```
We have released a pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).

## ⛵ How to Finetune

### Standard Fine-tuning
```commandline
# 4-class training with v1 mapping (recommended)
python cbramod/training/finetuning/finetune_main.py \
    --downstream_dataset IDUN_EEG \
    --datasets_dir data/datasets/final_dataset \
    --datasets ORP,2023_Open_N,2019_Open_N,2017_Open_N \
    --use_pretrained_weights True \
    --model_dir "./saved_models" \
    --tune \
    --num_of_classes 4 \
    --multi_eval \
    --label_mapping_version v1
```

### Two-Phase Training (Advanced)
```commandline
python cbramod/training/finetuning/finetune_main.py \
    --two_phase_training True \
    --epochs 15 \
    --num_of_classes 4
```

### Inference
```commandline
python scripts/inference/Inference_local.py
```


## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:
```python
import torch
import torch.nn as nn
from models.cbramod import CBraMod
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

## 📁 Directory Structure

The codebase has been reorganized for clarity and maintainability:

```
CBraMod/
├── cbramod/                    # Core module (main implementation)
│   ├── models/                 # Model architectures (cbramod.py, criss_cross_transformer.py)
│   ├── load_datasets/          # Dataset loaders (idun_datasets.py, enhanced_dataset.py)
│   ├── preprocessing/          # EEG preprocessing pipelines
│   ├── training/               # Training scripts (finetuning/, pretraining/)
│   └── utils/                  # Utilities (signaltools.py, memory_manager.py)
├── saved_models/               # Consolidated model storage
│   ├── pretrained/            # Foundation model weights
│   ├── finetuned/             # Best performing fine-tuned models
│   └── production/            # Production-ready models
├── experiments/               # Experiment tracking and results
│   ├── logs/                  # Training logs
│   ├── configs/               # Reproducibility configurations
│   └── results/               # Analysis results and figures
├── data/                      # EEG datasets
│   └── datasets/              # Processed dataset files
├── deploy_prod/               # Production deployment code
├── scripts/                   # Utility scripts (inference, setup)
├── docs/                      # Documentation
├── Plot/                      # Research analysis plots
└── Plot_Clean/                # Publication-ready plots
```

### Key Features
- **Clean Structure**: Consolidated model storage and experiment tracking
- **No Duplicates**: Removed scattered weights and logs across multiple directories  
- **Easy Navigation**: Clear separation of concerns with logical grouping
- **Production Ready**: Dedicated deployment and production model directories

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