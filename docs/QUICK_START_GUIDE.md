# CBraMod Enhanced Pipeline - Quick Start Guide

## ✅ **Issue Fixed: Configuration Validation**

The configuration validation error has been resolved! The validator now supports all enhanced scheduler types.

## 🚀 **Ready to Use - Quick Start**

### **1. Test Everything Works**

```bash
# Quick validation test
python run_optimized_finetuning.py --check-only

# Should output: ✅ Prerequisites check passed
```

### **2. Set Up Your Pretrained Weights (Optional)**

```bash
# If you have pretrained weights, place them at:
mkdir -p pretrained_weights
# Copy your weights to: pretrained_weights/pretrained_weights.pth

# OR disable pretrained weights in config
# Edit config/environments/finetuning_optimized.yaml:
# model.pretrained_weights.enabled: false
```

### **3. Run Optimized Finetuning**

```bash
# Quick test run (2 epochs)
python run_optimized_finetuning.py \
  --config-env finetuning_optimized \
  --epochs 2 \
  --batch-size 32

# Full optimized run
python run_optimized_finetuning.py \
  --config-env finetuning_optimized \
  --adaptive-lr \
  --weighted-sampler

# With custom hyperparameters
python run_optimized_finetuning.py \
  --config-env finetuning_optimized \
  --epochs 150 \
  --lr 0.001 \
  --scheduler cosine_warmup
```

## 📊 **What's Working Now**

✅ **Configuration validation** - All scheduler types supported  
✅ **Data pipeline integration** - Quality checks and versioning  
✅ **Enhanced finetuning** - Smart pretrained weight loading  
✅ **Experiment tracking** - Comprehensive logging and monitoring  
✅ **Reproducibility** - Full environment capture  

## 🛠️ **Available Configurations**

- `development` - Basic development settings
- `production` - Production-ready settings  
- `finetuning_optimized` - **Optimized for best finetuning results**

## 🎯 **Key Features**

### **Smart Pretrained Weight Loading**
- Handles layer mismatches intelligently
- Adaptive learning rates for different layer types
- Graceful fallback to random initialization

### **Enhanced Training Strategies**
- Multiple scheduler types: `cosine`, `cosine_warmup`, `plateau`, `exponential`
- Early stopping with configurable patience
- Weighted sampling for class balancing
- Mixed precision training support

### **Data Quality Assurance**
- Comprehensive EEG-specific validation
- Real-time quality monitoring
- Automated data versioning with DVC
- Complete lineage tracking

## 🏃‍♂️ **Recommended First Run**

```bash
# 1. Test the setup
python run_optimized_finetuning.py --check-only

# 2. Quick test run
python run_optimized_finetuning.py \
  --config-env finetuning_optimized \
  --epochs 2 \
  --batch-size 32

# 3. If test works, run full training
python run_optimized_finetuning.py --config-env finetuning_optimized
```

## 📋 **Expected Output**

```
🚀 CBraMod Optimized Finetuning Pipeline
==========================================

📊 Configuration Summary:
  Environment: finetuning_optimized
  Epochs: 100
  Batch size: 64
  Learning rate: 0.0005
  Pretrained weights: True/False
  Adaptive LR: True
  Weighted sampler: True
  Scheduler: cosine_warmup

🔧 Initializing data pipeline components...
✅ Data pipeline components initialized
📊 Loading dataset with enhanced data pipeline...
📊 Data Quality Summary:
  Files processed: X
  Files passed: X
  Files failed: 0
  Warnings: 0
  Errors: 0

🚀 Starting enhanced training...
✅ Enhanced training completed successfully!

🎉 Training Completed Successfully!
📊 Results:
  Kappa Score: 0.XXXX
  Accuracy:    0.XXXX  
  F1-Score:    0.XXXX
```

## 🎉 **You're Ready!**

The enhanced CBraMod pipeline is now fully functional and ready to deliver the best finetuning results. The configuration validation issue has been fixed and all components are working together seamlessly.

**Start with:** `python run_optimized_finetuning.py --config-env finetuning_optimized`