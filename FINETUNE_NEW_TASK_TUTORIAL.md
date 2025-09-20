# Simple Tutorial: Finetuning CBraMod on a New Task

This tutorial shows the **minimal steps** to finetune CBraMod on your own downstream task.

## Overview
To adapt CBraMod to a new task, you need to:
1. Create a dataset loader for your data
2. Modify the model output layer for your classification task
3. Run training with the new setup

## Step 1: Create Your Dataset Loader

Create `cbramod/load_datasets/your_dataset.py`:

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from cbramod.utils.util import to_tensor

class YourDataset(Dataset):
    def __init__(self, data_paths, num_classes=3):  # Change num_classes as needed
        self.samples = []
        self.num_classes = num_classes

        # Load your data files
        for data_path, label_path in data_paths:
            # Load your EEG data (shape: [n_epochs, n_channels, epoch_length])
            data = np.load(data_path)  # Adjust loading method as needed
            labels = np.load(label_path)  # Your labels (0, 1, 2, ...)

            # Add each epoch as a sample
            for i in range(len(data)):
                # Reshape to match CBraMod input: [n_channels, seq_len, patch_size]
                # For single channel: [1, 30, 200] where 30*200=6000 samples
                epoch = self.reshape_epoch(data[i])
                self.samples.append((epoch, int(labels[i])))

    def reshape_epoch(self, epoch_data):
        """Reshape your EEG epoch to [1, 30, 200] format."""
        # Example: if epoch_data is [6000] samples, reshape to [1, 30, 200]
        if epoch_data.shape == (6000,):
            return epoch_data.reshape(1, 30, 200)
        # Add your reshaping logic here based on your data format
        return epoch_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch, label = self.samples[idx]
        # Return: (data_tensor, label_tensor, subject_id)
        return to_tensor(epoch), torch.tensor(label, dtype=torch.long), f"subj_{idx}"

class LoadYourDataset:
    def __init__(self, params):
        self.params = params
        self.data_paths = self.get_data_paths()

    def get_data_paths(self):
        """Return list of (data_file, label_file) tuples."""
        data_dir = self.params.datasets_dir  # e.g., "path/to/your/data"
        paths = []

        # Example: load all .npy files in data_dir
        for file in os.listdir(data_dir):
            if file.endswith('_data.npy'):
                data_path = os.path.join(data_dir, file)
                label_path = os.path.join(data_dir, file.replace('_data.npy', '_labels.npy'))
                if os.path.exists(label_path):
                    paths.append((data_path, label_path))

        return paths

    def get_all_pairs(self):
        return self.data_paths
```

## Step 2: Modify Model for Your Task

Create `cbramod/models/your_model.py`:

```python
import torch
import torch.nn as nn
from cbramod.models.cbramod import CBraMod

class YourTaskModel(nn.Module):
    def __init__(self, num_classes=3, use_pretrained=True):
        super().__init__()

        # Load CBraMod backbone
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8
        )

        # Load pretrained weights
        if use_pretrained:
            pretrained_path = "saved_models/pretrained/pretrained_weights.pth"
            self.backbone.load_state_dict(torch.load(pretrained_path))
            # Remove the original projection layer
            self.backbone.proj_out = nn.Identity()

        # Add your task-specific head
        self.classifier = nn.Sequential(
            nn.Linear(1 * 30 * 200, 512),  # Flatten: [1, 30, 200] -> [6000]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)    # Your number of classes
        )

    def forward(self, x):
        # x shape: [batch_size, 1, 30, 200]
        features = self.backbone(x)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten
        output = self.classifier(features)
        return output

```

## Step 3: Create Training Script

Create `train_your_task.py`:

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

# Import your components
from cbramod.load_datasets.your_dataset import YourDataset, LoadYourDataset
from cbramod.models.your_model import YourTaskModel

class SimpleParams:
    def __init__(self):
        self.datasets_dir = "path/to/your/data"  # Update this path
        self.num_classes = 3                     # Update number of classes
        self.batch_size = 32
        self.epochs = 50
        self.lr = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model():
    params = SimpleParams()

    # 1. Load dataset
    print("Loading dataset...")
    data_loader_helper = LoadYourDataset(params)
    data_paths = data_loader_helper.get_all_pairs()
    dataset = YourDataset(data_paths, num_classes=params.num_classes)

    # Simple train/test split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # 2. Load model
    print("Loading model...")
    model = YourTaskModel(num_classes=params.num_classes)
    model.freeze_backbone()  # Start with frozen backbone
    model = model.to(params.device)

    # 3. Setup training
    optimizer = Adam(model.parameters(), lr=params.lr)
    criterion = F.cross_entropy

    # 4. Training loop
    print("Starting training...")
    model.train()
    for epoch in range(params.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, labels, _) in enumerate(train_loader):
            data, labels = data.to(params.device), labels.to(params.device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{params.epochs}: Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%')

        # Unfreeze backbone after 10 epochs for full fine-tuning
        if epoch == 10:
            print("Unfreezing backbone...")
            model.unfreeze_backbone()
            optimizer = Adam(model.parameters(), lr=params.lr/10)  # Lower LR for backbone

    # 5. Test the model
    print("Testing...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data, labels = data.to(params.device), labels.to(params.device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # 6. Save model
    torch.save(model.state_dict(), 'your_task_model.pth')
    print("Model saved!")

if __name__ == "__main__":
    train_model()
```

## Step 4: Run Training

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure you have pretrained weights
ls saved_models/pretrained/pretrained_weights.pth

# Run your training
python train_your_task.py
```

## Key Points

1. **Data Format**: CBraMod expects input shape `[batch, channels, seq_len, patch_size]` = `[B, 1, 30, 200]`
2. **Pretrained Weights**: Always start with pretrained CBraMod backbone
3. **Two-Phase Training**: Start with frozen backbone, then unfreeze for full fine-tuning
4. **Classes**: Modify `num_classes` and labels (0, 1, 2, ...) based on your task

## Your Data Requirements

- EEG epochs of 6000 samples (30 seconds at 200Hz)
- Labels as integers: 0, 1, 2, ... for your classes
- Data organized as numpy arrays

That's it! This minimal setup will get CBraMod working on your new task.