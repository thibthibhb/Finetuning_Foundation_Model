import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(42)
# Load Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
model.proj_out = nn.Identity()  # Remove the projection layer for inference

# Define Classifier
classifier = nn.Sequential(
    Rearrange('b c s p -> b (c s p)'),
    nn.Linear(6000, 5*200),  # Adjust this line to match the input shape
    nn.ELU(),
    nn.Dropout(0.1),
    nn.Linear(5 * 200, 200),
    nn.ELU(),
    nn.Dropout(0.1),
    nn.Linear(200, 5),  # Output layer for 5 classes
).to(device)

# Sleep Stage Labels
sleep_classes = ["Wake", "N1", "N2", "N3", "REM"]

# Load the EEG data (assuming you have the EEG data as .npy file)
# Example path: '.\\Takeda_fine_tuning\\seq_npy.npy'
import os

print(os.getcwd())
eeg_data = np.load('/home/ubuntu/CBraMod/Takeda_fine_tuning/seq_npy/S001.npy')
eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)  # Convert to PyTorch tensor and move to device
print("Shape of EEG data:", eeg_tensor.shape)
output_from_model = model(eeg_tensor)
print("Shape after passing through model:", output_from_model.shape)
real_labels = np.load('/home/ubuntu/CBraMod/Takeda_fine_tuning/labels_npy/S001.npy')
real_labels_tensor = torch.tensor(real_labels, dtype=torch.long).to(device)  # Assuming labels are in long type for classification

# Ensure the data has the correct shape
# Adjust shape if necessary, model expects (batch_size, channels, seq_len, features) 
# For example, if your data needs reshaping, you can do that here

# Forward Pass
logits = classifier(model(eeg_tensor))  # Get raw predictions from model
probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
predictions = torch.argmax(probs, dim=1)  # Get predicted class indices

# Calculate Accuracy
accuracy = (predictions == real_labels_tensor).float().mean()
print(f"Accuracy: {accuracy.item():.4f}")
from sklearn.metrics import f1_score

# Convert predictions to numpy array for comparison
predictions_np = predictions.cpu().numpy()  # Move predictions back to CPU and convert to numpy array
real_labels_np = real_labels_tensor.cpu().numpy()  # Convert real labels to numpy array

# Calculate F1 Score (macro-average, or you can use 'weighted' or 'micro' depending on your needs)
f1 = f1_score(real_labels_np, predictions_np, average='macro')
print(f"F1 Score (Macro): {f1:.4f}")

# Calculate Cohen's Kappa Score
kappa = cohen_kappa_score(real_labels_np, predictions_np)
print(f"Cohen's Kappa Score: {kappa:.4f}")

cm = confusion_matrix(real_labels_np, predictions_np)
print("Confusion Matrix:\n", cm)
# # Print Predictions
# for i, pred in enumerate(predictions):
#     print(f"Sample {i+1}: Predicted Sleep Stage -> {sleep_classes[pred.item()]}")
