import torch
import torch.nn.functional as F
import numpy as np
from models.model_for_idun import Model
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
import json

# ------------------------- Parameters -------------------------
class Params:
    use_pretrained_weights = False               # We are loading finetuned weights
    foundation_dir = None                        # Only used if using pretrained
    cuda = 0
    num_of_classes = 5                           # Must match the checkpoint

# ------------------------- Setup -------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------- Load Model -------------------------
param = Params()
model = Model(param).to(device)

checkpoint_path = "./saved_models/epoch2_acc_0.56190_kappa_0.56193_f1_0.68224.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load with strict=False and print key issues
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
print("✅ Checkpoint loaded")
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

# Force backbone.proj_out = Identity if that was used during fine-tuning
if hasattr(model.backbone, 'proj_out'):
    model.backbone.proj_out = torch.nn.Identity()

model.eval()

# ------------------------- Load Data -------------------------
eeg_data = np.load('Final_dataset/ORP/eeg_data_npy/S009_night2.npy')   # [N, 1, 30, 200] expected
real_labels = np.load('Final_dataset/ORP/label_npy/S009_night2.npy')   # [N]

print("Original EEG shape:", eeg_data.shape)
eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)
real_labels_tensor = torch.tensor(real_labels, dtype=torch.long).to(device)

# Add channel dimension if input is 3D (assumed: [N, 30, 200])
if eeg_tensor.ndim == 3:
    eeg_tensor = eeg_tensor.unsqueeze(1)  # → [N, 1, 30, 200]

print("EEG tensor shape:", eeg_tensor.shape)
print("Real label shape:", real_labels_tensor.shape)

# ------------------------- Inference -------------------------
with torch.no_grad():
    logits = model(eeg_tensor)                   # → [N, num_classes]
    probs = F.softmax(logits, dim=1)             # → [N, num_classes]
    predictions = torch.argmax(probs, dim=1)     # → [N]

# ------------------------- Save Results -------------------------
predictions_np = predictions.cpu().numpy()
real_labels_np = real_labels_tensor.cpu().numpy()

np.save("predicted_labels.npy", predictions_np)
np.save("real_labels.npy", real_labels_np)

metrics = {
    "f1_macro": f1_score(real_labels_np, predictions_np, average='macro'),
    "cohen_kappa": cohen_kappa_score(real_labels_np, predictions_np),
    "confusion_matrix": confusion_matrix(real_labels_np, predictions_np).tolist(),
}

with open("metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Inference complete. Metrics saved to metrics_summary.json")
print("📊 Prediction counts:", np.bincount(predictions_np))
