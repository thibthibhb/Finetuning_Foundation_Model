import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.model_for_idun import Model
import time
# ------------------------- Parameters -------------------------
class Params:
    use_pretrained_weights = False
    foundation_dir = None
    cuda = 0
    num_of_classes = 5

# ------------------------- Setup -------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
device = torch.device("cpu")

# ------------------------- Preprocess EEG -------------------------
def preprocess_raw_1d_eeg(signal, ch=1, seq_len=30, epoch_size=200):
    """
    Reshape a 1D EEG array into (N, 1, 30, 200) format.
    """
    total_points = signal.shape[0]
    total_epochs = total_points // (seq_len * epoch_size)
    usable_points = total_epochs * seq_len * epoch_size
    trimmed = signal[:usable_points]
    reshaped = trimmed.reshape(total_epochs, ch, seq_len, epoch_size)
    return reshaped
Start = time.time()  # Start timer for preprocessing
# ------------------------- Load EEG -------------------------
eeg_path = "Raw_data_Test/eeg_1740519564613.csv"
raw_signal = np.loadtxt(eeg_path, delimiter=',', skiprows=1)  # Skip header

# If multiple columns, flatten to 1D
if raw_signal.ndim > 1:
    raw_signal = raw_signal[:, 1]  # Assumes second column holds EEG

raw_eeg = preprocess_raw_1d_eeg(raw_signal)  # Shape: (N, 1, 30, 200)
eeg_tensor = torch.tensor(raw_eeg, dtype=torch.float32).to(device)

# ------------------------- Inference Function -------------------------
def evaluate(use_pretrained):
    param = Params()
    param.use_pretrained_weights = use_pretrained
    model = Model(param).to(device)

    checkpoint_path = (
        "./saved_models/epoch9_acc_0.63467_kappa_0.63408_f1_0.73217.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    if hasattr(model.backbone, 'proj_out'):
        model.backbone.proj_out = torch.nn.Identity()

    model.eval()
    with torch.no_grad():
        logits = model(eeg_tensor)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)

    return predictions.cpu().numpy()

# ------------------------- Run Evaluations -------------------------
predictions_orp = evaluate(use_pretrained=False)
end = time.time()  # End timer for preprocessing
print(f"Preprocessing and inference took {end - Start:.2f} seconds.")
