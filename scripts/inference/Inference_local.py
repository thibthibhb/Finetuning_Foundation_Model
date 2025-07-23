import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up two levels to reach project root
sys.path.insert(0, str(project_root))

from cbramod.models.model_for_idun import Model
import time
from scipy.signal import butter, filtfilt, iirnotch, resample
import mne  # for filtering convenience

# ------------------------- Parameters -------------------------
class Params:
    use_pretrained_weights = False
    foundation_dir = None
    cuda = 0
    num_of_classes = 4

# ------------------------- Setup -------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
device = torch.device("cpu")

# ------------------------- Filtering Functions -------------------------
def apply_bandpass(signal, sfreq, l_freq=0.3, h_freq=35.):
    return mne.filter.filter_data(signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

def apply_notch(signal, sfreq, freq=50.0):
    return mne.filter.notch_filter(signal, Fs=sfreq, freqs=[freq], verbose=False)

def apply_resample(signal, orig_sfreq, target_sfreq):
    num_samples = int(len(signal) * target_sfreq / orig_sfreq)
    return resample(signal, num_samples)

# ------------------------- Preprocess EEG -------------------------
def preprocess_raw_1d_eeg(signal, ch=1, seq_len=30, epoch_size=200):
    """
    Reshape a 1D EEG array into (N, ch, seq_len, epoch_size)
    """
    total_points = signal.shape[0]
    total_epochs = total_points // (seq_len * epoch_size)
    usable_points = total_epochs * seq_len * epoch_size
    trimmed = signal[:usable_points]
    reshaped = trimmed.reshape(total_epochs, ch, seq_len, epoch_size)
    return reshaped

# ------------------------- Load and Filter EEG -------------------------
Start = time.time()
eeg_path = "data/datasets/test_data/eeg_1740519564613.csv"
raw_data = np.loadtxt(eeg_path, delimiter=',', skiprows=1)

# Assumes EEG is in second column
if raw_data.ndim > 1:
    raw_signal = raw_data[:, 1]
else:
    raw_signal = raw_data

# Filter settings
orig_sfreq = 250  # replace with your actual EEG sample rate
target_sfreq = 100

# Apply filters
raw_signal = apply_notch(raw_signal, sfreq=orig_sfreq, freq=50)
raw_signal = apply_bandpass(raw_signal, sfreq=orig_sfreq, l_freq=0.3, h_freq=35)
raw_signal = apply_resample(raw_signal, orig_sfreq=orig_sfreq, target_sfreq=target_sfreq)

# Preprocess to tensor shape
raw_eeg = preprocess_raw_1d_eeg(raw_signal)
eeg_tensor = torch.tensor(raw_eeg, dtype=torch.float32).to(device)

# ------------------------- Inference Function -------------------------
def evaluate(use_pretrained):
    param = Params()
    param.use_pretrained_weights = use_pretrained
    param.foundation_dir = "./dummy.pth"  # Prevent foundation load
    model = Model(param).to(device)

    checkpoint_path = (
        "deploy_prod/4_class_weights.pth"
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

# ------------------------- Run Evaluation -------------------------
predictions_orp = evaluate(use_pretrained=False)
print("Predictions (ORP):", predictions_orp)
end = time.time()
print(f"✅ Preprocessing and inference took {end - Start:.2f} seconds.")
