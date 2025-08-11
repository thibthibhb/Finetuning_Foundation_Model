import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path
import boto3
import io
import os
# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up two levels to reach project root
sys.path.insert(0, str(project_root))

from deploy_prod.code.models.model_for_idun import Model
import time
from scipy.signal import butter, filtfilt, iirnotch, resample
import mne  # for filtering convenience
import logging
# ------------------------- Setup Logging -------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_eeg(signal, sfreq, title, filename):
    """Plot EEG and save to file."""
    duration = 5  # seconds
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(duration * sfreq) / sfreq, signal[:duration * sfreq])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ------------------------- Parameters -------------------------
class Params:
    use_pretrained_weights = False
    foundation_dir = None
    cuda = 0
    num_of_classes = 4 #5

# ------------------------- Setup -------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
device = torch.device("cpu")

# ------------------------- Filtering Functions -------------------------
def apply_bandpass(signal, sfreq, l_freq=0.3, h_freq=35.0):
    return mne.filter.filter_data(signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

def apply_notch(signal, sfreq, freq=50.0):
    return mne.filter.notch_filter(signal, Fs=sfreq, freqs=[freq], verbose=False)

def apply_resample(signal, orig_sfreq, target_sfreq):
    num_samples = int(len(signal) * target_sfreq / orig_sfreq)
    return resample(signal, num_samples)

# ------------------------- Preprocess EEG -------------------------
def preprocess_raw_1d_eeg(signal, ch=1, seq_len=30, epoch_size=200):
    total_points = signal.shape[0]
    total_epochs = total_points // (seq_len * epoch_size)
    usable_points = total_epochs * seq_len * epoch_size
    trimmed = signal[:usable_points]
    reshaped = trimmed.reshape(total_epochs, ch, seq_len, epoch_size)
    return reshaped

# ------------------------- SageMaker Inference API -------------------------
def model_fn(model_dir):
    param = Params()
    model = Model(param).to(device)

    # Replace projection layer BEFORE loading weights
    if hasattr(model.backbone, 'proj_out'):
        model.backbone.proj_out = torch.nn.Identity()
        logger.info("Replaced backbone.proj_out with Identity()")

    ckpt_path = os.path.join(model_dir, "4_class_weights.pth") #"epoch7_acc_0.61463_kappa_0.62701_f1_0.71911.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    logger.info(f"🚩 missing_keys: {missing_keys}")
    logger.info(f"🚩 unexpected_keys: {unexpected_keys}")

    model.eval()
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError("Unsupported content type")

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if isinstance(request_body, str):
        try:
            data = json.loads(request_body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")
    elif isinstance(request_body, dict):
        data = request_body
    else:
        raise ValueError("Unexpected request_body type: " + str(type(request_body)))

    # Extract inputs
    bucket_name = data["bucket_name"]
    userId = data["userId"]
    deviceId = data["deviceId"]
    recordingId = data["recordingId"]
    eeg = f"eeg_{recordingId}.csv"
    orig_sfreq = data.get("orig_sfreq", 250)
    target_sfreq = 200

    key = f"{userId}/{deviceId}/{recordingId}/{eeg}"

    # Fetch EEG from S3
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=key)
    eeg_bytes = response["Body"].read()

    eeg_np = np.genfromtxt(io.BytesIO(eeg_bytes), delimiter=",", names=True)
    print("🏷️ S3 object etag: %s", response["ETag"])
    eeg_np = eeg_np["ch1"] 
    # Extract raw signal
    if eeg_np.ndim > 1:
        raw_signal = eeg_np[:, 1]
    else:
        raw_signal = eeg_np

    print("Before filtering:", raw_signal.min(), raw_signal.max(), raw_signal.shape, raw_signal.dtype, raw_signal.mean(), raw_signal.std())

    # Plot raw
    #plot_eeg(raw_signal, orig_sfreq, "Raw EEG", f"{recordingId}_raw.png")

    # Preprocessing
    raw_signal = apply_notch(raw_signal, sfreq=orig_sfreq, freq=50)
    print("notch filtering:", raw_signal.min(), raw_signal.max(), raw_signal.shape, raw_signal.dtype, raw_signal.mean(), raw_signal.std())
    raw_signal = apply_bandpass(raw_signal, sfreq=orig_sfreq, l_freq=0.3, h_freq=35)
    print("bandpass filtering:", raw_signal.min(), raw_signal.max(), raw_signal.shape, raw_signal.dtype, raw_signal.mean(), raw_signal.std())
    raw_signal = apply_resample(raw_signal, orig_sfreq, target_sfreq)
    print("resample  filtering:", raw_signal.min(), raw_signal.max(), raw_signal.shape, raw_signal.dtype, raw_signal.mean(), raw_signal.std())

    # Reshape into epochs: [n_epochs, ch=1, 30, 200]
    eeg = preprocess_raw_1d_eeg(raw_signal)

    # Stack clean epochs
    eeg_tensor = torch.tensor(np.stack(eeg), dtype=torch.float32).to(device)

    # Optionally log shape
    logger.info(f"Tensor shape: {eeg_tensor.shape}")  # [batch, 1, 30, 200]
    return eeg_tensor


def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
    logger.info(f"Predictions: {predictions}")
    return predictions.cpu().numpy()

def output_fn(prediction, accept='application/json'):
    return json.dumps({"prediction": prediction.tolist()})

# ------------------------- Local Testing -------------------------
if __name__ == "__main__":
    from pathlib import Path

    model_dir = Path("deploy_prod")    
    model = model_fn(model_dir)
    ckpt_path = model_dir / "4_class_weights.pth" #"epoch7_acc_0.61463_kappa_0.62701_f1_0.71911.pth"

    sample_input = {
        "bucket_name": "idn-dev-raw-recordings-bucket",
        "userId": "036d5eb6-e177-475a-bb64-5c09e51062a7",
        "deviceId": "F9-79-78-54-CA-15",
        "recordingId": "1707409115815",
        "orig_sfreq": 250
    }
    
    # sample_input = {
    #     "bucket_name": "idn-prod-raw-recordings-bucket",
    #     "userId": "3f5dc485-49f8-434e-baa9-bb6be6551a13",
    #     "deviceId": "F3-49-79-D6-D3-73",
    #     "recordingId": "1704880350000", #1713871572620
    #     "orig_sfreq": 250
    # }
    
    # sample_input = {
    #     "bucket_name": "idn-dev-raw-recordings-bucket",
    #     "userId": "572fcb78-9bc2-437f-b315-89617a2d6778",
    #     "deviceId": "FF-00-00-00-00-04",
    #     "recordingId": "1721303866588",
    #     "orig_sfreq": 250
    # }
    eeg_tensor = input_fn(sample_input)
    preds = predict_fn(eeg_tensor, model)

    print("Local Predictions:", preds, 'lenght:', len(preds), "unqiue:", np.unique(preds))

    
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from pathlib import Path
# import boto3
# import io

# # Add project root to path
# current_dir = Path(__file__).parent
# project_root = current_dir.parent.parent  # Go up two levels to reach project root
# sys.path.insert(0, str(project_root))

# from cbramod.models.model_for_idun import Model
# import time
# from scipy.signal import butter, filtfilt, iirnotch, resample
# import mne  # for filtering convenience

# # ------------------------- Parameters -------------------------
# class Params:
#     use_pretrained_weights = False
#     foundation_dir = None
#     cuda = 0
#     num_of_classes = 5

# # ------------------------- Setup -------------------------
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# setup_seed(42)
# device = torch.device("cpu")

# # ------------------------- Filtering Functions -------------------------
# def apply_bandpass(signal, sfreq, l_freq=0.3, h_freq=35.):
#     return mne.filter.filter_data(signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

# def apply_notch(signal, sfreq, freq=50.0):
#     return mne.filter.notch_filter(signal, Fs=sfreq, freqs=[freq], verbose=False)

# def apply_resample(signal, orig_sfreq, target_sfreq):
#     num_samples = int(len(signal) * target_sfreq / orig_sfreq)
#     return resample(signal, num_samples)

# # ------------------------- Preprocess EEG -------------------------
# def preprocess_raw_1d_eeg(signal, ch=1, seq_len=30, epoch_size=200):
#     """
#     Reshape a 1D EEG array into (N, ch, seq_len, epoch_size)
#     """
#     total_points = signal.shape[0]
#     total_epochs = total_points // (seq_len * epoch_size)
#     usable_points = total_epochs * seq_len * epoch_size
#     trimmed = signal[:usable_points]
#     reshaped = trimmed.reshape(total_epochs, ch, seq_len, epoch_size)
#     return reshaped

# # ------------------------- Load and Filter EEG -------------------------
# Start = time.time()

# bucket_name = 'idn-dev-raw-recordings-bucket'

# # S3 path structure
# userId = "026e854e-4bac-41ae-97b3-3b39c66cef89"
# deviceId = "49-44-55-4E-00-02"
# recordingId = "1713871572620"
# eeg = f"eeg_{recordingId}.csv"
# key = f"{userId}/{deviceId}/{recordingId}/{eeg}"

# # # Download EEG data from S3
# s3_client = boto3.client('s3')
# response = s3_client.get_object(Bucket=bucket_name, Key=key)
# csv_content = response['Body'].read().decode('utf-8')
# raw_data = np.loadtxt(io.StringIO(csv_content), delimiter=',', skiprows=1)

# # Load EEG data from local file
# # eeg_data_path = project_root / "data/datasets/final_dataset/2017_Open_N/eeg_data_npy/2017-sub-002_ses-001_2.npy"
# # raw_data = np.load(eeg_data_path)

# # Assumes EEG is in second column or flatten if needed
# # EEG column selection
# if raw_data.ndim > 1:
#     raw_signal = raw_data[:, 1]  # consistently pick EEG column
# else:
#     raw_signal = raw_data

# # Filter settings
# orig_sfreq = 250  # replace with your actual EEG sample rate
# target_sfreq = 200  # target sample rate for resampling

# # Apply filters
# raw_signal = apply_notch(raw_signal, sfreq=orig_sfreq, freq=50)
# raw_signal = apply_bandpass(raw_signal, sfreq=orig_sfreq, l_freq=0.3, h_freq=35)
# raw_signal = apply_resample(raw_signal, orig_sfreq=orig_sfreq, target_sfreq=target_sfreq)

# # Preprocess to tensor shape
# raw_eeg = preprocess_raw_1d_eeg(raw_signal)
# eeg_tensor = torch.tensor(raw_eeg, dtype=torch.float32).to(device)
# print("Tensor shape:", eeg_tensor.shape)

# # ------------------------- Inference Function -------------------------
# def evaluate(use_pretrained):
#     param = Params()
#     param.use_pretrained_weights = False  # Don't load backbone weights
#     model = Model(param).to(device)

#     # Load only the classifier weights
#     checkpoint_path = str(project_root / "deploy_prod" / "epoch7_acc_0.61463_kappa_0.62701_f1_0.71911.pth")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint, strict=False)
    
#     missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
#     print("🚩 missing_keys:", missing_keys)
#     print("🚩 unexpected_keys:", unexpected_keys)

#     if hasattr(model.backbone, 'proj_out'):
#         model.backbone.proj_out = torch.nn.Identity()

#     model.eval()
#     with torch.no_grad():
#         logits = model(eeg_tensor)
#         probs = F.softmax(logits, dim=1)
#         predictions = torch.argmax(probs, dim=1)

#     return predictions.cpu().numpy()

# # ------------------------- Run Evaluation -------------------------
# predictions_orp = evaluate(use_pretrained=True)
# print("Predictions (ORP):", predictions_orp)
# print(len(predictions_orp), "predictions made.")
# end = time.time()
# print(f"✅ Preprocessing and inference took {end - Start:.2f} seconds.")


