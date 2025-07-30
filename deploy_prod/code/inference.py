# inference.py
import torch
import torch.nn.functional as F
import numpy as np
from models.model_for_idun import Model
from scipy.signal import resample
import mne
import json
import boto3
import os
import io
# ------------------------- Parameters -------------------------
class Params:
    use_pretrained_weights = False
    foundation_dir = None
    cuda = 0
    num_of_classes = 4

# ------------------------- Filtering -------------------------
def apply_bandpass(signal, sfreq, l_freq=0.3, h_freq=35.):
    return mne.filter.filter_data(signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

def apply_notch(signal, sfreq, freq=50.0):
    return mne.filter.notch_filter(signal, Fs=sfreq, freqs=[freq], verbose=False)

def apply_resample(signal, orig_sfreq, target_sfreq):
    num_samples = int(len(signal) * target_sfreq / orig_sfreq)
    return resample(signal, num_samples)

def preprocess_raw_1d_eeg(signal, ch=1, seq_len=30, epoch_size=200):
    total_points = signal.shape[0]
    total_epochs = total_points // (seq_len * epoch_size)
    usable_points = total_epochs * seq_len * epoch_size
    trimmed = signal[:usable_points]
    reshaped = trimmed.reshape(total_epochs, ch, seq_len, epoch_size)
    return reshaped

# ------------------------- SageMaker Inference API -------------------------

def model_fn(model_dir):
    device = torch.device("cpu")
    param = Params()
    model = Model(param).to(device)
    ckpt_path = os.path.join(model_dir, "4_class_weights.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    if hasattr(model.backbone, 'proj_out'):
        model.backbone.proj_out = torch.nn.Identity()

    model.eval()
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError("Unsupported content type")

    # Defensive: decode only if needed
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
    if bucket_name is None:
        raise EnvironmentError("RAW_RECORDINGS_BUCKET not set in environment variables.")
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

    eeg_np = np.loadtxt(io.BytesIO(eeg_bytes), delimiter=",", skiprows=1)

    if eeg_np.ndim > 1:
        raw_signal = eeg_np[:, 1]
    else:
        raw_signal = eeg_np

    raw_signal = apply_notch(raw_signal, sfreq=orig_sfreq, freq=50)
    raw_signal = apply_bandpass(raw_signal, sfreq=orig_sfreq, l_freq=0.3, h_freq=35)
    raw_signal = apply_resample(raw_signal, orig_sfreq, target_sfreq)

    eeg = preprocess_raw_1d_eeg(raw_signal)
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

    return eeg_tensor




def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
    print("Predictions:", predictions)
    return predictions.cpu().numpy()

def output_fn(prediction, accept='application/json'):
    return json.dumps({"prediction": prediction.tolist()})
















# import torch
# import torch.nn.functional as F
# import numpy as np
# from models.model_for_idun import Model
# from scipy.signal import resample
# import mne
# import json
# import boto3
# import os
# import io

# # ------------------------- Parameters -------------------------
# class Params:
#     use_pretrained_weights = False
#     foundation_dir = None
#     cuda = 0
#     num_of_classes = 5

# # ------------------------- Filtering -------------------------
# def apply_bandpass(signal, sfreq, l_freq=0.3, h_freq=35.):
#     return mne.filter.filter_data(signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)

# def apply_notch(signal, sfreq, freq=50.0):
#     return mne.filter.notch_filter(signal, Fs=sfreq, freqs=[freq], verbose=False)

# def apply_resample(signal, orig_sfreq, target_sfreq):
#     num_samples = int(len(signal) * target_sfreq / orig_sfreq)
#     return resample(signal, num_samples)

# def preprocess_raw_1d_eeg(signal, ch=1, seq_len=30, epoch_size=200):
#     total_points = signal.shape[0]
#     total_epochs = total_points // (seq_len * epoch_size)
#     usable_points = total_epochs * seq_len * epoch_size
#     trimmed = signal[:usable_points]
#     reshaped = trimmed.reshape(total_epochs, ch, seq_len, epoch_size)
#     return reshaped

# # ------------------------- SageMaker Inference API -------------------------

# def model_fn(model_dir):
#     device = torch.device("cpu")
#     param = Params()
#     model = Model(param).to(device)

#     ckpt_path = os.path.join(model_dir, "model_finetune.pth")
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(checkpoint, strict=False)

#     if hasattr(model.backbone, 'proj_out'):
#         model.backbone.proj_out = torch.nn.Identity()

#     model.eval()
#     return model

# def input_fn(request_body, content_type='application/json'):
#     if content_type != 'application/json':
#         raise ValueError("Unsupported content type")

#     data = json.loads(request_body)

#     # Expects: {"s3_uri": "s3://bucket-name/path/to/eeg.csv", "orig_sfreq": 250}
#     s3_uri = data["s3_uri"]
#     orig_sfreq = data.get("orig_sfreq", 250)
#     target_sfreq = 200

#     # Parse S3 URI
#     assert s3_uri.startswith("s3://")
#     _, _, bucket, *key_parts = s3_uri.split("/")
#     key = "/".join(key_parts)

#     # Download EEG file from S3
#     s3 = boto3.client("s3")
#     response = s3.get_object(Bucket=bucket, Key=key)
#     eeg_bytes = response["Body"].read()

#     eeg_np = np.loadtxt(io.BytesIO(eeg_bytes), delimiter=",", skiprows=1)

#     # Extract EEG from second column if 2D
#     if eeg_np.ndim > 1:
#         raw_signal = eeg_np[:, 1]
#     else:
#         raw_signal = eeg_np

#     # Filter and preprocess
#     raw_signal = apply_notch(raw_signal, sfreq=orig_sfreq, freq=50)
#     raw_signal = apply_bandpass(raw_signal, sfreq=orig_sfreq, l_freq=0.3, h_freq=35)
#     raw_signal = apply_resample(raw_signal, orig_sfreq, target_sfreq)

#     eeg = preprocess_raw_1d_eeg(raw_signal)
#     eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

#     return eeg_tensor

# def predict_fn(input_data, model):
#     with torch.no_grad():
#         logits = model(input_data)
#         probs = F.softmax(logits, dim=1)
#         predictions = torch.argmax(probs, dim=1)
#     return predictions.cpu().numpy()

# def output_fn(prediction, accept='application/json'):
#     return json.dumps({"prediction": prediction.tolist()})