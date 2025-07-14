import scipy.io as sio
import numpy as np
import os
from scipy.signal import butter, filtfilt, iirnotch, resample
from glob import glob
import torch

# Plot channel 1 - channel 2 for the first file
import matplotlib.pyplot as plt

# Define preprocessing functions
def bandpass_filter(data, lowcut=0.3, highcut=75.0, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, notch_freq=60.0, fs=250, Q=30):
    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, data)

def resample_data(data, original_fs, target_fs=200):
    num_samples = int(len(data) * target_fs / original_fs)
    return resample(data, num_samples)

def preprocess_subject(seq_mat_path, label_mat_path, output_seq_dir, output_label_dir):
    mat_seq = sio.loadmat(seq_mat_path)
    mat_label = sio.loadmat(label_mat_path)

    eeg_data = mat_seq['EEG']
    if eeg_data.shape[0] != 2:
        print(f"⛔ Unexpected EEG shape in {seq_mat_path}: {eeg_data.shape}")
        return

    raw_eeg = eeg_data[0] - eeg_data[1]


    stage_data = mat_label['stageData']
    stages = stage_data['stages'][0][0].squeeze()
    orig_fs = int(stage_data['srate'][0][0][0][0])
    epoch_len = 30 * orig_fs
    total_samples = raw_eeg.shape[0]

    eeg_filtered = bandpass_filter(raw_eeg, fs=orig_fs)
    eeg_filtered = notch_filter(eeg_filtered, fs=orig_fs)

    eeg_resampled = resample_data(eeg_filtered, original_fs=orig_fs, target_fs=200)

    n_epochs = min(len(eeg_resampled) // (30 * 200), len(stages))
    eeg_epochs = eeg_resampled[:n_epochs * 30 * 200].reshape(n_epochs, 30, 200)
    stages = stages[:n_epochs]
    print(f"Processing {seq_mat_path}: {n_epochs*30*200} epochs, {total_samples} samples")
    
    label_map = {
        0: 0,  # Wake
        1: 1,  # N1
        2: 2,  # N2
        3: 3,  # N3
        5: 4,  # REM
        4: None, 6: None, 7: None  # Drop or ignore
    }

    clean_epochs = []
    clean_labels = []

    for i in range(n_epochs):
        raw_label = int(stages[i])
        mapped_label = label_map.get(raw_label, None)
        if mapped_label is None:
            continue

        epoch = eeg_epochs[i]
        if np.any(np.abs(epoch) > 100):
            print("warning: extreme values in epoch")
            continue
        # each epoch independently
        epoch = (epoch - epoch.mean()) / epoch.std()
        # or global standardization per subject:
        # global_mean = eeg_resampled.mean()
        # global_std = eeg_resampled.std()
        # epoch = (epoch - global_mean) / global_std
        clean_epochs.append(epoch)
        clean_labels.append(mapped_label)

    if not clean_epochs:
        print(f"⛔ No clean epochs in {seq_mat_path}")
        return

    clean_epochs = np.array(clean_epochs)
    clean_epochs = clean_epochs[:, None, :]

    os.makedirs(output_seq_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    subject_id = os.path.basename(seq_mat_path).split('_')[0]
    np.save(os.path.join(output_seq_dir, f"{subject_id}_takeda.npy"), clean_epochs)
    np.save(os.path.join(output_label_dir, f"{subject_id}_takeda.npy"), np.array(clean_labels))
    unique, counts = np.unique(clean_labels, return_counts=True)
    print(dict(zip(unique, counts)))


    # Assuming clean_labels is a 1D numpy array of class labels
    unique, counts = np.unique(clean_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class counts:", class_counts)

    # Inverse frequency weighting
    total = sum(counts)
    class_weights = [c / total for c in counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Save weights
    output_path = os.path.join(output_label_dir, f"{subject_id}_takeda_tensor.npy")
    np.save(output_path, class_weights.numpy())
    print(f"Saved class weights to {output_path}")
    clean_labels = torch.tensor(clean_labels)
    print("Label shape:", clean_labels.shape)
    print("Unique labels:", torch.unique(clean_labels, return_counts=True))

    print("[DEBUG] Raw sample stats:", clean_epochs[0, 0].mean().item(), clean_epochs[0, 0].std().item())
    print("[DEBUG] Raw sample values:", clean_epochs[0, 0, :2, :2])  # show a small patch of the signal
    print(f"✅ Processed {subject_id}_takeda: {len(clean_labels)} clean epochs.")

# Paths
input_seq_dir = './Takeda_fine_tuning/seq'
input_label_dir = './Takeda_fine_tuning/labels'
output_seq_dir = './Final_dataset/seq_npy'
output_label_dir = './Final_dataset/label_npy'

# Match files by subject ID
seq_files = sorted(glob(os.path.join(input_seq_dir, '*.mat')))
label_files = sorted(glob(os.path.join(input_label_dir, '*.mat')))
label_map = {os.path.basename(f).split('_')[0]: f for f in label_files}

# Run preprocessing
for seq_path in seq_files:
    subject_id = os.path.basename(seq_path).split('_')[0]
    label_path = label_map.get(subject_id)
    if label_path:
        preprocess_subject(seq_path, label_path, output_seq_dir, output_label_dir)
    else:
        print(f"⚠️ Label file not found for {subject_id}")
