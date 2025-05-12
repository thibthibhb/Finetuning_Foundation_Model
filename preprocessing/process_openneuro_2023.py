import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, resample
from glob import glob
import pandas as pd
import torch

# === Filtering functions ===
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

# === Label extraction with artifact handling ===
def extract_stage_labels(tsv_path, total_epochs):
    df = pd.read_csv(tsv_path, sep='\t')
    stages = np.full(total_epochs, -1)
    stage_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'Artefact': -1}

    for _, row in df.iterrows():
        onset = int(float(row['onset']))
        duration = int(float(row['duration']))
        label = row['scoring']
        epoch_start = onset // 30
        epoch_end = (onset + duration) // 30
        mapped_label = stage_map.get(label, -1)
        stages[epoch_start:epoch_end] = mapped_label

    return stages

def create_bipolar_channel(data, ch_names, ch1='LT', ch2='RT'):  # LB - RB
    try:
        idx1 = ch_names.index(ch1)
        idx2 = ch_names.index(ch2)
        return data[idx1] - data[idx2]
    except ValueError:
        raise ValueError(f"One of the channels {ch1} or {ch2} not found in {ch_names}")

import gzip

def parse_data_quality(dataqual_path):
    with gzip.open(dataqual_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t')

    print(f"[DEBUG] Columns in {dataqual_path}:", df.columns.tolist())

    if 'channel' not in df.columns:
        raise KeyError("Missing 'channel' column in dataQual file")

    df.columns = df.columns.str.strip()
    df['channel'] = df['channel'].str.strip()

    # You could use any metric: here we use 'peakToPeak'
    sorted_df = df.sort_values(by='peakToPeak', ascending=False)

    # Try pairing best left-ear and right-ear electrodes
    left_candidates = ['LT', 'LB']
    right_candidates = ['RT', 'RB']

    best_left = next((row['channel'] for _, row in sorted_df.iterrows() if row['channel'] in left_candidates), None)
    best_right = next((row['channel'] for _, row in sorted_df.iterrows() if row['channel'] in right_candidates), None)

    return best_left, best_right

import matplotlib.pyplot as plt

def plot_raw_vs_filtered(raw_data, filtered_data, sfreq, title='Signal'):
    t = np.arange(len(raw_data)) / sfreq
    plt.figure(figsize=(15, 4))
    plt.plot(t, raw_data, label='Raw', alpha=0.5)
    plt.plot(t, filtered_data, label='Filtered', alpha=0.8)
    plt.legend()
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True)
    plt.show()

def preprocess_openneuro_session(subject_id, session_id, eeg_file, scoring_file, output_seq_dir, output_label_dir):
    import os
    import numpy as np
    import mne
    import torch

    def create_bipolar(data, ch_names, ch1, ch2):
        idx1 = ch_names.index(ch1)
        idx2 = ch_names.index(ch2)
        return data[idx1] - data[idx2]

    try:
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        print(f"[INFO] Loaded channels: {raw.ch_names}")
        data = raw.get_data()

        # Create both bipolar signals
        bipolar_signals = {}
        if 'LB' in raw.ch_names and 'RB' in raw.ch_names:
            bipolar_signals['1'] = create_bipolar(data, raw.ch_names, 'LB', 'RB')
        else:
            print(f"[WARN] Missing LB or RB for {subject_id} {session_id}")
        
        if 'LT' in raw.ch_names and 'RT' in raw.ch_names:
            bipolar_signals['2'] = create_bipolar(data, raw.ch_names, 'LT', 'RT')
        else:
            print(f"[WARN] Missing LT or RT for {subject_id} {session_id}")

        if not bipolar_signals:
            print(f"[ERROR] No bipolar signals available for {subject_id} {session_id}")
            return

    except Exception as e:
        print(f"⚠️ Error loading {eeg_file}: {e}")
        return

    # === Filtering, resampling, epoching ===
    fs = int(raw.info['sfreq'])
    labels = extract_stage_labels(scoring_file, total_epochs=len(data[0]) // (30 * fs))

    os.makedirs(output_seq_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for suffix, bipolar_signal in bipolar_signals.items():
        print(f"[INFO] Processing Bipolar {suffix} for {subject_id} {session_id}")

        # Preprocess
        signal = np.nan_to_num(bipolar_signal, nan=np.nanmean(bipolar_signal))
        raw_filtered = bandpass_filter(signal, fs=fs)
        raw_filtered = notch_filter(raw_filtered, fs=fs)
        raw_resampled = resample_data(raw_filtered, original_fs=fs, target_fs=200)

        n_epochs = len(raw_resampled) // (30 * 200)
        epochs = raw_resampled[:n_epochs * 30 * 200].reshape(n_epochs, 30, 200)
        if len(labels) != epochs.shape[0]:
            min_len = min(len(labels), epochs.shape[0])
            epochs = epochs[:min_len]
            labels_used = labels[:min_len]
        else:
            labels_used = labels

        clean_epochs = []
        clean_labels = []
        for epoch, label in zip(epochs, labels_used):
            if label == -1:
                continue
            epoch = (epoch - epoch.mean()) / epoch.std()
            if np.any(np.abs(epoch) > 100):
                continue
            clean_epochs.append(epoch)
            clean_labels.append(label)
        
        if not clean_epochs:
            print(f"[WARN] No clean data for bipolar {suffix} in {subject_id} {session_id}")
            continue

        clean_epochs = np.array(clean_epochs)[:, None, :, :]
        clean_labels = np.array(clean_labels)

        base_filename = f"{subject_id}_{session_id}_{suffix}"
        np.save(os.path.join(output_seq_dir, f"2023-{base_filename}.npy"), clean_epochs)
        np.save(os.path.join(output_label_dir, f"2023-{base_filename}.npy"), clean_labels)

        print(f"✅ Saved {base_filename} with {len(clean_labels)} epochs")
        unique, counts = np.unique(clean_labels, return_counts=True)
        print("Class counts:", dict(zip(unique, counts)))

        class_weights = [c / sum(counts) for c in counts]
        torch_weights = torch.tensor(class_weights, dtype=torch.float32)
        #np.save(os.path.join(output_label_dir, f"{base_filename}_tensor.npy"), torch_weights.numpy())


# === Run preprocessing for all subjects & sessions ===
root_dir = './OpenNeuro/OpenNeuro2023'
output_seq_dir = './Final_dataset/seq_npy'
output_label_dir = './Final_dataset/label_npy'

subjects = sorted(glob(os.path.join(root_dir, 'sub-*')))    

for subject_path in subjects:   
    subject_id = os.path.basename(subject_path)
    for session_id in ['ses-001', 'ses-002']:
        eeg_file = glob(os.path.join(subject_path, session_id, 'eeg', '*earEEG_eeg.set'))
        scoring_file = glob(os.path.join(subject_path, session_id, 'eeg', '*scoring_events.tsv'))

        if eeg_file and scoring_file:
            preprocess_openneuro_session(
                subject_id, session_id,
                eeg_file[0], scoring_file[0],
                output_seq_dir, output_label_dir,
            )
