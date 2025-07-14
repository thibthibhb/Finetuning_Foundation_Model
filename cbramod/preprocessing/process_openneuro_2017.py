import os
import numpy as np
import mne
import torch
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd
from glob import glob

def bandpass_filter(data, lowcut=0.3, highcut=75.0, fs=200, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, notch_freq=50.0, fs=200, Q=30):
    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, data)

def extract_stage_labels(tsv_path, total_epochs):
    label_map = {
        1: 0,  # Wake
        2: 4,  # REM
        3: 1,  # N1
        4: 2,  # N2
        5: 3,  # N3
        6: -1, # A
        7: -1, # Artefact
        8: -1  # Unscored
    }

    df = pd.read_csv(tsv_path, sep='\t')
    stages = np.full(total_epochs, -1)

    for _, row in df.iterrows():
        onset = int(float(row['onset']))
        duration = 30
        try:
            raw_label = int(float(row['staging']))
        except ValueError:
            print(f"[WARN] Non-integer label found: {row['staging']}, defaulting to -1")
            raw_label = -1
        label = label_map.get(raw_label, -1)
        epoch_start = onset // 30
        epoch_end = (onset + duration) // 30
        stages[epoch_start:epoch_end] = label

    print("First 10 labels:", stages[:10])
    return stages

def create_bipolar_channel(data, ch_names, ch1, ch2):
    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)
    return data[idx1] - data[idx2]

def preprocess_openneuro_session(subject_id, session_id, eeg_file, scoring_file, output_seq_dir, output_label_dir):
    try:
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        print(f"[INFO] Loaded channels: {raw.ch_names}")
        data = raw.get_data()
    except Exception as e:
        print(f"⚠️ Error loading {eeg_file}: {e}")
        return

    preferred_pairs = [
        ('ELG', 'ERG'),
        ('ELE', 'ERE'),
        ('ELI', 'ERI'),
        ('ELK', 'ERK'),
    ]

    pair_index = 1
    for ch1, ch2 in preferred_pairs:
        if ch1 not in raw.ch_names or ch2 not in raw.ch_names:
            continue

        print(f"[INFO] Using bipolar pair: {ch1} - {ch2}")
        bipolar = create_bipolar_channel(data, raw.ch_names, ch1, ch2)
        bipolar = np.expand_dims(np.nan_to_num(bipolar, nan=np.nanmean(bipolar)), axis=0)

        info = mne.create_info(['Bipolar'], 200, ch_types='eeg')
        raw_bipolar = mne.io.RawArray(bipolar, info)

        fs = 200
        data_bipolar = raw_bipolar.get_data()
        labels = extract_stage_labels(scoring_file, total_epochs=len(data_bipolar[0]) // (30 * fs))
        print("Raw label distribution:", dict(zip(*np.unique(labels, return_counts=True))))

        raw_data = data_bipolar[0]
        raw_filtered = bandpass_filter(raw_data, fs=fs)
        raw_filtered = notch_filter(raw_filtered, fs=fs)
        raw_resampled = raw_filtered

        n_epochs = len(raw_resampled) // (30 * fs)
        epochs = raw_resampled[:n_epochs * 30 * fs].reshape(n_epochs, 30, fs)

        if len(labels) != epochs.shape[0]:
            min_len = min(len(labels), epochs.shape[0])
            epochs = epochs[:min_len]
            labels = labels[:min_len]

        clean_epochs = []
        clean_labels = []
        for epoch, label in zip(epochs, labels):
            if label == -1:
                continue
            epoch = (epoch - epoch.mean()) / epoch.std()
            if np.any(np.abs(epoch) > 500):
                continue
            clean_epochs.append(epoch)
            clean_labels.append(label)

        if not clean_epochs:
            print(f"[WARN] No clean data for pair {ch1}-{ch2} in {subject_id} {session_id}")
            continue

        clean_epochs = np.array(clean_epochs)[:, None, :, :]
        clean_labels = np.array(clean_labels)

        base_filename = f"{subject_id}_{session_id}_{pair_index}"
        np.save(os.path.join(output_seq_dir, f"2017-{base_filename}.npy"), clean_epochs)
        np.save(os.path.join(output_label_dir, f"2017-{base_filename}.npy"), clean_labels)

        unique, counts = np.unique(clean_labels, return_counts=True)
        print(f"[INFO] Saved {base_filename} with class counts: {dict(zip(unique, counts))}")

        class_weights = [c / sum(counts) for c in counts]
        torch_weights = torch.tensor(class_weights, dtype=torch.float32)
        #np.save(os.path.join(output_label_dir, f"{base_filename}_tensor.npy"), torch_weights.numpy())

        pair_index += 1

# === Run preprocessing ===
root_dir = './OpenNeuro2017'
output_seq_dir = './Final_dataset/2017_Open_N/eeg_data_npy'
output_label_dir = './Final_dataset/2017_Open_N/label_npy'

os.makedirs(output_seq_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

subjects = sorted(glob(os.path.join(root_dir, 'sub-*')))

for subject_path in subjects:
    subject_id = os.path.basename(subject_path)
    for session_id in ['ses-001']:
        eeg_file = glob(os.path.join(subject_path, session_id, 'eeg', '*task-sleep_eeg.set'))
        scoring_file = glob(os.path.join(subject_path, session_id, 'eeg', '*scoring_events.tsv'))

        if eeg_file and scoring_file:
            preprocess_openneuro_session(
                subject_id, session_id,
                eeg_file[0], scoring_file[0],
                output_seq_dir, output_label_dir
            )
