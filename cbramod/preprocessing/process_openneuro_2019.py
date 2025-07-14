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
def extract_stage_labels(tsv_path, total_epochs):
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"[DEBUG] Columns in {tsv_path}: {df.columns.tolist()}")
    print(f"[DEBUG] Unique stage codes in 'Scoring1': {df['Scoring1'].unique()}")

    # Integer-based mapping
    stage_code_map = {
        1: 0,  # Wake
        2: 4,  # REM
        3: 1,  # N1
        4: 2,  # N2
        5: 3,  # N3
        6: -1, # Arousal
        7: -1, # Artefact
        8: -1  # Unscored
    }

    stages = np.full(total_epochs, -1)
    for _, row in df.iterrows():
        onset = int(float(row['onset']))
        duration = int(float(row['duration']))
        stage_code = int(row['Scoring1'])
        mapped_label = stage_code_map.get(stage_code, -1)
        epoch_start = onset // 30
        epoch_end = (onset + duration) // 30
        stages[epoch_start:epoch_end] = mapped_label

    return stages


def create_bipolar(data, ch_names, ch1, ch2):
    try:
        return data[ch_names.index(ch1)] - data[ch_names.index(ch2)]
    except ValueError:
        return None

def preprocess_openneuro_session(subject_id, session_id, eeg_file, scoring_file, output_seq_dir, output_label_dir):
    try:
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        print(f"[INFO] Loaded {eeg_file} with channels: {raw.ch_names}")
        data = raw.get_data()
        fs = int(raw.info['sfreq'])
    except Exception as e:
        print(f"⚠️ Error loading {eeg_file}: {e}")
        return

    total_epochs = len(data[0]) // (30 * fs)
    labels = extract_stage_labels(scoring_file, total_epochs=total_epochs)
    print(f"[DEBUG] Label summary for {subject_id} {session_id}: {dict(pd.Series(labels).value_counts())}")

    preferred_pairs = [
        ('ELE', 'ERE'),
        ('ELI', 'ERI'),
    ]

    bipolar_signals = {}
    for idx, (ch1, ch2) in enumerate(preferred_pairs, start=1):
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            signal = create_bipolar(data, raw.ch_names, ch1, ch2)
            if signal is not None:
                bipolar_signals[f"p{idx}"] = signal
                print(f"[INFO] Using pair ({ch1}, {ch2}) as bipolar p{idx}")
        else:
            print(f"[WARN] Missing channels ({ch1}, {ch2}) in {subject_id} {session_id}")

    if not bipolar_signals:
        print(f"[WARN] No valid bipolar pair found in {subject_id} {session_id}")
        return

    os.makedirs(output_seq_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for suffix, signal in bipolar_signals.items():
        print(f"[INFO] Processing Bipolar {suffix} for {subject_id} {session_id}")

        signal = np.nan_to_num(signal, nan=np.nanmean(signal))
        filtered = bandpass_filter(signal, fs=fs)
        filtered = notch_filter(filtered, fs=fs)
        resampled = resample_data(filtered, original_fs=fs, target_fs=200)

        n_epochs = len(resampled) // (30 * 200)
        epochs = resampled[:n_epochs * 30 * 200].reshape(n_epochs, 30, 200)

        if len(labels) != n_epochs:
            min_len = min(len(labels), n_epochs)
            labels = labels[:min_len]
            epochs = epochs[:min_len]

        clean_epochs = []
        clean_labels = []
        for epoch, label in zip(epochs, labels):
            if label == -1:
                continue
            norm_epoch = (epoch - epoch.mean()) / epoch.std()
            if np.any(np.abs(norm_epoch) > 100):
                print(f"[DEBUG] Epoch {i} rejected (label={label}) with max abs z-score = {max_abs:.2f}")
                continue
            clean_epochs.append(norm_epoch)
            clean_labels.append(label)

        if not clean_epochs:
            print(f"[WARN] No clean data for {subject_id} {session_id}, bipolar {suffix}")
            continue

        clean_epochs = np.array(clean_epochs)[:, None, :, :]
        clean_labels = np.array(clean_labels)

        base_filename = f"{subject_id}_{session_id}_{suffix}"
        np.save(os.path.join(output_seq_dir, f"2019-{base_filename}.npy"), clean_epochs)
        np.save(os.path.join(output_label_dir, f"2019-{base_filename}.npy"), clean_labels)

        print(f"✅ Saved {base_filename} with {len(clean_labels)} epochs")
        print("Class counts:", dict(zip(*np.unique(clean_labels, return_counts=True))))


# === Batch Process all subjects and sessions ===
root_dir = '/root/cbramod/CBraMod/OpenNeuro_2019'
output_seq_dir = './Final_dataset/2019_Open_N/eeg_data_npy'
output_label_dir = './Final_dataset/2019_Open_N/label_npy'

subjects = sorted(glob(os.path.join(root_dir, 'sub-*')))
sessions = [f"ses-00{i}" for i in range(1, 5)]

for subject_path in subjects:
    subject_id = os.path.basename(subject_path)
    for session_id in sessions:
        eeg_files = glob(os.path.join(subject_path, session_id, 'eeg', '*sleep_acq-PSG_eeg.set'))
        scoring_files = glob(os.path.join(subject_path, session_id, 'eeg', '*scoring1_events.tsv'))

        if eeg_files and scoring_files:
            preprocess_openneuro_session(
                subject_id=subject_id,
                session_id=session_id,
                eeg_file=eeg_files[0],
                scoring_file=scoring_files[0],
                output_seq_dir=output_seq_dir,
                output_label_dir=output_label_dir
            )
        else:
            print(f"[SKIP] Missing files for {subject_id} {session_id}")
