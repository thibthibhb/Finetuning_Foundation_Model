import os
import numpy as np
import pandas as pd
import boto3
from scipy.signal import butter, filtfilt, iirnotch, resample

#### I am around here: 5363b8a5-9df4-4e47-b77e-9a63ae566956/     ### of bucket run through 
# ==== Configuration ====
bucket_name = 'idn-dev-raw-recordings-bucket'
sleep_root = '/root/cbramod/CBraMod/Unlabelled/Test'
max_files = 25
original_fs = 250
target_fs = 200
min_duration_hours = 7
min_duration_sec = min_duration_hours * 3600

os.makedirs(sleep_root, exist_ok=True)

# ==== Preprocessing Functions ====
def bandpass_filter(data, lowcut=0.3, highcut=75.0, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, notch_freq=60.0, fs=250, Q=30):
    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, data, axis=0)

def resample_data(data, original_fs, target_fs=200):
    num_samples = int(len(data) * target_fs / original_fs)
    return resample(data, num_samples, axis=0)

# ==== S3 Initialization ====
s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

seen_sleep = set(os.listdir(sleep_root))
print(f"Found {len(seen_sleep)} existing files in Sleep. Skipping those...")

downloaded = 0
sleep_copied = 0

for page in paginator.paginate(Bucket=bucket_name):
    if 'Contents' not in page:
        continue

    for obj in page['Contents']:
        key = obj['Key']
  

        fname = os.path.basename(key)

        if not fname.startswith("eeg_") or not fname.endswith(".csv"):
            continue

        npy_fname = fname.replace(".csv", ".npy")
        sleep_path = os.path.join(sleep_root, npy_fname)

        if os.path.exists(sleep_path):
            continue

        try:
            print(f"⬇️ Downloading {key}")
            tmp_csv = '/tmp/tmp_eeg.csv'
            s3.download_file(bucket_name, key, tmp_csv)

            df = pd.read_csv(tmp_csv)

            if df.shape[0] < 10 or 'timestamp' not in df.columns or 'ch1' not in df.columns:
                print(f"⏭️ Skipping {key}: too few samples or missing columns")
                continue

            # Duration check
            start_time = df['timestamp'].iloc[0]
            end_time = df['timestamp'].iloc[-1]
            duration_sec = end_time - start_time
            duration_hr = duration_sec / 3600
            print(f"  🕒 Duration: {duration_hr:.2f} h, samples: {df.shape[0]}")

            if duration_sec < min_duration_sec:
                print(f"⏩ Skipped {npy_fname}: only {duration_hr:.2f} hours")
                continue

            eeg = df['ch1'].values.astype(np.float32)
            eeg = notch_filter(eeg, fs=original_fs)
            eeg = bandpass_filter(eeg, fs=original_fs)
            eeg = resample_data(eeg, original_fs, target_fs)

            np.save(sleep_path, eeg)
            print(f"🌙 Saved to Sleep: {npy_fname} ({len(eeg)/(target_fs*3600):.2f} hours)")
            sleep_copied += 1
            downloaded += 1

            if downloaded >= max_files:
                print("🚫 Reached download limit.")
                break

        except Exception as e:
            print(f"⚠️ Error processing {key}: {e}")

    if downloaded >= max_files:
        break

print(f"\n✅ Done. {downloaded} valid files saved to Sleep.")
print(f"🌙 Total files copied to Sleep: {sleep_copied}")
