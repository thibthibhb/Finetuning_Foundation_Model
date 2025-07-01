import os
import numpy as np
from collections import Counter

# === Parameters
input_labels_dir = "/root/cbramod/CBraMod/ORP_output_npy/labels_npy"
input_seq_dir = "/root/cbramod/CBraMod/ORP_output_npy/eeg_data_npy"
output_dir = "/root/cbramod/CBraMod/Final_dataset/ORP_5_classes"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'eeg_data'), exist_ok=True)
epoch_len = 30
start_epoch = 180
lookahead = 50
first_cycle_max = 30
rem_cooldown = 120

# === Patient and night loop
for subj in range(1, 17):
    for night in range(1, 6):
        tag = f"S{subj:03d}_night{night}"
        label_path = os.path.join(input_labels_dir, f"{tag}_labels.npy")
        seq_path = os.path.join(input_seq_dir, f"{tag}_eeg_chan_idun.npy")

        if not os.path.exists(label_path) or not os.path.exists(seq_path):
            print(f"[SKIP] Missing files for {tag}")
            continue

        try:
            guardian_labels = np.load(label_path)
            guardian_5class = guardian_labels.copy()
            min_len = len(guardian_labels)

            first_rem_done = False
            cooldown_end = -1

            for i in range(start_epoch, min_len - lookahead):
                current = guardian_5class[i]
                future = guardian_5class[i + 1:i + 1 + lookahead]

                if not first_rem_done:
                    if current in [0, 1]:
                        deep_count = sum(f in [2, 3] for f in future)
                        if deep_count >= 30 and guardian_5class[i] != 4:
                            rem_block_end = i + first_cycle_max
                            for j in range(i, min(rem_block_end, len(guardian_5class))):
                                if guardian_5class[j] in [0, 1]:
                                    guardian_5class[j] = 4
                            cooldown_end = rem_block_end + rem_cooldown
                            first_rem_done = True

                elif i > cooldown_end:
                    if current in [0, 1]:
                        past = guardian_5class[i - 20:i]
                        future = guardian_5class[i + 1:i + 1 + lookahead]
                        n2_past = sum(p == 2 for p in past)
                        n3_past = sum(p == 3 for p in past)
                        light_future = sum(f in [0, 1, 2] for f in future)

                        if n2_past >= 15 and n3_past <= 3 and light_future >= 30:
                            guardian_5class[i] = 4

            label_counts = Counter(guardian_5class)
            print(f"[DISTRIBUTION] {tag}: {label_counts}")

            # === Save updated labels
            out_labels_path = os.path.join(output_dir, 'label_npy', f"{tag}.npy")
            out_seq_path = os.path.join(output_dir, 'eeg_data_npy', f"{tag}.npy")

            np.save(out_labels_path, guardian_5class)

            # === Copy EEG data unchanged
            eeg_seq = np.load(seq_path)
            np.save(out_seq_path, eeg_seq)

            print(f"[OK] Saved 5-class labels for {tag}")

        except Exception as e:
            print(f"[ERROR] {tag}: {e}")
