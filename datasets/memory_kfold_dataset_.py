import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from collections import defaultdict
from utils.util import to_tensor


class MemoryEfficientKFoldDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        self.samples = []
        self.metadata = []

        for seq_path, label_path in seqs_labels_path_pair:
            base = os.path.basename(seq_path)
            sid = base.split("_")[0]  # works for S001_night1 or sub-001_ses-001

            try:
                seq = np.load(seq_path)       # shape: [N, 1, 30, 200]
                label = np.load(label_path)   # shape: [N]
            except Exception as e:
                print(f"[❌] Failed loading {seq_path}: {e}")
                continue

            if seq.shape[0] != label.shape[0]:
                print(f"[ERROR] Mismatch: {seq.shape[0]} vs {label.shape[0]} in {seq_path}")
                continue

            if not self.check_integrity(seq, seq_path):
                print(f"[❌] Integrity check failed: {seq_path}")
                continue

            for i in range(len(label)):
                if label[i] is None or label[i] < 0:
                    continue
                self.samples.append((seq[i], int(label[i])))
                self.metadata.append({
                    'subject': sid,
                    'file': base,
                    'index': i,
                    'path': seq_path
                })

        if len(self.samples) == 0:
            raise ValueError("All samples filtered out. Empty dataset.")

        print(f"[✅] Loaded total {len(self.samples)} valid samples from {len(seqs_labels_path_pair)} files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch, label = self.samples[idx]
        return to_tensor(epoch), torch.tensor(label, dtype=torch.long)

    def get_metadata(self):
        return self.metadata

    def check_integrity(self, data, filename):
        if np.isnan(data).any() or np.isinf(data).any():
            return False
        if np.abs(data).max() > 1e6:
            return False
        variances = np.var(data.reshape(data.shape[0], -1), axis=1)
        if np.sum(variances == 0) > 0:
            return False
        if np.any(np.abs(data) >= 32768):
            return False
        return True
    
    
def get_custom_split(dataset, n_splits=3, seed=42):
    metadata = dataset.get_metadata()
    openneuro_indices = []
    orp_subject_to_indices = defaultdict(list)

    for idx, meta in enumerate(metadata):
        sid = meta['subject']
        if sid.startswith('S'):
            orp_subject_to_indices[sid].append(idx)
        else:
            openneuro_indices.append(idx)

    orp_subjects = sorted(orp_subject_to_indices.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(orp_subjects)

    # Split ORP subjects: 40% train, 30% val candidates, 30% test candidates
    n = len(orp_subjects)
    n_train = int(0.4 * n)
    n_val = int(0.3 * n)
    n_test = n - n_train - n_val

    orp_train_subjects = orp_subjects[:n_train]
    orp_val_subjects = orp_subjects[n_train:n_train + n_val]
    orp_test_subjects = orp_subjects[n_train + n_val:]

    # Indices
    train_indices = openneuro_indices + [i for sid in orp_train_subjects for i in orp_subject_to_indices[sid]]
    val_indices = [i for sid in orp_val_subjects for i in orp_subject_to_indices[sid]]
    test_indices = [i for sid in orp_test_subjects for i in orp_subject_to_indices[sid]]

    print(f"\n✅ Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    yield 0, train_indices, val_indices, test_indices
