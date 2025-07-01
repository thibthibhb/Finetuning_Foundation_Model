# merged_idun_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from utils.util import to_tensor

            

    
class MemoryEfficientKFoldDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        self.samples = []
        self.metadata = []

        for seq_path, label_path in seqs_labels_path_pair:
            base = os.path.basename(seq_path)
            sid = base.split("_")[0]

            try:
                seq = np.load(seq_path)
                label = np.load(label_path)
            except Exception as e:
                print(f"[❌] Failed loading {seq_path}: {e}")
                continue

            if seq.shape[0] != label.shape[0]:
                print(f"[ERROR] Mismatch: {seq.shape[0]} vs {label.shape[0]} in {seq_path}")
                continue

            if not self.check_integrity(seq, seq_path):
                print(f"[❌] Integrity check failed: {seq_path}")
                continue

            label = label.astype(int)

            # 🔁 Remap labels to 4 classes
            #label = np.array([self.remap_label(l) for l in label], dtype=int)

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

    # def remap_label(self, l):
    #     if l == 0:
    #         return 0  # Wake
    #     elif l in [1, 2]:
    #         return 1  # Light
    #     elif l == 3:
    #         return 2  # Deep
    #     elif l == 4:
    #         return 3  # REM
    #     else:
    #         raise ValueError(f"Unknown label value: {l}")

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

def get_custom_split(dataset, n_splits=3, seed=42, orp_train_frac=0.1):
    from collections import defaultdict
    import numpy as np

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

    n = len(orp_subjects)
    n_train = int(orp_train_frac * n)
    n_val = int(0.20 * n)

    orp_train_subjects = orp_subjects[:n_train]
    orp_val_subjects = orp_subjects[n_train:n_train + n_val]
    orp_test_subjects = orp_subjects[n_train + n_val:]

    train_indices = openneuro_indices + [i for sid in orp_train_subjects for i in orp_subject_to_indices[sid]]
    val_indices = [i for sid in orp_val_subjects for i in orp_subject_to_indices[sid]]
    test_indices = [i for sid in orp_test_subjects for i in orp_subject_to_indices[sid]]

    train_subjects = {metadata[i]["subject"] for i in train_indices}
    num_subjects_train = len(train_subjects)
    dataset.num_subjects_train = num_subjects_train

    print(
        f"\n✅ Dataset split:"
        f" {len(train_indices)} train"
        f" ({num_subjects_train} subjects),"
        f" {len(val_indices)} val,"
        f" {len(test_indices)} test"
    )
    yield 0, train_indices, val_indices, test_indices


# def get_custom_split(dataset, seed=42, train_frac=0.7, val_frac=0.15, orp_train_frac=0.6):
#     import numpy as np

#     metadata = dataset.get_metadata()
#     all_indices = list(range(len(metadata)))

#     rng = np.random.default_rng(seed)
#     rng.shuffle(all_indices)

#     n_total = len(all_indices)
#     n_train = int(train_frac * n_total)
#     n_val = int(val_frac * n_total)
#     n_test = n_total - n_train - n_val

#     train_indices = all_indices[:n_train]
#     val_indices = all_indices[n_train:n_train + n_val]
#     test_indices = all_indices[n_train + n_val:]

#     train_subjects = {metadata[i]["subject"] for i in train_indices}
#     num_subjects_train = len(train_subjects)
#     dataset.num_subjects_train = num_subjects_train

#     print(
#         f"\n✅ Unified dataset split:"
#         f" {len(train_indices)} train"
#         f" ({num_subjects_train} subjects),"
#         f" {len(val_indices)} val,"
#         f" {len(test_indices)} test"
#     )

#     yield 0, train_indices, val_indices, test_indices


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.dataset_names = self.params.dataset_names
        print(f"[INFO] Using {self.params.num_datasets} datasets: {self.dataset_names}")
        self.seqs_labels_path_pair = self.load_paths()

    def get_all_pairs(self):
        return self.seqs_labels_path_pair

    def load_paths(self):
        seqs_labels_path_pair = []

        for dataset_name in self.dataset_names:
            seqs_dir = os.path.join(self.params.datasets_dir, dataset_name, 'eeg_data_npy')
            labels_dir = os.path.join(self.params.datasets_dir, dataset_name, 'label_npy')

            all_files = sorted(f for f in os.listdir(seqs_dir) if f.endswith('.npy'))

            for seq_fname in all_files:
                subj_id = seq_fname.replace('.npy', '')
                label_fname = f"{subj_id}.npy"
                seq_path = os.path.join(seqs_dir, seq_fname)
                label_path = os.path.join(labels_dir, label_fname)

                if os.path.exists(label_path):
                    seqs_labels_path_pair.append((seq_path, label_path))
                else:
                    print(f"[WARN] Label not found for {subj_id} in {dataset_name}")

        return seqs_labels_path_pair
