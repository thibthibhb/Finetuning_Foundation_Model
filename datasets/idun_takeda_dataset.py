import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
from utils.util import to_tensor


class EpochDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        self.samples = []
        for seq_path, label_path in seqs_labels_path_pair:
            seq = np.load(seq_path)     # shape: [N, 1, 30, 200]
            label = np.load(label_path) # shape: [N]
            if seq.shape[0] != label.shape[0]:
                print(f"[ERROR] Shape mismatch in {seq_path} — seq.shape[0] = {seq.shape[0]}, label.shape[0] = {label.shape[0]}")
                continue
            if not self.check_integrity(seq, seq_path):
                print(f"[❌] Skipping corrupted file: {seq_path}")
                continue
            else:
                print(f"[✔️] Loaded {seq_path}")

            for i in range(len(label)):
                if label[i] is None or label[i] < 0:
                    continue  # skip invalid labels
                self.samples.append((seq[i], int(label[i])))
        if len(self.samples) == 0:
            print(f"[ERROR] All samples were removed due to integrity checks.")
            raise ValueError("Dataset is empty after integrity filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch, label = self.samples[idx]
        return to_tensor(epoch), torch.tensor(label, dtype=torch.long)

    def check_integrity(self, data, filename):
        if np.isnan(data).any():
            print(f"[WARN] NaNs in {filename}")
            return False
        if np.isinf(data).any():
            print(f"[WARN] Infs in {filename}")
            return False
        if np.abs(data).max() > 1e6:
            print(f"[WARN] Extreme values in {filename}: max={data.max()}, min={data.min()}")
            return False
        variances = np.var(data.reshape(data.shape[0], -1), axis=1)
        flat = np.sum(variances == 0)
        if flat > 0:
            print(f"[WARN] Flat epochs in {filename}: {flat} zero-variance epochs")
            return False
        if np.any(np.abs(data) >= 32768):
            print(f"[WARN] Clipping detected in {filename}")
            return False
        return True


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.seqs_dir = os.path.join(params.datasets_dir, 'eeg_data_npy')
        self.labels_dir = os.path.join(params.datasets_dir, 'label_npy')
        self.seqs_labels_path_pair, self.num_subjects = self.load_path()

    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair)

        train_set = EpochDataset(train_pairs)
        val_set = EpochDataset(val_pairs)
        test_set = EpochDataset(test_pairs)

        labels = [train_set[i][1].item() for i in range(len(train_set))]
        class_counts = np.bincount(labels)
        print(f"[INFO] Class counts: {class_counts}")
        class_weights = 1. / class_counts
        sample_weights = [class_weights[label] for label in labels]

        if self.params.use_weighted_sampler:
            print("[INFO] Using WeightedRandomSampler")
            # compute your sample_weights here
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = DataLoader(train_set, batch_size=self.params.batch_size, sampler=sampler)
        else:
            print("[INFO] Using standard random shuffling")
            train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True)

        print(f"[INFO] #Train: {len(train_set)} | #Val: {len(val_set)} | #Test: {len(test_set)}")

        return {
            'train': train_loader,
            'val': DataLoader(val_set, batch_size=self.params.batch_size, shuffle=False),
            'test': DataLoader(test_set, batch_size=self.params.batch_size, shuffle=False),
        }

    def load_path(self):
        seqs_labels_path_pair = []
        unique_subjects = set()

        all_files = sorted(f for f in os.listdir(self.seqs_dir) if f.endswith('.npy'))

        for seq_fname in all_files:
            subj_id = seq_fname.replace('.npy', '')
            label_fname = f"{subj_id}.npy"

            seq_path = os.path.join(self.seqs_dir, seq_fname)
            label_path = os.path.join(self.labels_dir, label_fname)

            if os.path.exists(label_path):
                seqs_labels_path_pair.append((seq_path, label_path))
                unique_subjects.add(subj_id)
            else:
                print(f"[WARN] Label not found for {subj_id}")

        return seqs_labels_path_pair, len(unique_subjects)

    def split_dataset(self, seqs_labels_path_pair):
        # Identify ORP by prefix 'S', all others are OpenNeuro
        orp_pairs = [pair for pair in seqs_labels_path_pair if os.path.basename(pair[0]).startswith("S")]
        openneuro_pairs = [pair for pair in seqs_labels_path_pair if not os.path.basename(pair[0]).startswith("S")]

        # Shuffle ORP to ensure random split
        np.random.seed(42)
        np.random.shuffle(orp_pairs)
        np.random.shuffle(openneuro_pairs)
        
        # Split ORP: 80% train, 10% val, 10% test
        total_orp = len(orp_pairs)
        train_end = int(0.7 * total_orp)
        val_end = int(0.9 * total_orp)

        orp_train = orp_pairs[:train_end]
        orp_val = orp_pairs[train_end:val_end]
        orp_test = orp_pairs[val_end:]

        # Combine ORP-train and OpenNeuro for training set
        train_pairs = orp_train + openneuro_pairs
        val_pairs = orp_val
        test_pairs = orp_test

        print(f"Total: {len(seqs_labels_path_pair)} | Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
        return train_pairs, val_pairs, test_pairs

    # def split_dataset(self, seqs_labels_path_pair):
    #     test_filename = "sub-002_ses-002_2"

    #     # Separate the designated test file from the rest
    #     test_pairs = [pair for pair in seqs_labels_path_pair if test_filename in pair[0]]
    #     other_pairs = [pair for pair in seqs_labels_path_pair if test_filename not in pair[0]]

    #     # Shuffle only the remaining (non-test) pairs
    #     np.random.seed(42)
    #     np.random.shuffle(other_pairs)

    #     # Split shuffled pairs into train and validation
    #     total = len(other_pairs)
    #     train_end = int(0.8 * total)
    #     val_end = int(0.9 * total)

    #     train_pairs = other_pairs[:train_end]
    #     val_pairs = other_pairs[train_end:val_end]

    #     print(f"Total: {len(seqs_labels_path_pair)} | Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
    #     return train_pairs, val_pairs, test_pairs
