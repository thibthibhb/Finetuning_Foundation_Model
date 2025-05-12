import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import mne  # For reading EDF files
from utils.util import to_tensor


class CustomDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

        if len(self.seqs_labels_path_pair) == 0:
            raise ValueError("❌ Error: No valid sequence-label pairs found!")

    def __len__(self):
        return len(self.seqs_labels_path_pair)

    def __getitem__(self, idx):
        seq_path, label_path = self.seqs_labels_path_pair[idx]

        # Load EEG data from EDF file
        try:
            raw = mne.io.read_raw_edf(seq_path, preload=True)
            seq = raw.get_data()  # (channels, timepoints)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to read EDF file {seq_path}: {e}")

        # Load labels from JSON file
        try:
            with open(label_path, "r") as f:
                label_data = json.load(f)
            raw_label = label_data.get("label", 0)  # Default 0 if missing
        except Exception as e:
            raise RuntimeError(f"❌ Failed to read JSON file {label_path}: {e}")

        # Ensure the label is a 1D array (even if it is a scalar)
        label = np.array([raw_label])  # wrap the raw label in a list for a 1D array

        # Check label shape and type for debugging
        print(f"🧪 Loaded label: {label} (shape: {label.shape}, type: {type(label)})")

        # Ensure 4D input for IDUN dataset: Add a new dimension for epoch_size (set to 1)
        if len(seq.shape) == 2:  # If the shape is (channels, timepoints)
            seq = seq[..., np.newaxis]  # Add a dummy dimension for epoch_size (channels, timepoints, 1)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


    def collate(self, batch):
        x_seq = torch.stack([x[0] for x in batch])
        y_label = torch.tensor([x[1] for x in batch])
        return x_seq, y_label


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.seqs_dir = os.path.join(params.datasets_dir, "seq")
        self.labels_dir = os.path.join(params.datasets_dir, "labels")
        self.seqs_labels_path_pair = self.load_path()

        if len(self.seqs_labels_path_pair) == 0:
            raise ValueError("❌ No valid EEG-Label pairs found in dataset!")

    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair)

        if len(train_pairs) == 0:
            raise ValueError("❌ Error: Training dataset is empty!")

        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs) if len(val_pairs) > 0 else None
        test_set = CustomDataset(test_pairs) if len(test_pairs) > 0 else None

        data_loader = {
            "train": DataLoader(
                train_set, batch_size=self.params.batch_size, collate_fn=train_set.collate, shuffle=True
            ),
        }

        if val_set:
            data_loader["val"] = DataLoader(val_set, batch_size=1, collate_fn=val_set.collate, shuffle=False)

        if test_set:
            data_loader["test"] = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate, shuffle=False)

        return data_loader

    def load_path(self):
        seqs_labels_path_pair = []
        
        # List all EEG and label files
        seq_files = sorted([f for f in os.listdir(self.seqs_dir) if f.endswith(".edf")])
        label_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith(".json")])

        print(f"DEBUG: Found {len(seq_files)} EEG (.edf) files:", seq_files)
        print(f"DEBUG: Found {len(label_files)} Label (.json) files:", label_files)

        if len(seq_files) == 0 or len(label_files) == 0:
            print("⚠️ Warning: No EEG or label files found! Check dataset paths.")
            return []

        # Extract subject/session IDs from filenames (last number in filename)
        def extract_id(filename):
            return filename.split("_")[-1].split("-")[-1].split(".")[0]  # Extracts '11759'

        seq_dict = {extract_id(f): f for f in seq_files}  # { "11759": "1713791311469_11759.edf" }
        label_dict = {extract_id(f): f for f in label_files}  # { "11759": "scoring_run_1736952052135-11759.json" }

        # Match EEG files to labels using the extracted subject/session ID
        for subject_id in seq_dict.keys():
            if subject_id in label_dict:
                seq_path = os.path.join(self.seqs_dir, seq_dict[subject_id])
                label_path = os.path.join(self.labels_dir, label_dict[subject_id])

                print(f"✅ Pairing EEG {seq_path} with Label {label_path}")
                seqs_labels_path_pair.append((seq_path, label_path))

        if not seqs_labels_path_pair:
            print("❌ No valid EEG-Label pairs found! Check filename formats.")

        return seqs_labels_path_pair

    def split_dataset(self, seqs_labels_path_pair):
        total_samples = len(seqs_labels_path_pair)
        
        if total_samples == 0:
            raise ValueError("❌ Error: No valid EEG-Label pairs found!")

        # Split percentages
        train_ratio = 1/3
        val_ratio = 1/3
        test_ratio = 1/3

        # Compute split sizes
        num_train = int(total_samples * train_ratio)
        num_val = int(total_samples * val_ratio)

        # Ensure at least 1 sample per set
        if num_train == 0 and total_samples > 1:
            num_train = 1
        if num_val == 0 and total_samples > 2:
            num_val = 1
        
        # Ensure that we have at least 1 sample in the test set
        num_test = total_samples - num_train - num_val
        if num_test == 0:
            num_test = 1
            # Adjust train and val splits if necessary
            if total_samples > 2:
                num_val = total_samples - num_train - num_test
        
        # Split the data
        train_pairs = seqs_labels_path_pair[:num_train]
        val_pairs = seqs_labels_path_pair[num_train:num_train + num_val]
        test_pairs = seqs_labels_path_pair[num_train + num_val:]

        print(f"✅ Dataset split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

        return train_pairs, val_pairs, test_pairs



    def split_dataset(self, seqs_labels_path_pair):
        total_samples = len(seqs_labels_path_pair)
        
        if total_samples == 0:
            raise ValueError("❌ Error: No valid EEG-Label pairs found!")

        # Split percentages
        train_ratio = 1/3
        val_ratio = 1/3
        test_ratio = 1/3

        # Compute split sizes
        num_train = int(total_samples * train_ratio)
        num_val = int(total_samples * val_ratio)

        # Ensure at least 1 sample per set
        if num_train == 0 and total_samples > 1:
            num_train = 1
        if num_val == 0 and total_samples > 2:
            num_val = 1
        
        # Ensure that we have at least 1 sample in the test set
        num_test = total_samples - num_train - num_val
        if num_test == 0:
            num_test = 1
            # Adjust train and val splits if necessary
            if total_samples > 2:
                num_val = total_samples - num_train - num_test
        
        # Split the data
        train_pairs = seqs_labels_path_pair[:num_train]
        val_pairs = seqs_labels_path_pair[num_train:num_train + num_val]
        test_pairs = seqs_labels_path_pair[num_train + num_val:]

        print(f"✅ Dataset split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

        return train_pairs, val_pairs, test_pairs


