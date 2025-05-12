import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
from utils.util import to_tensor

# class CustomDataset(Dataset):
#     def __init__(self, seqs_labels_path_pair):
#         super(CustomDataset, self).__init__()
#         self.seqs_labels_path_pair = seqs_labels_path_pair
        
#     def __len__(self):
#         return len(self.seqs_labels_path_pair)
    
#     def __getitem__(self, idx):
#         seq_path, label_path = self.seqs_labels_path_pair[idx]
#         seq = np.load(seq_path)     # This loads [N, 1, 30, 200] array where N varies by subject
#         label = np.load(label_path) # This loads [N] array where N varies by subject
#         # Create a list of (epoch, label) pairs
#         epoch_label_pairs = []
#         for i in range(len(label)):
#             epoch_label_pairs.append((seq[i], label[i]))
            
#         return epoch_label_pairs
    
#     @staticmethod
#     def collate(batch):
#         # Batch is a list of lists of (epoch, label) pairs
#         # Flatten the batch into a single list of (epoch, label) pairs
#         flattened_batch = []
#         for subject_data in batch:
#             flattened_batch.extend(subject_data)
        
#         # Now create tensors from the flattened batch
#         x_seq = np.array([item[0] for item in flattened_batch], dtype=np.float32)
#         y_label = np.array([item[1] for item in flattened_batch], dtype=np.int64)
        
#         return to_tensor(x_seq), to_tensor(y_label).long()


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
                self.samples.append((seq[i], label[i]))  # Now it's (1, 30, 200), label
        if len(self.samples) == 0:
            print(f"[ERROR] All samples were removed due to integrity checks.")
            raise ValueError("Dataset is empty after integrity filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch, label = self.samples[idx]
        return to_tensor(epoch), torch.tensor(label, dtype=torch.long)

    def check_integrity(self, data, filename):
        """
        Check for NaNs, Infs, extreme values, flat signals.
        """
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


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.seqs_dir = os.path.join(params.datasets_dir, 'seq_npy')
        self.labels_dir = os.path.join(params.datasets_dir, 'label_npy')
        self.seqs_labels_path_pair, self.num_subjects = self.load_path()
        
    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair)
        
        train_set = EpochDataset(train_pairs)
        val_set = EpochDataset(val_pairs)
        test_set = EpochDataset(test_pairs)

        # Compute label distribution for sampler
        labels = [train_set[i][1].item() for i in range(len(train_set))]  # Extract labels
        class_counts = np.bincount(labels)
        print(f"[INFO] Class counts: {class_counts}")
        class_weights = 1. / class_counts
        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"[INFO] #Train: {len(train_set)} | #Val: {len(val_set)} | #Test: {len(test_set)}")

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                shuffle=True,         # <- Use sampler instead of shuffle=True
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                shuffle=False,
            ),
        }
        return data_loader

    # ONLY FOR TAKEDA IDUN DATASET !!!!
    # def load_path(self):
    #     seqs_labels_path_pair = []
    #     subjects = sorted([
    #         f for f in os.listdir(self.seqs_dir)
    #         if f.endswith('_takeda.npy') and f.startswith('S')
    #     ])
        
    #     for fname in subjects:
    #         subject_id = fname.replace('.npy', '')
    #         seq_path = os.path.join(self.seqs_dir, fname)
    #         label_path = os.path.join(self.labels_dir, f"{subject_id}.npy")
            
    #         if os.path.exists(label_path):
    #             seqs_labels_path_pair.append((seq_path, label_path))
    #         else:
    #             print(f"[WARN] Label not found for {subject_id}")
                
    #     return seqs_labels_path_pair

    def load_path(self):
        seqs_labels_path_pair = []
        unique_subjects = set()

        all_files = sorted([
            f for f in os.listdir(self.seqs_dir)
            if f.endswith('.npy')
        ])

        for fname in all_files:
            valid_takeda = fname.startswith('S') and fname.endswith('_takeda.npy')
            valid_openneuro = (
                (fname.startswith('2023-sub-')) and #fname.startswith('2017-sub-') or 
                '_ses-' in fname and
                fname.count('_') in [2, 3]
            )

            if not (valid_takeda or valid_openneuro):
                continue

            subject_id = fname.replace('.npy', '')
            dataset_prefix = fname.split('-')[0]  # '2017' or '2023' or 'S1' -> this is your dataset marker
            subject_key = f"{dataset_prefix}_{subject_id}"  # makes subject ID globally unique
            unique_subjects.add(subject_key)

            seq_path = os.path.join(self.seqs_dir, fname)
            label_path = os.path.join(self.labels_dir, f"{subject_id}.npy")

            if os.path.exists(label_path):
                seqs_labels_path_pair.append((seq_path, label_path))
            else:
                print(f"[WARN] Label not found for {subject_id}")

        return seqs_labels_path_pair, len(unique_subjects)



    
    # def split_dataset(self, seqs_labels_path_pair):
    #     np.random.seed(42)
    #     np.random.shuffle(seqs_labels_path_pair)
        
    #     total = len(seqs_labels_path_pair)
    #     train_end = int(0.8 * total)
    #     val_end = int(0.9 * total)
        
    #     train_pairs = seqs_labels_path_pair[:train_end]
    #     val_pairs = seqs_labels_path_pair[train_end:val_end]
    #     test_pairs = seqs_labels_path_pair[val_end:]
    #     print(f"Total: {len(seqs_labels_path_pair)} | Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    #     return train_pairs, val_pairs, test_pairs
    
    def split_dataset(self, seqs_labels_path_pair):
        # Separate takeda files from the rest
        #takeda_pairs = [pair for pair in seqs_labels_path_pair if 'takeda' in pair[0]]
        other_pairs = [pair for pair in seqs_labels_path_pair if 'takeda' not in pair[0]]

        # Manually split takeda: first 4 to train, rest to test
        # takeda_pairs_train = takeda_pairs[0:5]
        # takeda_pairs_val = takeda_pairs[5:8]
        # takeda_pairs_test = takeda_pairs[8:10]
        
        # Shuffle only the non-takeda pairs
        np.random.seed(42)
        np.random.shuffle(other_pairs)

        # Split the non-takeda files
        total = len(other_pairs)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)

        train_pairs = other_pairs[:train_end]  #takeda_pairs_train # + other_pairs 
        val_pairs =  other_pairs[train_end:val_end] #takeda_pairs_val #other_pairs[train_end:val_end] +
        test_pairs = other_pairs[val_end:]  #takeda_pairs_test
        print(val_pairs, test_pairs)


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
