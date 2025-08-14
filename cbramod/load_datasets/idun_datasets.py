import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from utils.util import to_tensor

import numpy as np
from scipy.signal import iirnotch, filtfilt, savgol_filter
# Optional: import pywt for wavelet denoising

def preprocess_epoch(epoch, sfreq):
    # 1) DC offset / baseline
    #epoch = epoch - np.mean(epoch, axis=-1, keepdims=True)

    # # 2) Artifact removal - remove epochs with extreme amplitudes (likely artifacts)
    # amplitude_thresh = np.percentile(np.abs(epoch), 99.5)  # 99.5th percentile threshold
    # if np.max(np.abs(epoch)) > 5 * amplitude_thresh:
    #     # Replace extreme values with median
    #     epoch = np.clip(epoch, -3 * amplitude_thresh, 3 * amplitude_thresh)
    
    # # 3) Savitzky-Golay smoothing (window 11 samples, polyorder 2)
    # epoch = savgol_filter(epoch, window_length=11, polyorder=2, axis=-1)
    
    # # 4) Z-score normalization per epoch (crucial for cross-subject consistency)
    # epoch_std = np.std(epoch, axis=-1, keepdims=True)
    # if epoch_std > 1e-6:  # Avoid division by zero
    #     epoch = epoch / (epoch_std + 1e-8)
    
    # # 5) Robust scaling - clip extreme outliers after normalization
    # epoch = np.clip(epoch, -4, 4)  # Remove values beyond ±4 standard deviations
    
    # 6) (Optional) Linear detrend - ENABLED for better baseline stability
    # t = np.arange(epoch.shape[-1])
    # coeffs = np.polyfit(t, epoch.flatten() if epoch.ndim == 1 else epoch.squeeze(), 1)
    # trend = np.polyval(coeffs, t)
    # epoch = epoch - trend.reshape(epoch.shape)
    
    # 7) (Optional) Wavelet denoising - ENABLED for noise reduction
    # coeffs = pywt.wavedec(epoch, 'db4', level=3, axis=-1)
    # thresh = np.median(np.abs(coeffs[-1])) / 0.6745
    # denoised = [pywt.threshold(c, thresh, mode='soft') for c in coeffs]
    # epoch = pywt.waverec(denoised, 'db4', axis=-1)
    
    return epoch
            
    
class MemoryEfficientKFoldDataset(Dataset):
    def __init__(self, seqs_labels_path_pair, num_of_classes=5, do_preprocess: bool = False, sfreq: float = 200.0, label_mapping_version='v1'):
        # ✨ store the flags on the object
        self.do_preprocess = do_preprocess
        self.sfreq = sfreq        
        
        self.samples = []
        self.metadata = []
        self.num_of_classes = num_of_classes
        self.label_mapping_version = label_mapping_version
        
        # Log the label mapping being used
        if num_of_classes == 4:
            if label_mapping_version == 'v0':
                print(f"🏷️ Using 4-class LEGACY mapping (v0): Awake=Wake+N1, Light=N2, Deep=N3, REM=REM")
            elif label_mapping_version == 'v1':
                print(f"🏷️ Using 4-class NEW mapping (v1): Awake=Wake, Light=N1+N2, Deep=N3, REM=REM")
        else:
            print(f"🏷️ Using {num_of_classes}-class mapping: Wake, N1, N2, N3, REM")

        for seq_path, label_path in seqs_labels_path_pair:
            base = os.path.basename(seq_path)
            sid = base.split("_")[0]

            try:
                seq = np.load(seq_path)
                if self.do_preprocess:
                    for i in range(seq.shape[0]):
                        # apply to each epoch + channel
                        seq[i] = preprocess_epoch(seq[i], self.sfreq)
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

            # Note: Label remapping is handled by subclasses if needed
            label = label.astype(int)
            
            # Validate raw labels before processing
            try:
                self.validate_labels(label, 5)  # Original data should have 5 classes [0,1,2,3,4]
            except ValueError as e:
                print(f"⚠️ Warning in file {base}: {e}")
                # Continue processing but log the warning

            processed_labels = []
            for i in range(len(label)):
                if label[i] is None or label[i] < 0:
                    continue
                
                # Apply label remapping if num_of_classes == 4
                final_label = self.remap_label(int(label[i])) if self.num_of_classes == 4 else int(label[i])
                processed_labels.append(final_label)
                
                self.samples.append((seq[i], final_label))
                self.metadata.append({
                    'subject': sid,
                    'file': base,
                    'index': i,
                    'path': seq_path
                })
            
            # Validate final processed labels
            if processed_labels:
                try:
                    self.validate_labels(processed_labels, self.num_of_classes)
                except ValueError as e:
                    raise ValueError(f"Label validation failed for file {base} after processing: {e}")

        if len(self.samples) == 0:
            raise ValueError("All samples filtered out. Empty dataset.")

        print(f"[✅] Loaded total {len(self.samples)} valid samples from {len(seqs_labels_path_pair)} files.")

    # Create a mapping for 4 classes
    # Uncomment if you want to remap labels
    
    def validate_labels(self, labels, expected_classes):
        """
        Validate that all labels are within the expected range [0, expected_classes-1].
        
        Args:
            labels: Array or list of label values
            expected_classes: Number of expected classes (labels should be 0 to expected_classes-1)
            
        Raises:
            ValueError: If any labels are outside the valid range
        """
        unique_labels = set(labels)
        valid_range = set(range(expected_classes))
        if not unique_labels.issubset(valid_range):
            invalid_labels = unique_labels - valid_range
            raise ValueError(f"Invalid labels found: {sorted(invalid_labels)}. Expected range: [0, {expected_classes-1}]")

    def remap_label(self, l):
        """Remap 5-class labels to 4-class based on mapping version.
        
        Original 5-class: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        
        v0 (legacy): Awake=Wake+N1, Light=N2, Deep=N3, REM=REM
        v1 (new):    Awake=Wake,    Light=N1+N2, Deep=N3, REM=REM
        """
        if self.label_mapping_version == 'v0':
            # Legacy mapping: Awake=Wake+N1, Light=N2, Deep=N3, REM=REM
            if l in [0, 1]:  # Wake + N1 -> Awake
                return 0
            elif l == 2:     # N2 -> Light  
                return 1
            elif l == 3:     # N3 -> Deep
                return 2
            elif l == 4:     # REM -> REM
                return 3
            else:
                raise ValueError(f"Unknown label value: {l}")
        
        elif self.label_mapping_version == 'v1':
            # New mapping: Awake=Wake, Light=N1+N2, Deep=N3, REM=REM
            if l == 0:       # Wake -> Awake
                return 0
            elif l in [1, 2]: # N1 + N2 -> Light
                return 1
            elif l == 3:     # N3 -> Deep
                return 2
            elif l == 4:     # REM -> REM
                return 3
            else:
                raise ValueError(f"Unknown label value: {l}")
        
        else:
            raise ValueError(f"Unknown label_mapping_version: {self.label_mapping_version}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch, label = self.samples[idx]
        sid = self.metadata[idx]['subject']   # <- already tracked in self.metadata
        return to_tensor(epoch), torch.tensor(label, dtype=torch.long), sid

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

def get_custom_split(dataset, n_splits=3, seed=42, orp_train_frac=0.6):
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
    n_train = round(orp_train_frac * n)
    x = n - n_train
    n_val = round(x/2)

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
        f" ({num_subjects_train} subjects train),"
        f" {len(val_indices)} val,"
        f" ({len(orp_val_subjects)} subjects val),"
        f" {len(test_indices)} test"
        f" ({len(orp_test_subjects)} subjects test),"
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
        self.num_of_classes = getattr(params, 'num_of_classes', 5)

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
