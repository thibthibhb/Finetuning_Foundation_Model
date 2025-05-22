import os
import numpy as np
from torch.utils.data import Dataset
from utils.util import to_tensor


class PretrainingDataset(Dataset):
    def __init__(self, dataset_dir, patch_num=30, patch_size=200):
        super(PretrainingDataset, self).__init__()
        self.file_paths = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith('.npy')
        ]
        self.file_paths.sort()
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.total_len = patch_num * patch_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])  # shape: (time, channels)
        data = data.T  # shape: (channels, time)

        # Pad or truncate to fixed length
        if data.ndim == 1:
            ch, t = 1, data.shape[0]
            data = data[np.newaxis, :]  # convert to shape (1, time)
        else:
            ch, t = data.shape
        if t < self.total_len:
            pad_width = self.total_len - t
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        else:
            data = data[:, :self.total_len]

        data = data.reshape(ch, self.patch_num, self.patch_size)  # shape: (channels, patch_num, patch_size)
        return to_tensor(data)
