import os

def rename_files(dataset_dir):
    eeg_dir = os.path.join(dataset_dir, 'eeg_data_npy')
    label_dir = os.path.join(dataset_dir, 'label_npy')

    # Rename EEG files
    for fname in os.listdir(eeg_dir):
        if fname.endswith('_eeg_chan_idun.npy'):
            new_name = fname.replace('_eeg_chan_idun', '')
            old_path = os.path.join(eeg_dir, fname)
            new_path = os.path.join(eeg_dir, new_name)
            os.rename(old_path, new_path)
            print(f"[EEG] Renamed: {fname} -> {new_name}")

    # Rename label files
    for fname in os.listdir(label_dir):
        if fname.endswith('_labels.npy'):
            new_name = fname.replace('_labels', '')
            old_path = os.path.join(label_dir, fname)
            new_path = os.path.join(label_dir, new_name)
            os.rename(old_path, new_path)
            print(f"[Label] Renamed: {fname} -> {new_name}")

if __name__ == "__main__":
    # Change this to the root directory of your dataset
    dataset_dir = "//root/cbramod/CBraMod/Final_dataset/"
    rename_files(dataset_dir)