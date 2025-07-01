import os

# Define paths
eeg_path = "/root/cbramod/CBraMod/Final_dataset/1_ORP_dataset/eeg_data_npy"
label_path = "/root/cbramod/CBraMod/Final_dataset/1_ORP_dataset/labels_npy"

# Rename EEG files
for filename in os.listdir(eeg_path):
    if filename.endswith("_eeg_chan_idun.npy"):
        new_name = filename.replace("_eeg_chan_idun", "")
        os.rename(os.path.join(eeg_path, filename), os.path.join(eeg_path, new_name))
        print(f"Renamed: {filename} -> {new_name}")

# Rename label files
for filename in os.listdir(label_path):
    if filename.endswith("_labels.npy"):
        new_name = filename.replace("_labels", "")
        os.rename(os.path.join(label_path, filename), os.path.join(label_path, new_name))
        print(f"Renamed: {filename} -> {new_name}")
