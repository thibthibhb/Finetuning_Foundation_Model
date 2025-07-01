import numpy as np

# Load the label file
labels = np.load("Final_dataset/Tuning_dataset/label_npy/S016_night1.npy")

# Get unique classes and their counts
unique_labels, counts = np.unique(labels, return_counts=True)

# Print number of labels per class
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} labels")