import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

base_path = "/root/cbramod/CBraMod/OpenNeuro_2019"
subjects = [f"sub-{i:03d}" for i in range(1, 21)]
sessions = [f"ses-{i:03d}" for i in range(1, 5)]
epoch_length = 30  # seconds

results = []

for sub in subjects:
    for ses in sessions:
        try:
            path1 = f"{base_path}/{sub}/{ses}/eeg/{sub}_{ses}_task-sleep_acq-scoring1_events.tsv"
            path2 = f"{base_path}/{sub}/{ses}/eeg/{sub}_{ses}_task-sleep_acq-scoring2_events.tsv"

            if not os.path.exists(path1) or not os.path.exists(path2):
                print(f"Missing files: {sub}, {ses}")
                continue

            df1 = pd.read_csv(path1, sep='\t')
            df2 = pd.read_csv(path2, sep='\t')

            # Detect scoring label column automatically
            label_col1 = [col for col in df1.columns if col not in ['onset', 'duration']][0]
            label_col2 = [col for col in df2.columns if col not in ['onset', 'duration']][0]

            def label_to_epochs(df, label_col):
                epochs = []
                for _, row in df.iterrows():
                    n_epochs = int(row['duration'] // epoch_length)
                    label = row[label_col]
                    epochs.extend([label] * n_epochs)
                return epochs

            epochs1 = label_to_epochs(df1, label_col1)
            epochs2 = label_to_epochs(df2, label_col2)

            min_len = min(len(epochs1), len(epochs2))
            if min_len == 0:
                print(f"No overlapping epochs for {sub}, {ses}")
                continue

            e1 = epochs1[:min_len]
            e2 = epochs2[:min_len]

            kappa = cohen_kappa_score(e1, e2)
            accuracy = sum(a == b for a, b in zip(e1, e2)) / min_len

            results.append((sub, ses, kappa, accuracy))

        except Exception as e:
            print(f"Error processing {sub} {ses}: {e}")
            continue

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Subject", "Session", "Cohen_Kappa", "Accuracy"])
csv_path = os.path.join(base_path, "interrater_agreement_results.csv")
results_df.to_csv(csv_path, index=False)

# Compute means
mean_kappa = results_df["Cohen_Kappa"].mean()
mean_accuracy = results_df["Accuracy"].mean()

# Print
print(f"\nSaved results to: {csv_path}")
print("\nFinal Interrater Agreement Table:")
print(results_df)

print(f"\nMean Cohen's Kappa across all sessions: {mean_kappa:.4f}")
print(f"Mean Accuracy (percent agreement) across all sessions: {mean_accuracy:.4f}")
