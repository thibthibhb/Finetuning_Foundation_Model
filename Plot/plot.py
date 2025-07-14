import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# === Settings ===
ENTITY = "thibaut_hasle-epfl"
PROJECT = "CBraMod-earEEG-tuning"
PLOT_DIR = "./Plot/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Authenticate and fetch runs ===
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# === Collect run data (updated with dataset_names) ===
data = []
for run in runs:
    if (
        run.state == "finished"
        and "test_on_2017" not in run.name
        and "normal_train_val_test_split" not in run.name
        and "test_4class" not in run.name
    ):
        print(f"🔍 Processing run: {run.name}")
        row = dict(run.summary)
        row.update(run.config)
        row["name"] = run.name
        row["id"] = run.id

        match = re.search(r"ORP_hyper_tune_(0\.\d+)", run.name)
        if match:
            row["orp_train_frac"] = float(match.group(1))
        elif "data_ORP" in row:
            row["orp_train_frac"] = float(row["data_ORP"])
        else:
            print(f"⚠️ Skipping run (no ORP fraction found): {run.name}")
            continue

        # Clean and sort dataset_names for consistent grouping
        raw_datasets = run.config.get("dataset_names", [])
        if isinstance(raw_datasets, str):
            raw_datasets = [ds.strip() for ds in raw_datasets.split(",")]
        dataset_names_cleaned = sorted(raw_datasets)
        row["dataset_combo"] = "+".join(dataset_names_cleaned)

        data.append(row)

# === Create DataFrame ===
df = pd.DataFrame(data)


# === Check what's available ===
print("📋 Available columns in DataFrame:", df.columns.tolist())
print(f"📊 Number of runs collected: {len(df)}")

# === Required columns check ===
required_columns = ["test_kappa", "data_ORP", "num_datasets"]
existing_columns = [col for col in required_columns if col in df.columns]
missing_columns = [col for col in required_columns if col not in df.columns]

if not existing_columns:
    print("❌ None of the required columns are present. Exiting.")
    exit()

if missing_columns:
    print(f"⚠️ Missing columns: {missing_columns}")

# === Drop incomplete entries ===
df = df.dropna(subset=existing_columns)

# === Normalize num_datasets for consistent coloring ===
if "num_datasets" in df.columns:
    df["num_datasets"] = df["num_datasets"].astype(int)

# === Count how many runs have more than 1000 hours of data ===
if "hours_of_data" in df.columns:
    count_over_1000h = (df["hours_of_data"] > 1000).sum()
    print(f"🔍 Number of runs with more than 1000 hours of data: {count_over_1000h}")


plt.figure(figsize=(14, 7))
sns.set(style="whitegrid", font_scale=1.2)

# Ensure categories are ordered
combo_order = sorted(df["dataset_combo"].unique())
df["dataset_combo"] = pd.Categorical(df["dataset_combo"], categories=combo_order, ordered=True)

# Normalize orp_train_frac into a visual-friendly scale (e.g., 0.1–0.6 → size range)
# Optional: round to reduce legend clutter
df["orp_train_frac"] = df["orp_train_frac"].round(2)

# === Keep only the best run per (dataset_combo, hours_of_data, orp_train_frac) ===
group_cols = ["dataset_combo", "hours_of_data", "orp_train_frac"]
df = df.sort_values("test_kappa", ascending=False)
df = df.groupby(group_cols, as_index=False).first()
print(f"✅ Filtered DataFrame to best test_kappa per {group_cols} – now {len(df)} rows")

# Plot with marker size for orp_train_frac
scatter = sns.scatterplot(
    data=df,
    x="orp_train_frac",
    y="test_kappa",
    hue="dataset_combo",
    size="hours_of_data",
    sizes=(40, 300),  # min and max circle size
    palette="colorblind",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.3
)

# Final polish
plt.title("Test Kappa vs Hours of Data by Dataset Combo\n(Circle Size = ORP Train Fraction)", fontsize=15)
plt.xlabel("ORP fraction training set", fontsize=13)
plt.ylabel("Test Kappa", fontsize=13)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Legend")
plt.tight_layout()

# Save
output_path = os.path.join(PLOT_DIR, "scatter_kappa_vs_hours_with_orp_frac_size.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Plot saved to: {output_path}")
plt.show()
