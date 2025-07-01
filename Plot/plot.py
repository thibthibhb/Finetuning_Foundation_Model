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

# === Collect run data ===
data = []
for run in runs:
    if (
        run.state == "finished"
        and "test_on_2017" not in run.name
        and "normal_train_val_test_split" not in run.name
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
            row["orp_train_frac"] = float(row["data_ORP"])  # fallback to config value
        else:
            print(f"⚠️ Skipping run (no ORP fraction found): {run.name}")
            continue


        data.append(row)

# === Create DataFrame ===
df = pd.DataFrame(data)

# === Check what's available ===
print("📋 Available columns in DataFrame:", df.columns.tolist())
print(f"📊 Number of runs collected: {len(df)}")

# === Required columns check ===
required_columns = ["test_kappa", "data_ORP", "hours_of_data", "num_datasets"]
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

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Make sure input is clean
df_plot = df.dropna(subset=["orp_train_frac", "test_kappa", "num_datasets"])

# Convert train fraction to string for grouping
df_plot["orp_train_frac"] = df_plot["orp_train_frac"].astype(str)

# Convert ORP train fraction to ordered categorical for proper sorting
frac_order = sorted(df_plot["orp_train_frac"].unique())
df_plot["orp_train_frac"] = pd.Categorical(df_plot["orp_train_frac"], categories=frac_order, ordered=True)

# Boxplot
sns.boxplot(
    data=df_plot,
    x="orp_train_frac",
    y="test_kappa",
    hue="num_datasets",
    palette="Set2"
)

plt.title("📊 Test Kappa by ORP Train Fraction and # Datasets")
plt.xlabel("ORP Train Fraction")
plt.ylabel("Test Kappa")
plt.legend(title="# Datasets", loc="best")
plt.tight_layout()

# Save to PNG
output_path = os.path.join(PLOT_DIR, "boxplot_kappa_vs_orp_frac.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Plot saved to: {output_path}")
plt.show()
