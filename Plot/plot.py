import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import re
import numpy as np
import functools
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === Settings ===
ENTITY = "thibaut_hasle-epfl"
PROJECT = "CBraMod-earEEG-tuning"
PLOT_DIR = "./artifacts/results/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Cached WandB API calls ===
@functools.lru_cache(maxsize=1)
def fetch_wandb_runs(entity, project):
    """Cached WandB data fetching to avoid repeated API calls"""
    print("🔄 Fetching data from WandB API (cached)...")
    api = wandb.Api()
    return api.runs(f"{entity}/{project}")

# === Authenticate and fetch runs ===
runs = fetch_wandb_runs(ENTITY, PROJECT)

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


# === Data Analysis and Visualization Enhancement ===

# Ensure categories are ordered
combo_order = sorted(df["dataset_combo"].unique())
df["dataset_combo"] = pd.Categorical(df["dataset_combo"], categories=combo_order, ordered=True)

# Normalize orp_train_frac into a visual-friendly scale
df["orp_train_frac"] = df["orp_train_frac"].round(2)

# === Keep only the best run per (dataset_combo, hours_of_data, orp_train_frac) ===
group_cols = ["dataset_combo", "hours_of_data", "orp_train_frac"]
df = df.sort_values("test_kappa", ascending=False)
df = df.groupby(group_cols, as_index=False).first()
print(f"✅ Filtered DataFrame to best test_kappa per {group_cols} – now {len(df)} rows")

# Create figure with multiple subplots for comprehensive analysis
fig = plt.figure(figsize=(20, 15))
sns.set(style="whitegrid", font_scale=1.1)

# 1. Hours of Data vs Performance by Dataset Combinations
plt.subplot(2, 3, 1)
scatter1 = sns.scatterplot(
    data=df,
    x="hours_of_data",
    y="test_kappa",
    hue="dataset_combo",
    size="orp_train_frac",
    sizes=(50, 200),
    palette="Set1",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5
)
plt.title("Performance vs Hours of Data\n(Size = ORP Fraction)", fontsize=12, fontweight='bold')
plt.xlabel("Hours of Data", fontsize=11)
plt.ylabel("Test Kappa", fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

# 2. ORP Fraction vs Performance with Subject Analysis
plt.subplot(2, 3, 2)
scatter2 = sns.scatterplot(
    data=df,
    x="orp_train_frac",
    y="test_kappa",
    hue="dataset_combo",
    size="hours_of_data",
    sizes=(50, 200),
    palette="Set1",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5
)
plt.title("Performance vs ORP Subject Fraction\n(Size = Hours of Data)", fontsize=12, fontweight='bold')
plt.xlabel("ORP Fraction in Training", fontsize=11)
plt.ylabel("Test Kappa", fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

# 3. Dataset Specificity Analysis (Box plot)
plt.subplot(2, 3, 3)
if len(df["dataset_combo"].unique()) > 1:
    sns.boxplot(data=df, x="dataset_combo", y="test_kappa", palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title("Performance Distribution\nby Dataset Combination", fontsize=12, fontweight='bold')
    plt.xlabel("Dataset Combination", fontsize=11)
    plt.ylabel("Test Kappa", fontsize=11)

# 4. Data Efficiency Analysis (Hours needed for performance thresholds)
plt.subplot(2, 3, 4)
performance_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_data = []

for threshold in performance_thresholds:
    for combo in df["dataset_combo"].unique():
        combo_data = df[df["dataset_combo"] == combo]
        good_runs = combo_data[combo_data["test_kappa"] >= threshold]
        if len(good_runs) > 0:
            min_hours = good_runs["hours_of_data"].min()
            threshold_data.append({
                "threshold": threshold,
                "dataset_combo": combo,
                "min_hours": min_hours
            })

if threshold_data:
    threshold_df = pd.DataFrame(threshold_data)
    sns.lineplot(data=threshold_df, x="threshold", y="min_hours", 
                hue="dataset_combo", marker="o", linewidth=2)
    plt.title("Minimum Hours Required\nfor Performance Thresholds", fontsize=12, fontweight='bold')
    plt.xlabel("Test Kappa Threshold", fontsize=11)
    plt.ylabel("Minimum Hours of Data", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

# 5. ORP Subject Impact Analysis
plt.subplot(2, 3, 5)
if "orp_train_frac" in df.columns and len(df["orp_train_frac"].unique()) > 1:
    # Create bins for ORP fraction for clearer analysis
    df["orp_bin"] = pd.cut(df["orp_train_frac"], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
    sns.boxplot(data=df, x="orp_bin", y="test_kappa", palette="viridis")
    plt.title("Performance by ORP Subject\nFraction Categories", fontsize=12, fontweight='bold')
    plt.xlabel("ORP Fraction Category", fontsize=11)
    plt.ylabel("Test Kappa", fontsize=11)

# 6. Data Requirements Heatmap
plt.subplot(2, 3, 6)
if len(df) > 5:  # Only create heatmap if we have enough data
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values="test_kappa", 
        index="dataset_combo", 
        columns=pd.cut(df["hours_of_data"], bins=5), 
        aggfunc="mean"
    )
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap="RdYlGn", 
                   cbar_kws={'label': 'Test Kappa'})
        plt.title("Performance Heatmap:\nDataset vs Hours", fontsize=12, fontweight='bold')
        plt.xlabel("Hours of Data (Binned)", fontsize=11)
        plt.ylabel("Dataset Combination", fontsize=11)
        plt.xticks(rotation=45)

plt.tight_layout()

# Save comprehensive plot
output_path = os.path.join(PLOT_DIR, "comprehensive_data_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Comprehensive plot saved to: {output_path}")
plt.show()

# === Additional Analysis: Data Requirements Summary ===
print("\n" + "="*60)
print("📊 DATA REQUIREMENTS ANALYSIS SUMMARY")
print("="*60)

# Best performing configurations
best_overall = df.loc[df["test_kappa"].idxmax()]
print(f"\n🏆 BEST OVERALL PERFORMANCE:")
print(f"   Dataset: {best_overall['dataset_combo']}")
print(f"   Test Kappa: {best_overall['test_kappa']:.3f}")
print(f"   Hours: {best_overall['hours_of_data']:.1f}")
print(f"   ORP Fraction: {best_overall['orp_train_frac']:.2f}")

# Efficiency analysis
print(f"\n⚡ EFFICIENCY ANALYSIS:")
df["efficiency"] = df["test_kappa"] / df["hours_of_data"]
most_efficient = df.loc[df["efficiency"].idxmax()]
print(f"   Most efficient config:")
print(f"   Dataset: {most_efficient['dataset_combo']}")
print(f"   Test Kappa: {most_efficient['test_kappa']:.3f}")
print(f"   Hours: {most_efficient['hours_of_data']:.1f}")
print(f"   Efficiency: {most_efficient['efficiency']:.6f} kappa/hour")

# Dataset combination analysis
print(f"\n📁 DATASET COMBINATION INSIGHTS:")
combo_stats = df.groupby("dataset_combo").agg({
    "test_kappa": ["mean", "max", "count"],
    "hours_of_data": ["mean", "min"]
}).round(3)

for combo in combo_stats.index:
    combo_stat = combo_stats.loc[combo]
    print(f"   {combo}:")
    print(f"     Avg Kappa: {combo_stat[('test_kappa', 'mean')]:.3f}")
    print(f"     Max Kappa: {combo_stat[('test_kappa', 'max')]:.3f}")
    print(f"     Runs: {int(combo_stat[('test_kappa', 'count')])}")
    print(f"     Avg Hours: {combo_stat[('hours_of_data', 'mean')]:.1f}")

# ORP subject analysis
print(f"\n👥 ORP SUBJECT ANALYSIS:")
orp_stats = df.groupby(pd.cut(df["orp_train_frac"], bins=3))["test_kappa"].agg(["mean", "count"]).round(3)
for orp_range, orp_stat in orp_stats.iterrows():
    print(f"   ORP Fraction {orp_range}: Avg Kappa = {orp_stat['mean']:.3f} ({int(orp_stat['count'])} runs)")


print("="*60)

# ===============================================================================
# RESEARCH QUESTION ANALYSIS - EAR-EEG SLEEP STAGING
# ===============================================================================

def analyze_hyperparameter_importance(df):
    """Analyze which hyperparameters matter most for ear-EEG performance"""
    print("\n🔬 HYPERPARAMETER IMPORTANCE ANALYSIS")
    print("="*50)
    from sklearn.preprocessing import LabelEncoder

    # # Safely convert categorical and boolean features
    # if 'optimizer' in df.columns:
    #     df['optimizer'] = LabelEncoder().fit_transform(df['optimizer'].astype(str))

    # if 'scheduler' in df.columns:
    #     df['scheduler'] = LabelEncoder().fit_transform(df['scheduler'].astype(str))

    for b in ['use_amp', 'frozen', 'multi_lr', 'use_weighted_sampler']:
        if b in df.columns:
            df[b] = df[b].fillna(False).astype(int)  # Handle missing values safely


    # Prepare features for analysis
    feature_columns = [
        'lr', 'batch_size', 'weight_decay', 'dropout', 'clip_value',
        'head_type', 'orp_train_frac', 'hours_of_data', 'num_subjects_train', 'epochs',
        'label_smoothing', 'sample_rate',
        'use_amp', 'frozen', 'multi_lr', 'use_weighted_sampler'
    ] #'optimizer', 'scheduler'
    
    # Filter available features
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📊 Analyzing features: {available_features}")
    
    if len(available_features) < 3 or len(df) < 10:
        print("⚠️ Insufficient data for hyperparameter analysis")
        return None
    
    # Prepare data
    X = df[available_features].fillna(df[available_features].median())
    y = df['test_kappa'].fillna(0)
    
    # Remove rows with missing target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) < 5:
        print("⚠️ Too few valid samples for analysis")
        return None
    
    # Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_,
        'feature_type': ['hyperparameter'] * len(available_features)
    }).sort_values('importance', ascending=False)
    
    # Classify feature types for research insights
    data_features = ['hours_of_data', 'num_subjects_train', 'orp_train_frac']
    training_features = ['lr', 'batch_size', 'weight_decay', 'dropout', 'epochs']
    optimization_features = ['head_type', 'clip_value']
    
    for idx, row in importance_df.iterrows():
        if row['feature'] in data_features:
            importance_df.at[idx, 'feature_type'] = 'data_scaling'
        elif row['feature'] in training_features:
            importance_df.at[idx, 'feature_type'] = 'training_config'
        elif row['feature'] in optimization_features:
            importance_df.at[idx, 'feature_type'] = 'optimization'
    
    # Interactive plot
    fig = px.bar(
        importance_df, 
        x='importance', 
        y='feature',
        color='feature_type',
        title="Hyperparameter Importance for Ear-EEG Sleep Staging Performance",
        labels={'importance': 'Feature Importance', 'feature': 'Parameter'},
        color_discrete_map={
            'data_scaling': '#ff7f0e',
            'training_config': '#2ca02c', 
            'optimization': '#1f77b4'
        }
    )
    fig.update_layout(height=600, showlegend=True)
    pio.write_image(fig, os.path.join(PLOT_DIR, "hyperparameter_importance.png"), width=1200, height=800)
    fig.show()
    
    print("\n📈 Top 5 Most Important Parameters:")
    for i, (idx, row) in enumerate(importance_df.head().iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.3f} ({row['feature_type']})")
    
    return importance_df

def analyze_scaling_laws(df):
    """Research Question: Scaling laws with respect to samples, subjects, and data quality"""
    print("\n📏 SCALING LAWS ANALYSIS")
    print("="*50)
    
    # Create scaling analysis plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Performance vs Training Hours", 
            "Performance vs Subject Diversity",
            "Performance vs ORP Data Quality",
            "Efficiency Analysis"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if 'hours_of_data' in df.columns:
        # Scaling with training samples (hours)
        df_sorted = df.sort_values('hours_of_data')
        fig.add_trace(
            go.Scatter(
                x=df_sorted['hours_of_data'], 
                y=df_sorted['test_kappa'],
                mode='markers+lines',
                name='Hours vs Performance',
                marker=dict(size=8, opacity=0.7)
            ),
            row=1, col=1
        )
    
    if 'num_subjects_train' in df.columns:
        # Scaling with subject diversity
        df_sorted = df.sort_values('num_subjects_train')
        fig.add_trace(
            go.Scatter(
                x=df_sorted['num_subjects_train'], 
                y=df_sorted['test_kappa'],
                mode='markers+lines',
                name='Subjects vs Performance',
                marker=dict(size=8, opacity=0.7)
            ),
            row=1, col=2
        )
    
    if 'orp_train_frac' in df.columns:
        # Data quality impact (ORP fraction as proxy for label quality)
        df_sorted = df.sort_values('orp_train_frac')
        fig.add_trace(
            go.Scatter(
                x=df_sorted['orp_train_frac'], 
                y=df_sorted['test_kappa'],
                mode='markers+lines',
                name='ORP Quality vs Performance',
                marker=dict(size=8, opacity=0.7)
            ),
            row=2, col=1
        )
    
    # Efficiency analysis (performance per hour)
    if 'hours_of_data' in df.columns:
        df['efficiency'] = df['test_kappa'] / df['hours_of_data']
        df_sorted = df.sort_values('efficiency', ascending=False)
        fig.add_trace(
            go.Bar(
                x=df_sorted['dataset_combo'][:10], 
                y=df_sorted['efficiency'][:10],
                name='Efficiency (Kappa/Hour)',
                marker=dict(opacity=0.7)
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="Scaling Laws for Ear-EEG Sleep Staging",
        showlegend=True
    )
    
    pio.write_image(fig, os.path.join(PLOT_DIR, "scaling_laws_analysis.png"), width=1200, height=800)

    fig.show()
    
    # Statistical analysis of scaling
    if 'hours_of_data' in df.columns and len(df) > 10:
        # Fit power law: performance = a * hours^b
        log_hours = np.log(df['hours_of_data'].replace(0, 1))
        log_kappa = np.log(df['test_kappa'].replace(0, 0.001))
        
        # Linear regression in log space
        valid_mask = ~(np.isinf(log_hours) | np.isinf(log_kappa))
        if valid_mask.sum() > 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_hours[valid_mask], log_kappa[valid_mask]
            )
            print(f"\n📊 SCALING LAW ESTIMATES:")
            print(f"   Power law exponent: {slope:.3f} ± {std_err:.3f}")
            print(f"   R² correlation: {r_value**2:.3f}")
            print(f"   P-value: {p_value:.3f}")
            print(f"   Interpretation: Performance scales as Hours^{slope:.2f}")

def analyze_task_granularity(df):
    """Research Question: 4-class vs 5-class sleep staging performance"""
    print("\n🎯 TASK GRANULARITY ANALYSIS")
    print("="*50)
    
    # Infer task granularity from num_of_classes or dataset names
    if 'num_of_classes' in df.columns:
        granularity_analysis = df.groupby('num_of_classes').agg({
            'test_kappa': ['mean', 'std', 'count'],
            'test_accuracy': ['mean', 'std'] if 'test_accuracy' in df.columns else ['count'],
            'test_f1': ['mean', 'std'] if 'test_f1' in df.columns else ['count']
        }).round(4)
        
        print("📊 Performance by Number of Classes:")
        for classes in sorted(df['num_of_classes'].unique()):
            mask = df['num_of_classes'] == classes
            subset = df[mask]
            print(f"   {classes}-class staging:")
            print(f"     Kappa: {subset['test_kappa'].mean():.3f} ± {subset['test_kappa'].std():.3f} (n={len(subset)})")
            if 'test_accuracy' in df.columns:
                print(f"     Accuracy: {subset['test_accuracy'].mean():.3f} ± {subset['test_accuracy'].std():.3f}")
        
        # Statistical comparison
        if len(df['num_of_classes'].unique()) >= 2:
            classes = sorted(df['num_of_classes'].unique())
            if len(classes) >= 2:
                group1 = df[df['num_of_classes'] == classes[0]]['test_kappa'].dropna()
                group2 = df[df['num_of_classes'] == classes[1]]['test_kappa'].dropna()
                
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    print(f"\n🔬 Statistical Comparison ({classes[0]} vs {classes[1]} classes):")
                    print(f"   T-statistic: {t_stat:.3f}")
                    print(f"   P-value: {p_value:.3f}")
                    print(f"   Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

def analyze_minimal_data_requirements(df):
    """Research Question: Minimal labeled data required for reliable performance"""
    print("\n💾 MINIMAL DATA REQUIREMENTS ANALYSIS")
    print("="*50)
    
    if 'hours_of_data' not in df.columns:
        print("⚠️ No hours_of_data column found")
        return
    
    # Define performance thresholds (based on commercial baselines)
    commercial_baseline = 0.65  # Typical commercial sleep staging kappa
    good_performance = 0.70
    excellent_performance = 0.75
    
    thresholds = {
        'Commercial Baseline': commercial_baseline,
        'Good Performance': good_performance, 
        'Excellent Performance': excellent_performance
    }
    
    print("🎯 Data Requirements for Performance Thresholds:")
    
    results = []
    for threshold_name, threshold_value in thresholds.items():
        good_runs = df[df['test_kappa'] >= threshold_value]
        if len(good_runs) > 0:
            min_hours = good_runs['hours_of_data'].min()
            median_hours = good_runs['hours_of_data'].median()
            success_rate = len(good_runs) / len(df) * 100
            
            print(f"   {threshold_name} (κ ≥ {threshold_value:.2f}):")
            print(f"     Minimum hours: {min_hours:.1f}")
            print(f"     Median hours: {median_hours:.1f}")
            print(f"     Success rate: {success_rate:.1f}% ({len(good_runs)}/{len(df)} runs)")
            
            results.append({
                'threshold': threshold_name,
                'kappa_threshold': threshold_value,
                'min_hours': min_hours,
                'median_hours': median_hours,
                'success_rate': success_rate
            })
        else:
            print(f"   {threshold_name} (κ ≥ {threshold_value:.2f}): No runs achieved this performance")
    
    # Interactive visualization
    if results:
        results_df = pd.DataFrame(results)
        fig = px.bar(
            results_df, 
            x='threshold', 
            y='min_hours',
            title="Minimal Data Requirements for Different Performance Levels",
            labels={'min_hours': 'Minimum Hours Required', 'threshold': 'Performance Threshold'},
            text='min_hours'
        )
        fig.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
        fig.update_layout(height=500)
        pio.write_image(fig, os.path.join(PLOT_DIR, "minimal_data_requirements.png"), width=1200, height=800)
        fig.show()

def create_research_dashboard(df):
    """Create comprehensive research dashboard"""
    print("\n🎛️ CREATING RESEARCH DASHBOARD")
    print("="*50)
    
    # Multi-dimensional analysis
    if all(col in df.columns for col in ['hours_of_data', 'orp_train_frac', 'test_kappa']):
        df['num_subjects_train'] = df['num_subjects_train'].apply(lambda x: max(x, 1) if pd.notnull(x) else 1)
        fig = px.scatter_3d(
            df,
            x='hours_of_data',
            y='orp_train_frac', 
            z='test_kappa',
            color='dataset_combo',
            size='num_subjects_train' if 'num_subjects_train' in df.columns else None,
            hover_data={
                'use_amp': True if 'use_amp' in df.columns else False,
                'optimizer': True if 'optimizer' in df.columns else False,
                'scheduler': True if 'scheduler' in df.columns else False
            },
            title="3D Performance Landscape: Hours × Quality × Performance",
            labels={
                'hours_of_data': 'Training Hours',
                'orp_train_frac': 'ORP Data Fraction (Quality)',
                'test_kappa': 'Test Kappa'
            }
        )
        fig.update_layout(height=700)
        pio.write_image(fig, os.path.join(PLOT_DIR, "research_dashboard_3d.png"), width=1200, height=800)
        fig.show()

# ===============================================================================
# RUN RESEARCH ANALYSIS
# ===============================================================================

# Run all research analyses
if len(df) > 5:  # Only run if we have sufficient data
    importance_df = analyze_hyperparameter_importance(df)
    analyze_scaling_laws(df)
    analyze_task_granularity(df)
    analyze_minimal_data_requirements(df)
    create_research_dashboard(df)
    
    print(f"\n✅ Research analysis complete! Interactive plots saved to: {PLOT_DIR}")
    print("📁 Generated files:")
    print("   - hyperparameter_importance.html")
    print("   - scaling_laws_analysis.html") 
    print("   - minimal_data_requirements.html")
    print("   - research_dashboard_3d.html")
else:
    print("⚠️ Insufficient data for research analysis (need > 5 runs)")

print("="*60)
