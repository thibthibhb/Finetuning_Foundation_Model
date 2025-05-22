import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_real_time_feasibility(df, metric='accuracy', latency_threshold=30):
    df['real_time_feasible'] = df['inference_latency_ms'] <= latency_threshold
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='inference_latency_ms', y=metric, hue='real_time_feasible')
    plt.xlabel("Inference Latency (ms)")
    plt.ylabel(metric.title())
    plt.title(f"{metric.title()} vs Inference Latency (≤{latency_threshold}ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/{metric}_vs_latency_feasibility.png", dpi=300)
    plt.show()

def plot_cost_per_night(df, metric='accuracy'):
    if 'est_cost_per_night_usd' not in df.columns:
        df['est_cost_per_night_usd'] = df['gpu_hours'] * 0.5
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='est_cost_per_night_usd', y=metric, hue='model_name')
    plt.xlabel("Estimated Cost per Night ($)")
    plt.ylabel(metric.title())
    plt.title(f"{metric.title()} vs Cost per Night")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/{metric}_vs_cost_per_night.png", dpi=300)
    plt.show()

def show_top_models(df, metric='kappa', cost_col='gpu_hours', top_k=5):
    df = df.copy()
    df['efficiency'] = df[metric] / df[cost_col]
    top_df = df.sort_values('efficiency', ascending=False).head(top_k)
    print(f"\nTop {top_k} models by {metric}/{cost_col} efficiency:")
    print(top_df[['model_name', 'model_size', metric, cost_col, 'efficiency']])

def plot_metric_vs_cost(df, metric='accuracy', cost_field='petaFLOPs'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=cost_field, y=metric, hue='model_name')
    plt.title(f"{metric.title()} vs. {cost_field}")
    plt.xscale('log')
    plt.xlabel(cost_field)
    plt.ylabel(metric.title())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/{metric}_vs_{cost_field}.png", dpi=300)
    plt.show()

def plot_metric_vs_latency(df, metric='accuracy'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='inference_latency_ms', y=metric, hue='model_name')
    plt.xlabel("Inference Latency (ms)")
    plt.ylabel(metric.title())
    plt.title(f"{metric.title()} vs. Inference Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/{metric}_vs_latency.png", dpi=300)
    plt.show()

def plot_pareto_flexible(df, metric='accuracy', cost_col='gpu_hours'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=cost_col, y=metric, hue='model_name', style='baseline')
    plt.xscale('log')
    plt.title(f"{metric.title()} vs. {cost_col} (Pareto)")
    plt.xlabel(cost_col.replace("_", " ").title())
    plt.ylabel(metric.title())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/pareto_{metric}_vs_{cost_col}.png", dpi=300)
    plt.show()

def plot_data_scaling(df, metric='accuracy'):
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="hours_of_data",
        y=metric,
        hue="model_name",
        marker="o"
    )
    plt.title(f"{metric.title()} vs. Hours of data")
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.xlabel("Hours of data used (log scale)")
    plt.ylabel(metric.title())
    plt.tight_layout()
    os.makedirs("Plot/figures", exist_ok=True)
    plt.savefig(f"Plot/figures/{metric}_vs_hours_of_data.png", dpi=300)
    plt.show()

def plot_model_scaling(df, metric='accuracy'):
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="model_size",
        y=metric,
        hue="data_fraction",
        style="model_name",
        markers=True,
        dashes=False
    )
    plt.title(f"{metric.title()} vs. Model Size")
    plt.xscale('log')
    plt.xlabel("Model Size (parameters, log scale)")
    plt.ylabel(metric.title())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/{metric}_vs_model_size.png", dpi=300)
    plt.show()

def plot_pareto(df, metric='accuracy'):
    df['compute_cost'] = df['model_size'] * df['data_fraction']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='compute_cost', y=metric, hue='model_name', style='baseline')
    plt.title(f"{metric.title()} vs. Compute Cost")
    plt.xscale('log')
    plt.xlabel("Compute (Model Size × Data Fraction)")
    plt.ylabel(metric.title())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot/figures/{metric}_vs_compute.png", dpi=300)
    plt.show()

def plot_scaling_fit(df, metric='error', fit_func='power_law'):
    # Error is 1 - accuracy
    df['error'] = 1.0 - df['accuracy']
    grouped = df.groupby("data_fraction")[metric].mean().reset_index()

    # Fit power law: y = a * x^-b + c
    from scipy.optimize import curve_fit

    def power_law(x, a, b, c):
        return a * x**(-b) + c

    xdata = grouped['data_fraction'].values
    ydata = grouped['error'].values

    popt, _ = curve_fit(power_law, xdata, ydata)
    x_fit = np.linspace(min(xdata), max(xdata), 100)
    y_fit = power_law(x_fit, *popt)

    plt.figure(figsize=(8, 6))
    plt.scatter(xdata, ydata, label='Empirical')
    plt.plot(x_fit, y_fit, 'r--', label=f'Fit: a*x^-b+c')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Data Fraction (log)")
    plt.ylabel("Error (log)")
    plt.title("Scaling Law Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plot/figures/scaling_law_fit.png", dpi=300)
    plt.show()
