import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_data_scaling(df, metric='accuracy'):
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="data_fraction",
        y=metric,
        hue="model_name",
        marker="o"
    )
    plt.title(f"{metric.title()} vs. Data Fraction")
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.xlabel("Fraction of Fine-Tuning Data (log scale)")
    plt.ylabel(metric.title())
    plt.tight_layout()
    os.makedirs("Plot/figures", exist_ok=True)
    plt.savefig(f"Plot/figures/{metric}_vs_data_fraction.png", dpi=300)
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
