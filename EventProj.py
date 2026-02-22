#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix libGL errors on headless servers
import matplotlib.pyplot as plt
from glob import glob
import mplhep as hep
from scipy.optimize import curve_fit

# -----------------------
# User Settings
# -----------------------
data_base_path = "/media/miguel/Expansion/ZDC_JLab_test_data"
sim_path       = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1_event_summary.pkl"
bins = 300
range_x = (-50, 50)
range_y = (-50, 50)

# -----------------------
# Helper functions
# -----------------------
def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    hep.style.use(hep.style.CMS)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

def combine_proj_data(df_list):
    """Combine projected x/y from multiple event_summary dfs"""
    x_full, x_low, x_high, x_avg = [], [], [], []
    y_full, y_low, y_high, y_avg = [], [], [], []
    for df in df_list:
        x_full.extend(df["x_proj_full"].values)
        x_low.extend(df["x_proj_low"].values)
        x_high.extend(df["x_proj_high"].values)
        x_avg.extend(df["x_proj_avg"].values)

        y_full.extend(df["y_proj_full"].values)
        y_low.extend(df["y_proj_low"].values)
        y_high.extend(df["y_proj_high"].values)
        y_avg.extend(df["y_proj_avg"].values)

    return {
        "x": np.array(x_full),
        "x_low": np.array(x_low),
        "x_high": np.array(x_high),
        "x_avg": np.array(x_avg),
        "y": np.array(y_full),
        "y_low": np.array(y_low),
        "y_high": np.array(y_high),
        "y_avg": np.array(y_avg)
    }

def compute_error_band(values, low, high, avg, bins, hist_range):
    mask = ~(np.isnan(values) | np.isnan(low) | np.isnan(high) | np.isnan(avg))
    values, low, high, avg = values[mask], low[mask], high[mask], avg[mask]

    counts_raw, bin_edges = np.histogram(values, bins=bins, range=hist_range)
    total = len(values)
    counts = counts_raw / total
    stat_err = np.sqrt(counts_raw) / total  # Statistical error as error bars

    counts_low, _  = np.histogram(low, bins=bins, range=hist_range)
    counts_high, _ = np.histogram(high, bins=bins, range=hist_range)
    counts_avg, _  = np.histogram(avg, bins=bins, range=hist_range)

    counts_low  = counts_low / total
    counts_high = counts_high / total
    counts_avg  = counts_avg / total

    # Systematic uncertainty bands only (exclude stat_err from band)
    ytop = np.sqrt(np.maximum(counts_high - counts, counts_low - counts)**2 + (counts_avg - counts)**2)
    ybot = np.sqrt(np.maximum(counts - counts_high, counts - counts_low)**2 + (counts - counts_avg)**2)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts, stat_err, ybot, ytop

# -----------------------
# Main plotting routine
# -----------------------
def plot_proj_1d_event_summary(data_proj, sim_df, bins=bins, range_x=range_x, range_y=range_y):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    coords = ["x", "y"]
    colors = {"data": "tab:blue", "sim": "tab:orange"}

    for i, coord in enumerate(coords):
        # ---- Data ----
        bc, counts, stat_err, ybot, ytop = compute_error_band(
            data_proj[coord],
            data_proj[f"{coord}_low"],
            data_proj[f"{coord}_high"],
            data_proj[f"{coord}_avg"],
            bins=bins,
            hist_range=(range_x if coord=="x" else range_y)
        )

        # Gaussian fit for data
        try:
            p0 = [counts.max(), 0, 15] if coord=="x" else [counts.max(), 0, 15]
            popt_d, _ = curve_fit(gaussian, bc, counts, p0=p0)
            A_d, mu_d, sigma_d = popt_d
            if coord == "x":
                sigma_corr = np.sqrt(max(sigma_d**2 - 3.1**2, 0))
            else:
                sigma_corr = np.sqrt(max(sigma_d**2 - 1.44**2, 0))
        except RuntimeError:
            A_d, mu_d, sigma_corr = 0, 0, 0

        # Plot data
        axs[i].errorbar(bc, counts, yerr=stat_err, fmt='o', color=colors["data"], markersize=4, capsize=2,
                        label=f"Data μ={mu_d:.1f} mm, σ={sigma_corr:.1f} mm")
        print(f"{coord}_proj Data μ={mu_d:.1f} mm, σ={sigma_corr:.3f} mm")
        axs[i].fill_between(bc, counts - ybot, counts + ytop, color=colors["data"], alpha=0.3)
        axs[i].plot(bc, gaussian(bc, A_d, mu_d, sigma_d), '--', color=colors["data"])

        # ---- Simulation ----
        sim_vals = sim_df[f"{coord}_proj"].dropna().values
        counts_s_raw, bin_edges = np.histogram(sim_vals, bins=bins, range=(range_x if coord=="x" else range_y))
        counts_s = counts_s_raw / len(sim_vals)
        stat_err_s = np.sqrt(counts_s_raw) / len(sim_vals)
        bc_s = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            p0 = [counts_s.max(), 0, 10]
            popt_s, _ = curve_fit(gaussian, bc_s, counts_s, p0=p0)
            A_s, mu_s, sigma_s = popt_s
        except RuntimeError:
            A_s, mu_s, sigma_s = 0, 0, 0

        axs[i].errorbar(bc_s, counts_s, yerr=stat_err_s, fmt='o', color=colors["sim"], markersize=4, capsize=2,
                        label=f"Sim μ={mu_s:.1f} mm, σ={sigma_s:.1f} mm")
        print(f"Sim μ={mu_s:.1f} mm, σ={sigma_s:.3f} mm")
        axs[i].plot(bc_s, gaussian(bc_s, A_s, mu_s, sigma_s), '--', color=colors["sim"])

        # Formatting
        axs[i].set_ylim(0, 0.08)
        axs[i].set_xlim(-30, 30)
        axs[i].set_xlabel(f"Projected {coord.upper()} Position [mm]")
        axs[i].set_ylabel("Norm. Counts")
        axs[i].legend(loc="upper right", fontsize=16)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/position_res.pdf", bbox_inches="tight")
    plt.show()

# -----------------------
# Load data
# -----------------------
data_files = sorted(glob(os.path.join(data_base_path, "*-Beam", "*_event_summary.pkl")))
data_dfs = [pd.read_pickle(f) for f in data_files]
data_proj = combine_proj_data(data_dfs)

# Load simulation
sim_df = pd.read_pickle(sim_path)

# -----------------------
# Plot
# -----------------------
configure_plotting()
plot_proj_1d_event_summary(data_proj, sim_df)
