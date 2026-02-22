#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import mplhep as hep

# -----------------------
# User Settings
# -----------------------
data_base_path = "/media/miguel/Expansion/ZDC_JLab_test_data"
sim_path       = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1_event_summary.pkl"

bins_theta = 200
bins_phi   = 200

# -----------------------
# Helper Functions
# -----------------------
def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    hep.style.use(hep.style.CMS)

def combine_orient_data(df_list):
    """Combine multiple event_summary dfs into a single orientation dict with low/high/avg columns."""
    data = {}
    for col in ["theta", "phi"]:
        for tag in ["_full", "_low", "_high", "_avg"]:
            key = col + tag
            vals = []
            for df in df_list:
                if key in df:
                    vals.extend(df[key].values)
            arr = np.array(vals)
            if col == "phi":
                arr = np.mod(arr, 2*np.pi)
            data[key] = arr
    return data

def compute_error_band(values, low, high, avg, bins, range_):
    """Compute normalized histogram counts and systematic error bands from full/low/high/avg arrays."""
    # Filter NaNs
    mask = ~(np.isnan(values) | np.isnan(low) | np.isnan(high) | np.isnan(avg))
    values = values[mask]
    low    = low[mask]
    high   = high[mask]
    avg    = avg[mask]

    total_events = len(values)
    if total_events == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    counts, bin_edges = np.histogram(values, bins=bins, range=range_)
    norm_counts = counts / total_events
    stat_err = np.sqrt(counts) / total_events  # statistical error

    counts_low, _  = np.histogram(low, bins=bins, range=range_)
    counts_high, _ = np.histogram(high, bins=bins, range=range_)
    counts_avg, _  = np.histogram(avg, bins=bins, range=range_)

    counts_low  = counts_low / total_events
    counts_high = counts_high / total_events
    counts_avg  = counts_avg / total_events

    # Systematic deviations only
    ytop = np.sqrt(np.maximum(counts_high - norm_counts, counts_low - norm_counts)**2 +
                   (counts_avg - norm_counts)**2)
    ybot = np.sqrt(np.maximum(norm_counts - counts_high, norm_counts - counts_low)**2 +
                   (norm_counts - counts_avg)**2)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, norm_counts, stat_err, ybot, ytop

def plot_theta_phi_distributions(orient_data, orient_sim, bins_theta=50, bins_phi=50):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    coords = ["theta", "phi"]
    bins_dict = {"theta": bins_theta, "phi": bins_phi}
    colors = {"data": "tab:blue", "sim": "tab:orange"}

    for i, coord in enumerate(coords):
        bin_range = (0, 0.5) if coord == "theta" else (0, 2*np.pi)

        # --- Data ---
        bin_centers, norm_counts, stat_err, ybot_sys, ytop_sys = compute_error_band(
            orient_data[f"{coord}_full"],
            orient_data[f"{coord}_low"],
            orient_data[f"{coord}_high"],
            orient_data[f"{coord}_avg"],
            bins=bins_dict[coord],
            range_=bin_range
        )
        if len(norm_counts) > 0:
            max_idx = np.argmax(norm_counts)
            max_bin = bin_centers[max_idx]
            legend_label = f"Data (max @ {max_bin:.2f} rad)"
            axes[i].errorbar(bin_centers, norm_counts, yerr=stat_err,
                             fmt='o', color=colors["data"], markersize=4, capsize=2, label=legend_label)
            axes[i].fill_between(bin_centers, norm_counts - ybot_sys, norm_counts + ytop_sys,
                                 color=colors["data"], alpha=0.3)

        # --- Simulation ---
        sim_vals = orient_sim[coord].dropna().values
        if coord == "phi":
            sim_vals = np.mod(sim_vals, 2*np.pi)
        sim_vals = sim_vals[(sim_vals >= bin_range[0]) & (sim_vals <= bin_range[1])]
        sim_evts = len(sim_vals)
        if sim_evts > 0:
            counts_sim, bin_edges_sim = np.histogram(sim_vals, bins=bins_dict[coord], range=bin_range)
            norm_counts_sim = counts_sim / sim_evts
            stat_err_sim = np.sqrt(counts_sim) / sim_evts
            bin_centers_sim = 0.5 * (bin_edges_sim[:-1] + bin_edges_sim[1:])
            max_idx_sim = np.argmax(norm_counts_sim)
            max_bin_sim = bin_centers_sim[max_idx_sim]
            legend_label_sim = f"Sim (max @ {max_bin_sim:.2f} rad)"
            axes[i].errorbar(bin_centers_sim, norm_counts_sim, yerr=stat_err_sim,
                             fmt='o', color=colors["sim"], markersize=4, capsize=2, label=legend_label_sim)

        axes[i].set_xlabel(f"{coord.capitalize()} [radians]", fontsize=20)
        axes[i].set_ylabel("Norm. counts", fontsize=20)
        axes[i].set_ylim(0, 0.04)
        axes[i].legend(fontsize=16, loc="upper right")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/cluster_angles.pdf")
    plt.show()

# -----------------------
# Load data files
# -----------------------
data_files = sorted(glob(os.path.join(data_base_path, "*-Beam", "*_event_summary.pkl")))
data_dfs = [pd.read_pickle(f) for f in data_files]
orient_data = combine_orient_data(data_dfs)

# -----------------------
# Load simulation
# -----------------------
sim_df = pd.read_pickle(sim_path)
orient_sim = sim_df[["theta", "phi"]]

# -----------------------
# Plot
# -----------------------
configure_plotting()
plot_theta_phi_distributions(orient_data, orient_sim, bins_theta, bins_phi)
