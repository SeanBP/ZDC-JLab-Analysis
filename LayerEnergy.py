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
sim_path       = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1_layer_energy.pkl"

bins       = 50
hist_range = (0, 2)   # GeV
ncols      = 5

# -----------------------
# Plot style
# -----------------------
def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    hep.style.use(hep.style.CMS)

# -----------------------
# Load data
# -----------------------
def load_data_layer_energy(base_path):
    files = sorted(glob(os.path.join(base_path, "*-Beam", "*_layer_energy.pkl")))
    return [pd.read_pickle(f) for f in files]

def combine_layer_arrays(dfs, tag, layer):
    vals = []
    col = f"{tag}_{float(layer):.1f}"
    for df in dfs:
        if col in df.columns:
            vals.extend(df[col].values)
    return np.asarray(vals)

# -----------------------
# Histogram utilities
# -----------------------
def compute_data_error_band(full, low, high, avg, bins, hist_range):
    mask = ~(np.isnan(full) | np.isnan(low) | np.isnan(high) | np.isnan(avg))
    full, low, high, avg = full[mask], low[mask], high[mask], avg[mask]

    n_events = len(full)
    if n_events == 0:
        return None

    counts, bin_edges = np.histogram(full, bins=bins, range=hist_range)
    counts = counts.astype(float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    norm = counts / n_events
    stat_err = np.sqrt(counts) / n_events

    c_low,  _ = np.histogram(low,  bins=bins, range=hist_range)
    c_high, _ = np.histogram(high, bins=bins, range=hist_range)
    c_avg,  _ = np.histogram(avg,  bins=bins, range=hist_range)

    c_low  = c_low.astype(float)  / n_events
    c_high = c_high.astype(float) / n_events
    c_avg  = c_avg.astype(float)  / n_events

    sys_low  = c_low  - norm
    sys_high = c_high - norm

    ytop_es = np.maximum(sys_high, sys_low)
    ybot_es = np.maximum(-sys_high, -sys_low)

    delta_calib = c_avg - norm
    calib_pos = np.maximum(delta_calib, 0)
    calib_neg = np.maximum(-delta_calib, 0)

    ytop = np.sqrt(stat_err**2 + ytop_es**2 + calib_pos**2)
    ybot = np.sqrt(stat_err**2 + ybot_es**2 + calib_neg**2)

    return bin_centers, norm, ybot, ytop

def compute_sim_hist(sim_vals, bins, hist_range):
    sim_vals = sim_vals[~np.isnan(sim_vals)]
    n_events = len(sim_vals)
    if n_events == 0:
        return None

    counts, bin_edges = np.histogram(sim_vals, bins=bins, range=hist_range)
    counts = counts.astype(float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    norm = counts / n_events
    err  = np.sqrt(counts) / n_events
    return bin_centers, norm, err

# -----------------------
# Per-layer distributions
# -----------------------
def plot_layer_energy_distributions(data_dfs, sim_df):
    layers = sorted(
        int(float(c.split("_", 1)[1]))
        for c in data_dfs[0].columns
        if c.startswith("full_")
    )

    n_layers = len(layers)
    nrows = int(np.ceil(n_layers / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * 4),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        # --- Data arrays ---
        full = combine_layer_arrays(data_dfs, "full", layer)
        low  = combine_layer_arrays(data_dfs, "low",  layer)
        high = combine_layer_arrays(data_dfs, "high", layer)
        avg  = combine_layer_arrays(data_dfs, "avg",  layer)

        data_hist = compute_data_error_band(full, low, high, avg, bins, hist_range)
        sim_hist  = compute_sim_hist(sim_df[layer].values, bins, hist_range)

        ax = axes[i]

        # --- Data ---
        if data_hist is not None:
            bc, norm, ybot, ytop = data_hist
            ax.scatter(
                bc, norm,
                color="tab:blue",
                s=40,
                label=f"Layer {layer} Data"
            )
            ax.fill_between(
                bc, norm - ybot, norm + ytop,
                color="tab:blue",
                alpha=0.3
            )

        # --- Simulation ---
        if sim_hist is not None:
            bc_s, norm_s, err_s = sim_hist
            ax.scatter(
                bc_s, norm_s,
                color="tab:orange",
                s=40,
                label=f"Layer {layer} Sim"
            )
            ax.fill_between(
                bc_s, norm_s - err_s, norm_s + err_s,
                color="tab:orange",
                alpha=0.3
            )

        ax.set_ylim(0, 0.2)
        ax.legend(fontsize=28)

        # Axis labels exactly as before
        if i % ncols == 0:
            ax.set_ylabel("Norm. Counts", fontsize=30)
        if layer >= layers[-5]:
            ax.set_xlabel("Energy [GeV]", fontsize=30)

    # Hide unused pads
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    hep.style.use("CMS")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'

    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/layer_energy.pdf", bbox_inches="tight")
    plt.show()


# -----------------------
# Layer energy summary
# -----------------------
def plot_layer_energy_summary(data_dfs, sim_df):
    layers = sorted(
        int(float(c.split("_", 1)[1]))
        for c in data_dfs[0].columns
        if c.startswith("full_")
    )

    data_means, data_me_lo, data_me_hi = [], [], []
    sim_means,  sim_me_err             = [], []

    data_stds, data_sd_lo, data_sd_hi = [], [], []
    sim_stds,  sim_sd_err             = [], []

    for layer in layers:
        full = combine_layer_arrays(data_dfs, "full", layer)
        low  = combine_layer_arrays(data_dfs, "low",  layer)
        high = combine_layer_arrays(data_dfs, "high", layer)
        avg  = combine_layer_arrays(data_dfs, "avg",  layer)
        sim  = sim_df[layer].values

        mask = ~np.isnan(full)
        full = full[mask]
        sim  = sim[~np.isnan(sim)]

        if len(full) == 0 or len(sim) == 0:
            continue

        # ---- Mean energy ----
        mean = np.mean(full)
        stat = np.std(full) / np.sqrt(len(full))

        mean_lo = np.mean(low)  if len(low)  else mean
        mean_hi = np.mean(high) if len(high) else mean

        data_means.append(mean)
        data_me_lo.append(np.sqrt(stat**2 + (mean - mean_lo)**2))
        data_me_hi.append(np.sqrt(stat**2 + (mean_hi - mean)**2))

        sim_means.append(np.mean(sim))
        sim_me_err.append(np.std(sim) / np.sqrt(len(sim)))

        # ---- Energy spread (std dev) ----
        std  = np.std(full)
        stat = std / np.sqrt(2 * len(full))

        std_lo = np.std(low)  if len(low)  > 1 else std
        std_hi = np.std(high) if len(high) > 1 else std

        data_stds.append(std)
        data_sd_lo.append(np.sqrt(stat**2 + (std - std_lo)**2))
        data_sd_hi.append(np.sqrt(stat**2 + (std_hi - std)**2))

        sim_stds.append(np.std(sim))
        sim_sd_err.append(sim_stds[-1] / np.sqrt(2 * len(sim)))

    layers = np.array(layers)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    # ---- Mean energy ----
    ax[0].scatter(layers, data_means, color="tab:blue", s=40, label="Data")
    ax[0].fill_between(
        layers,
        np.array(data_means) - np.array(data_me_lo),
        np.array(data_means) + np.array(data_me_hi),
        color="tab:blue",
        alpha=0.3
    )

    ax[0].scatter(layers, sim_means, color="tab:orange", s=40, label="Sim")
    ax[0].fill_between(
        layers,
        np.array(sim_means) - np.array(sim_me_err),
        np.array(sim_means) + np.array(sim_me_err),
        color="tab:orange",
        alpha=0.3
    )

    ax[0].set_ylabel("Mean Energy [GeV]", fontsize=30)
    ax[0].legend(fontsize=30)

    # ---- Energy spread ----
    ax[1].scatter(layers, data_stds, color="tab:blue", s=40, label="Data")
    ax[1].fill_between(
        layers,
        np.array(data_stds) - np.array(data_sd_lo),
        np.array(data_stds) + np.array(data_sd_hi),
        color="tab:blue",
        alpha=0.3
    )

    ax[1].scatter(layers, sim_stds, color="tab:orange", s=40, label="Sim")
    ax[1].fill_between(
        layers,
        np.array(sim_stds) - np.array(sim_sd_err),
        np.array(sim_stds) + np.array(sim_sd_err),
        color="tab:orange",
        alpha=0.3
    )

    ax[1].set_ylabel("Energy Spread [GeV]", fontsize=30)
    ax[1].legend(fontsize=30)

    for a in ax:
        a.set_xlabel("Layer", fontsize=30)
        a.tick_params(labelsize=20)

    hep.style.use("CMS")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'

    plt.tight_layout(pad=0.3)
    plt.savefig("plots/layer_energy_summary.pdf", bbox_inches="tight")
    plt.show()


# -----------------------
# Run
# -----------------------
configure_plotting()

data_dfs = load_data_layer_energy(data_base_path)
sim_df   = pd.read_pickle(sim_path)

plot_layer_energy_distributions(data_dfs, sim_df)
plot_layer_energy_summary(data_dfs, sim_df)
