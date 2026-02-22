#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from glob import glob
from scipy.optimize import curve_fit

# -----------------------
# User settings
# -----------------------
data_base_path = "/media/miguel/Expansion/ZDC_JLab_test_data"
sim_path       = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1_event_summary.pkl"

bins = 100
ranges = {
    "x": (-30, 30),
    "y": (-30, 30),
    "z": (0, 300),
}

outdir = "plots"
os.makedirs(outdir, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def configure_plotting():
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    hep.style.use(hep.style.CMS)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_and_plot_cog_err_band(
    ax,
    data_full, data_low, data_high, data_avg,
    sim_vals,
    coord, rng,
    bins=100,
    color="tab:blue",
    sim_color="tab:orange",
):
    # -----------------------
    # Clean NaNs
    # -----------------------
    data_full = np.asarray(data_full)
    data_low  = np.asarray(data_low)
    data_high = np.asarray(data_high)
    data_avg  = np.asarray(data_avg)
    sim_vals  = np.asarray(sim_vals)

    data_full = data_full[~np.isnan(data_full)]
    data_low  = data_low[~np.isnan(data_low)]
    data_high = data_high[~np.isnan(data_high)]
    data_avg  = data_avg[~np.isnan(data_avg)]
    sim_vals  = sim_vals[~np.isnan(sim_vals)]

    if len(data_full) == 0 or len(sim_vals) == 0:
        return

    # -----------------------
    # Nominal data histogram
    # -----------------------
    counts_raw, bin_edges = np.histogram(data_full, bins=bins, range=rng)
    total = len(data_full)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = counts_raw / total

    # -----------------------
    # Energy-scale systematics
    # -----------------------
    counts_low_raw, _  = np.histogram(data_low,  bins=bins, range=rng)
    counts_high_raw, _ = np.histogram(data_high, bins=bins, range=rng)

    delta_low  = counts_low_raw  / total - counts
    delta_high = counts_high_raw / total - counts

    ytop_sys = np.maximum(delta_low, delta_high)
    ybot_sys = np.maximum(-delta_low, -delta_high)
    ytop_sys[ytop_sys < 0] = 0
    ybot_sys[ybot_sys < 0] = 0

    # -----------------------
    # Calibration systematic
    # -----------------------
    counts_avg_raw, _ = np.histogram(data_avg, bins=bins, range=rng)
    delta_avg = counts_avg_raw / total - counts

    calib_pos = np.maximum(delta_avg, 0)
    calib_neg = np.maximum(-delta_avg, 0)

    # -----------------------
    # Statistical uncertainty
    # -----------------------
    stat_err = np.sqrt(counts_raw) / total

    # -----------------------
    # Gaussian fit (data)
    # -----------------------
    try:
        p0 = [counts.max(), centers[np.argmax(counts)], np.std(data_full)]
        popt, _ = curve_fit(gaussian, centers, counts, p0=p0)
        mu, sigma = popt[1], abs(popt[2])
        data_label = f"Data μ={mu:.1f} mm, σ={sigma:.1f} mm"
        ax.plot(centers, gaussian(centers, *popt), "--", color=color)
    except Exception:
        data_label = "Data (fit failed)"

    # --- Plot statistical error bars for data ---
    ax.errorbar(centers, counts, yerr=stat_err, fmt='o', color=color, markersize=4, capsize=2, label=data_label)
    # --- Plot systematic band for data ---
    ax.fill_between(centers, counts - ybot_sys - calib_neg, counts + ytop_sys + calib_pos,
                    color=color, alpha=0.3)

    # -----------------------
    # Simulation
    # -----------------------
    sim_counts_raw, _ = np.histogram(sim_vals, bins=bins, range=rng)
    sim_total = len(sim_vals)
    sim_counts = sim_counts_raw / sim_total
    sim_stat_err = np.sqrt(sim_counts_raw) / sim_total

    try:
        p0s = [sim_counts.max(), centers[np.argmax(sim_counts)], np.std(sim_vals)]
        popt_s, _ = curve_fit(gaussian, centers, sim_counts, p0=p0s)
        mu_s, sigma_s = popt_s[1], abs(popt_s[2])
        sim_label = f"Sim μ={mu_s:.1f} mm, σ={sigma_s:.1f} mm"
        ax.plot(centers, gaussian(centers, *popt_s), "--", color=sim_color)
    except Exception:
        sim_label = "Sim (fit failed)"

    # --- Plot statistical error bars for simulation ---
    ax.errorbar(centers, sim_counts, yerr=sim_stat_err, fmt='o', color=sim_color, markersize=4, capsize=2, label=sim_label)
    # --- Simulation systematics not included (keep same as before) ---

# -----------------------
# Load and combine DATA
# -----------------------
data = {
    "x": {"full": [], "low": [], "high": [], "avg": []},
    "y": {"full": [], "low": [], "high": [], "avg": []},
    "z": {"full": [], "low": [], "high": [], "avg": []},
}

beam_folders = sorted(glob(os.path.join(data_base_path, "*-Beam")))
for folder in beam_folders:
    evt_files = sorted(glob(os.path.join(folder, "*_event_summary.pkl")))
    for f in evt_files:
        df = pd.read_pickle(f)
        for c in ["x", "y", "z"]:
            for v in ["full", "low", "high", "avg"]:
                key = f"{c}_cog_{v}"
                if key in df.columns:
                    data[c][v].extend(df[key].values)

# -----------------------
# Load SIM (event summary)
# -----------------------
sim_evt_df = pd.read_pickle(sim_path)

sim_cog = {
    "x": sim_evt_df["x_cog"].values,
    "y": sim_evt_df["y_cog"].values,
    "z": sim_evt_df["z_cog"].values,
}

# -----------------------
# Plot
# -----------------------
configure_plotting()
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

for ax, coord in zip(axs, ["x", "y", "z"]):
    fit_and_plot_cog_err_band(
        ax,
        data_full=data[coord]["full"],
        data_low=data[coord]["low"],
        data_high=data[coord]["high"],
        data_avg=data[coord]["avg"],
        sim_vals=sim_cog[coord],
        coord=coord,
        rng=ranges[coord],
        bins=bins,
    )

    ax.set_xlabel(f"{coord.upper()} Center of Gravity Position [mm]", fontsize=30)
    ax.set_ylabel("Norm. Counts", fontsize=30)
    ax.set_ylim(0, 0.16)
    ax.legend(fontsize=25, loc="upper right")

plt.tight_layout()
plt.savefig(f"{outdir}/cog_distributions.pdf")
plt.show()
