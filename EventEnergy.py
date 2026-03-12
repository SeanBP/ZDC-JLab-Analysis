#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplhep as hep
from glob import glob

# -----------------------
# User Settings
# -----------------------
data_base_path = "/media/miguel/Expansion/ZDC_JLab_test_data"
sim_path       = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1_event_summary.pkl"

bins = 500
rng = (0, 10)

# -----------------------
# Helper functions
# -----------------------
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    hep.style.use(hep.style.CMS)

def fit_and_plot_err_band(
    evt_energy,
    evt_energy_low,
    evt_energy_high,
    evt_energy_avg,
    evt_energy_sim,
    label,
    sim_label,
    color,
    sim_color,
    bins=50,
    rng=(3, 9),
):

    # -----------------------
    # Nominal histogram
    # -----------------------
    counts_raw, bin_edges = np.histogram(evt_energy, bins=bins, range=rng)
    total = len(evt_energy)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = counts_raw / total
    stat_err = np.sqrt(counts_raw) / total

    # -----------------------
    # Systematics
    # -----------------------
    counts_low_raw, _  = np.histogram(evt_energy_low,  bins=bins, range=rng)
    counts_high_raw, _ = np.histogram(evt_energy_high, bins=bins, range=rng)

    counts_low  = counts_low_raw  / total
    counts_high = counts_high_raw / total

    delta_low  = counts_low  - counts
    delta_high = counts_high - counts

    ytop_sys_es = np.maximum(delta_low, delta_high)
    ytop_sys_es[ytop_sys_es < 0] = 0.0

    ybot_sys_es = np.maximum(-delta_low, -delta_high)
    ybot_sys_es[ybot_sys_es < 0] = 0.0

    counts_avg_raw, _ = np.histogram(evt_energy_avg, bins=bins, range=rng)
    counts_avg = counts_avg_raw / total
    calib_delta = counts_avg - counts

    calib_pos = np.maximum(calib_delta, 0.0)
    calib_neg = np.maximum(-calib_delta, 0.0)

    ytop_sys = np.sqrt(ytop_sys_es**2 + calib_pos**2)
    ybot_sys = np.sqrt(ybot_sys_es**2 + calib_neg**2)

    # -----------------------
    # Gaussian fit (data)
    # -----------------------
    sigma_fixed = 0.036

    try:
        A_guess = counts.max()
        mu_guess = np.mean(evt_energy)
        sigma_guess = np.std(evt_energy)

        popt, pcov = curve_fit(
            gaussian,
            bin_centers,
            counts,
            p0=[A_guess, mu_guess, sigma_guess],
        )

        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))
        mu_err = perr[1]
        sigma_err = perr[2]

        sigma_corr = np.sqrt(max(0.0, sigma_fit**2 - sigma_fixed**2))

        if sigma_corr > 0:
            sigma_corr_err = abs(sigma_fit / sigma_corr) * sigma_err
        else:
            sigma_corr_err = 0.0

        # --- Resolution (σ/μ)
        res_uncorr = sigma_fit / mu_fit
        res_uncorr_err = res_uncorr * np.sqrt(
            (sigma_err / sigma_fit)**2 +
            (mu_err / mu_fit)**2
        )

        res_corr = sigma_corr / mu_fit
        res_corr_err = (
            res_corr * np.sqrt(
                (sigma_corr_err / sigma_corr)**2 +
                (mu_err / mu_fit)**2
            )
            if sigma_corr > 0 else 0.0
        )

        print("Data:")
        print(f"  Uncorrected resolution = {res_uncorr*100:.3f} ± {res_uncorr_err*100:.3f} %")
        print(f"  Corrected resolution   = {res_corr*100:.3f} ± {res_corr_err*100:.3f} %")
        print(f"  Mean = {mu_fit*1:.3f}")
        # Legend EXACTLY as before, but now using σ/μ (uncorrected)
        fit_label = f"{label} (res={res_uncorr*100:.1f}%)"

        plt.plot(
            bin_centers,
            gaussian(bin_centers, *popt),
            color=color,
            linestyle="--",
        )

    except Exception:
        fit_label = f"{label} (fit failed)"

    # -----------------------
    # Simulation
    # -----------------------
    sim_counts_raw, _ = np.histogram(evt_energy_sim, bins=bins, range=rng)
    sim_total = len(evt_energy_sim)

    sim_counts = sim_counts_raw / sim_total
    sim_stat_err = np.sqrt(sim_counts_raw) / sim_total

    try:
        A_guess_sim = sim_counts.max()
        mu_guess_sim = bin_centers[np.argmax(sim_counts)]
        sigma_guess_sim = np.std(evt_energy_sim[evt_energy_sim < rng[1]])

        popt_sim, pcov_sim = curve_fit(
            gaussian,
            bin_centers,
            sim_counts,
            p0=[A_guess_sim, mu_guess_sim, sigma_guess_sim],
        )

        A_sim, mu_sim, sigma_sim = popt_sim
        perr_sim = np.sqrt(np.diag(pcov_sim))
        mu_sim_err = perr_sim[1]
        sigma_sim_err = perr_sim[2]

        res_sim = sigma_sim / mu_sim
        res_sim_err = res_sim * np.sqrt(
            (sigma_sim_err / sigma_sim)**2 +
            (mu_sim_err / mu_sim)**2
        )

        print("Simulation:")
        print(f"  Resolution = {res_sim*100:.3f} ± {res_sim_err*100:.3f} %")

        sim_fit_label = f"{sim_label} (res={res_sim*100:.1f}%)"

        plt.plot(
            bin_centers,
            gaussian(bin_centers, *popt_sim),
            color=sim_color,
            linestyle="--",
        )

    except Exception:
        sim_fit_label = f"{sim_label} (fit failed)"

    # -----------------------
    # Plotting
    # -----------------------
    plt.errorbar(
        bin_centers,
        counts,
        yerr=stat_err,
        fmt="o",
        color=color,
        markersize=4,
        capsize=2,
        label=fit_label,
    )

    plt.fill_between(
        bin_centers,
        counts - ybot_sys,
        counts + ytop_sys,
        color=color,
        alpha=0.3,
        linewidth=0,
    )

    plt.errorbar(
        bin_centers,
        sim_counts,
        yerr=sim_stat_err,
        fmt="o",
        color=sim_color,
        markersize=4,
        capsize=2,
        label=sim_fit_label,
    )

# -----------------------
# Load data
# -----------------------
data_energies_full = []
data_energies_low  = []
data_energies_high = []
data_energies_avg  = []

beam_folders = sorted(glob(os.path.join(data_base_path, "*1-Beam")))
for folder_path in beam_folders:
    evt_files = sorted(glob(os.path.join(folder_path, "*_event_summary.pkl")))
    for f in evt_files:
        df = pd.read_pickle(f)
        data_energies_full.extend(df["event_energy_full"].values)
        data_energies_low.extend(df["event_energy_low"].values)
        data_energies_high.extend(df["event_energy_high"].values)
        data_energies_avg.extend(df["event_energy_avg"].values)

# -----------------------
# Load simulation
# -----------------------
sim_df = pd.read_pickle(sim_path)
evt_energy_sim = sim_df["event_energy"].values

# -----------------------
# Plot
# -----------------------
configure_plotting()
plt.figure(figsize=(8, 8))

fit_and_plot_err_band(
    data_energies_full,
    data_energies_low,
    data_energies_high,
    data_energies_avg,
    evt_energy_sim,
    label="Data",
    sim_label="Sim",
    color="tab:blue",
    sim_color="tab:orange",
    bins=bins,
    rng=rng,
)

plt.ylim(0, 0.02)
plt.xlim(0, 10)

legend = plt.legend(fontsize=20, loc="upper left")
legend.set_title("ZDC Prototype")

plt.xlabel("Energy [GeV]")
plt.ylabel("Norm. Counts")

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/event_energy.pdf", bbox_inches="tight")
plt.show()
