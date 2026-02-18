#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep  # for CMS style
from glob import glob

# -----------------------
# User Settings
# -----------------------
data_base_path = "/media/miguel/Expansion/ZDC_JLab_test_data"
sim_path       = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1_event_summary.pkl"

bins = 100
rng = (3,9)

# -----------------------
# Helper functions
# -----------------------
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5*((x-mu)/sigma)**2)

def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    hep.style.use(hep.style.CMS)

def fit_and_plot_err_band(evt_energy, evt_energy_low, evt_energy_high, evt_energy_avg,
                          evt_energy_sim, label, sim_label, color, sim_color,
                          bins=50, rng=(3,9)):
    counts_raw, bin_edges = np.histogram(evt_energy, bins=bins, range=rng)
    total = len(evt_energy)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts = counts_raw / total

    # Energy spread systematics
    counts_low_raw, _  = np.histogram(evt_energy_low, bins=bins, range=rng)
    counts_high_raw, _ = np.histogram(evt_energy_high, bins=bins, range=rng)
    counts_low_delta  = (counts_low_raw / total) - counts
    counts_high_delta = (counts_high_raw / total) - counts
    ytop_sys_es = np.maximum(counts_high_delta, counts_low_delta)
    ytop_sys_es[ytop_sys_es < 0] = 0
    ybot_sys_es = np.maximum(-counts_high_delta, -counts_low_delta)
    ybot_sys_es[ybot_sys_es < 0] = 0

    # Calibration systematic (avg)
    counts_avg_raw, _ = np.histogram(evt_energy_avg, bins=bins, range=rng)
    calib_delta = (counts_avg_raw / total) - counts
    calib_pos = np.maximum(calib_delta, 0.0)
    calib_neg = np.maximum(-calib_delta, 0.0)

    # Statistical uncertainty
    stat_err = np.sqrt(counts_raw) / total
    ytop = np.sqrt(ytop_sys_es**2 + calib_pos**2 + stat_err**2)
    ybot = np.sqrt(ybot_sys_es**2 + calib_neg**2 + stat_err**2)
    sigma_fixed = 0.025

    # Gaussian fit
    try:
        A_guess = max(counts)
        mu_guess = np.mean(evt_energy)
        print(mu_guess)
        sigma_guess = np.std(evt_energy)
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[A_guess, mu_guess, sigma_guess])
        sigma_fit = popt[2]
        mu_fit    = popt[1]
        sigma_corr = np.sqrt(sigma_fit**2 - sigma_fixed**2)
        fit_label = f"{label} (res={sigma_corr/mu_fit*100:.1f}%)"
        plt.plot(bin_centers, gaussian(bin_centers, *popt), color=color, linestyle='--')
        print("Data")
        print(mu_fit)
    except Exception as e:
        import traceback
        print("Gaussian fit failed:")
        traceback.print_exc()
        fit_label = f"{label} (fit failed)"

    # Simulation
    sim_counts_raw, _ = np.histogram(evt_energy_sim, bins=bins, range=rng)
    sim_total = len(evt_energy_sim)
    sim_counts = sim_counts_raw / sim_total if sim_total>0 else sim_counts_raw
    sim_err = np.sqrt(sim_counts_raw)/sim_total if sim_total>0 else sim_counts_raw
    try:
        A_guess_sim = max(sim_counts)
        mu_guess_sim = bin_centers[np.argmax(sim_counts)]
        sigma_guess_sim = np.std(evt_energy_sim[np.array(evt_energy_sim)<9])
        popt_sim, _ = curve_fit(gaussian, bin_centers, sim_counts, p0=[A_guess_sim, mu_guess_sim, sigma_guess_sim])
        sim_fit_label = f"{sim_label} (res={(popt_sim[2]/popt_sim[1])*100:.1f}%)"
        plt.plot(bin_centers, gaussian(bin_centers, *popt_sim), color=sim_color, linestyle='--')
        print("Sim")
        print(popt_sim[1])
    except Exception:
        sim_fit_label = f"{sim_label} (fit failed)"

    # Plot points and error bands
    plt.scatter(bin_centers, counts, color=color, label=fit_label, s=20)
    plt.fill_between(bin_centers, counts- ybot, counts + ytop, color=color, alpha=0.3)
    plt.scatter(bin_centers, sim_counts, color=sim_color, label=sim_fit_label, s=20)
    plt.fill_between(bin_centers, sim_counts - sim_err, sim_counts + sim_err, color=sim_color, alpha=0.3)

# -----------------------
# Load all event energy summaries
# -----------------------
data_energies_full  = []
data_energies_low   = []
data_energies_high  = []
data_energies_avg   = []

beam_folders = sorted(glob(os.path.join(data_base_path, "*-Beam")))
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
# Convert hit energy by sampling fraction

# Sum over hits per event
evt_energy_sim = sim_df["event_energy"]


# -----------------------
# Make the plot
# -----------------------
configure_plotting()
plt.figure(figsize=(8,8))

fit_and_plot_err_band(
    data_energies_full, data_energies_low, data_energies_high, data_energies_avg,
    evt_energy_sim, label="Data", sim_label="Sim",
    color='tab:blue', sim_color='tab:orange',
    bins=bins, rng=rng
)

plt.ylim(0,0.08)
plt.xlim(3,9)
plt.legend(fontsize=20, loc="upper right")
plt.xlabel("Energy [GeV]")
plt.ylabel("Norm. Counts")
os.makedirs("plots", exist_ok=True)
plt.savefig('plots/event_energy.pdf', bbox_inches='tight')
plt.show()
