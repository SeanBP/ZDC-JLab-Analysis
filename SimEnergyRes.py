#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplhep as hep
from glob import glob
import matplotlib
matplotlib.use("Agg")
# -----------------------
# Settings
# -----------------------
sim_dir = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim"
bins = 100
rng = (0, 12)
do_fit = True

# -----------------------
# Gaussian
# -----------------------
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# -----------------------
# Resolution model (stochastic only)
# -----------------------
def resolution_func(E, A):
    return A / np.sqrt(E)

def fit_resolution(x, y, yerr, p0=[0.15]):
    popt, pcov = curve_fit(
        resolution_func,
        x,
        y,
        sigma=yerr,
        absolute_sigma=True,
        p0=p0,
        bounds=([0], [10]),
    )
    return popt, pcov

def configure_plotting():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'
    hep.style.use(hep.style.CMS)

# -----------------------
# Collect simulation files
# -----------------------
sim_files = sorted(
    glob(os.path.join(sim_dir, "e*GeV.edm4hep_event_summary.pkl"))
)

beam_energies = []
resolutions = []
res_errors = []

# -----------------------
# Loop over files
# -----------------------
for f in sim_files:

    match = re.search(r"e([\d\.]+)GeV", os.path.basename(f))
    if not match:
        continue

    beam_E = float(match.group(1))
    if beam_E > 10:
        continue
    df = pd.read_pickle(f)
    evt_energy = df["event_energy"].values

    counts_raw, bin_edges = np.histogram(evt_energy, bins=bins, range=rng)
    total = len(evt_energy)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = counts_raw / total

    A_guess = counts.max()
    mu_guess = bin_centers[np.argmax(counts)]
    sigma_guess = np.std(evt_energy)

    try:
        popt, pcov = curve_fit(
            gaussian,
            bin_centers,
            counts,
            p0=[A_guess, mu_guess, sigma_guess],
        )

        _, mu_fit, sigma_fit = popt

        # resolution
        R = sigma_fit / mu_fit

        var_mu = pcov[1, 1]
        var_sigma = pcov[2, 2]
        cov_mu_sigma = pcov[1, 2]

        dR_dsigma = 1 / mu_fit
        dR_dmu = -sigma_fit / mu_fit**2

        var_R = (
            dR_dsigma**2 * var_sigma +
            dR_dmu**2 * var_mu +
            2 * dR_dsigma * dR_dmu * cov_mu_sigma
        )

        R_err = np.sqrt(abs(var_R))

        beam_energies.append(beam_E)
        resolutions.append(R)
        res_errors.append(R_err)

        print(f"E = {beam_E:.1f} GeV  ->  R = {R*100:.3f}% ± {R_err*100:.3f}%")

    except Exception:
        print(f"Fit failed for {f}")
        continue

beam_energies = np.array(beam_energies)
resolutions = np.array(resolutions)
res_errors = np.array(res_errors)

order = np.argsort(beam_energies)
beam_energies = beam_energies[order]
resolutions = resolutions[order]
res_errors = res_errors[order]

# Convert to percent
res_percent = resolutions * 100
res_err_percent = res_errors * 100

# -----------------------
# Plot
# -----------------------
configure_plotting()
plt.figure(figsize=(8,6))

plt.errorbar(
    beam_energies,
    res_percent,
    yerr=res_err_percent,
    fmt='o',
    capsize=3,
    label="Sim"
)

# External data point with symmetric error (use larger of up/down)
data_E = 5.3
data_R = 11.026

data_err_up = 0.017
data_err_down = 0.079
data_err_sym = max(data_err_up, data_err_down)

plt.errorbar(
    [data_E],
    [data_R],
    yerr=[data_err_sym],  # symmetric error
    fmt='o',
    color='orange',
    markersize=8,
    capsize=4,
    label=r"Data (Corrected)"
)

# -----------------------
# Fit stochastic term
# -----------------------
if do_fit:

    popt, pcov = fit_resolution(
        beam_energies,
        resolutions,
        res_errors
    )

    A_fit = popt[0]
    A_err = np.sqrt(pcov[0, 0])

    E_fine = np.linspace(0.5, 10, 300)

    plt.plot(
        E_fine,
        resolution_func(E_fine, A_fit) * 100,
        'r--',
        label=r"$\frac{{{:.1f}\%}}{{\sqrt{{E}}}}$".format(A_fit * 100)
    )

    print("\nStochastic fit result:")
    print(f"A = {A_fit*100:.2f}% ± {A_err*100:.2f}%")

plt.xlabel("Electron Energy [GeV]")
plt.ylabel("Energy Resolution [%]")
plt.xlim(0, 10)
plt.ylim(0, None)
plt.legend(fontsize=20)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/energy_resolution_vs_energy.pdf")
