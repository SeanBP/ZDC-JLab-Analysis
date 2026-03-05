#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
import mplhep as hep
from scipy.optimize import curve_fit

# -----------------------
# Settings
# -----------------------
sim_dir = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim"
bins = 300
range_x = (-50, 50)
range_y = (-50, 50)
do_fit = True

# -----------------------
# Models
# -----------------------
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

def resolution_func(E, A, B):
    return np.sqrt((A / np.sqrt(E))**2 + B**2)

def fit_resolution(x, y, yerr):
    popt, pcov = curve_fit(
        resolution_func,
        x,
        y,
        sigma=yerr,
        absolute_sigma=True,
        p0=[10.0, 1.0],
        bounds=([0, 0], [1000, 1000]),
    )
    return popt, pcov

def configure_plotting():
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    hep.style.use(hep.style.CMS)

# -----------------------
# Collect simulation files
# -----------------------
sim_files = sorted(
    glob(os.path.join(sim_dir, "e*GeV.edm4hep_event_summary.pkl"))
)

energies = []
sigma_x_vals = []
sigma_x_errs = []
sigma_y_vals = []
sigma_y_errs = []

# -----------------------
# Loop over beam energies
# -----------------------
for f in sim_files:

    match = re.search(r"e([\d\.]+)GeV", os.path.basename(f))
    if not match:
        continue

    E = float(match.group(1))
    df = pd.read_pickle(f)

    sigmas = {}
    errors = {}

    for coord, rng in zip(["x", "y"], [range_x, range_y]):

        values = df[f"{coord}_proj"].dropna().values
        counts_raw, bin_edges = np.histogram(values, bins=bins, range=rng)
        total = len(values)
        counts = counts_raw / total
        bc = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        try:
            p0 = [counts.max(), 0, 10]
            popt, pcov = curve_fit(gaussian, bc, counts, p0=p0)
            _, _, sigma_fit = popt
            sigma_err = np.sqrt(pcov[2, 2])
        except RuntimeError:
            sigma_fit = 0
            sigma_err = 0

        sigmas[coord] = sigma_fit
        errors[coord] = sigma_err

    energies.append(E)
    sigma_x_vals.append(sigmas["x"])
    sigma_x_errs.append(errors["x"])
    sigma_y_vals.append(sigmas["y"])
    sigma_y_errs.append(errors["y"])

# Convert to arrays and sort
energies = np.array(energies)
order = np.argsort(energies)

energies = energies[order]
sigma_x_vals = np.array(sigma_x_vals)[order]
sigma_x_errs = np.array(sigma_x_errs)[order]
sigma_y_vals = np.array(sigma_y_vals)[order]
sigma_y_errs = np.array(sigma_y_errs)[order]

# -----------------------
# Plot
# -----------------------
configure_plotting()
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# ---- X resolution ----
axs[0].errorbar(
    energies,
    sigma_x_vals,
    yerr=sigma_x_errs,
    fmt='o',
    capsize=3,
    label="Sim"
)

# External data point with symmetric error (use larger of up/down)
data_E = 5.3
data_X = 5.933
X_err_up = 0.517
X_err_down = 0.017
X_err_sym = max(X_err_up, X_err_down)
axs[0].errorbar(
    [data_E],
    [data_X],
    yerr=[X_err_sym],
    fmt='o',
    color='orange',
    markersize=8,
    capsize=4,
    label="Data (Corrected)"
)

if do_fit:
    popt_x, pcov_x = fit_resolution(energies, sigma_x_vals, sigma_x_errs)
    A_x, B_x = popt_x
    E_fine = np.linspace(0.5, 20, 300)
    axs[0].plot(
        E_fine,
        resolution_func(E_fine, A_x, B_x),
        'r--',
        label=r"$\frac{{{:.1f}}}{{\sqrt{{E}}}} \oplus {:.2f}\ \mathrm{{mm}}$".format(A_x, B_x)
    )

axs[0].set_xlabel("Electron Energy [GeV]")
axs[0].set_ylabel("X Resolution [mm]")
axs[0].set_xlim(0, 20)
axs[0].set_ylim(0, 10)
axs[0].legend()

# ---- Y resolution ----
axs[1].errorbar(
    energies,
    sigma_y_vals,
    yerr=sigma_y_errs,
    fmt='o',
    capsize=3,
    label="Sim"
)

# External data point with symmetric error (use larger of up/down)
data_Y = 5.933
Y_err_up = 0.017
Y_err_down = 0.839
Y_err_sym = max(Y_err_up, Y_err_down)
axs[1].errorbar(
    [data_E],
    [data_Y],
    yerr=[Y_err_sym],
    fmt='o',
    color='orange',
    markersize=8,
    capsize=4,
    label="Data (Corrected)"
)

if do_fit:
    popt_y, pcov_y = fit_resolution(energies, sigma_y_vals, sigma_y_errs)
    A_y, B_y = popt_y
    E_fine = np.linspace(0.5, 20, 300)
    axs[1].plot(
        E_fine,
        resolution_func(E_fine, A_y, B_y),
        'r--',
        label=r"$\frac{{{:.1f}}}{{\sqrt{{E}}}} \oplus {:.2f}\ \mathrm{{mm}}$".format(A_y, B_y)
    )

axs[1].set_xlabel("Electron Energy [GeV]")
axs[1].set_ylabel("Y Resolution [mm]")
axs[1].set_xlim(0, 20)
axs[1].set_ylim(0, 10)
axs[1].legend()

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/position_resolution_vs_energy.pdf")
