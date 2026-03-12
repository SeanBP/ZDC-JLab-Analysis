import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataframe
df = pd.read_pickle("/media/miguel/Expansion/ZDC_JLab_test_data/63-58-1-Beam/Run1_calibrated.pkl")

# Ensure output directory exists
os.makedirs("./plots", exist_ok=True)

# Data
data = df["energy_MIP_full"].dropna().values

# Histogram
bins = 500
counts, edges = np.histogram(data, bins=bins, range = [0,2])
centers = 0.5 * (edges[:-1] + edges[1:])
errors = np.sqrt(counts)

# Plot
plt.figure()
plt.errorbar(centers, counts, yerr=errors, fmt='o', markersize=2)
plt.xlabel("energy_MIP_full")
plt.ylabel("Counts")
plt.yscale("log")
# Save
plt.savefig("./plots/energy_MIP_full_hist.png", dpi=300, bbox_inches="tight")
plt.close()
