import os
import gc
import numpy as np
import pandas as pd

from Event_Functions import (
    compute_event_cog,
    compute_moment_matrices,
    compute_orientations,
    project_to_z0,
    compute_layer_energies,
)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
pkl_file = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1.pkl"
sampling = 27.75
gev_MIP = 0.0012
hit_threshold = 0.5 * gev_MIP

# Event cuts
EVENT_ENERGY_CUT = gev_MIP * 10
MIN_HITS = 10

# -------------------------------------------------
# Counters
# -------------------------------------------------
total_events_seen = 0
events_below_cut = 0

# -------------------------------------------------
# Energy columns
# -------------------------------------------------
def apply_sampling_fraction(df):
    df = df.copy()
    df["energy_GeV"] = df["energy_GeV"] * sampling
    df.loc[df["energy_GeV"] < hit_threshold, "energy_GeV"] = 0.0
    return df

# -------------------------------------------------
# Main
# -------------------------------------------------
df = pd.read_pickle(pkl_file)
print(f"[LOAD] {len(df)} rows, columns: {list(df.columns)}")

# Apply sampling fraction calibration
df = apply_sampling_fraction(df)

gev_col = "energy_GeV"

# -------------------------------------------------
# Compute full event energy (for cut)
# -------------------------------------------------
cog_full = compute_event_cog(df, gev_col)

total_events_seen = len(cog_full)

# -------------------------------------------------
# Hit multiplicity per event
# -------------------------------------------------
hit_mask = df[gev_col] >= hit_threshold
hit_mult = (
    df.loc[hit_mask]
      .groupby("event")
      .size()
)
hit_mult = hit_mult.reindex(cog_full.index, fill_value=0)

# -------------------------------------------------
# Combined event cut
# -------------------------------------------------
event_mask = (
    (cog_full["event_energy"] >= EVENT_ENERGY_CUT) &
    (hit_mult >= MIN_HITS)
)

events_below_cut = (~event_mask).sum()
kept_events = cog_full.index[event_mask]

print(
    f"[CUT] Total events: {total_events_seen}, "
    f"below cut: {events_below_cut}, "
    f"kept: {len(kept_events)}"
)

# -------------------------------------------------
# Event summary (masked)
# -------------------------------------------------
event_summary = pd.DataFrame(index=kept_events)
event_summary.index.name = "event"

mm = compute_moment_matrices(df, gev_col)
ori = compute_orientations(mm)
proj = project_to_z0(cog_full, ori)

merged = cog_full.join(ori).join(proj)
merged = merged.loc[kept_events]

event_summary = event_summary.join(merged, how="left")

event_summary.reset_index().to_pickle(
    pkl_file.replace(".pkl", "_event_summary.pkl")
)
print(
    f"[SAVE] Event summary saved "
    f"({len(event_summary)} events)"
)

# -------------------------------------------------
# Layer energies (masked)
# -------------------------------------------------
layer_df = compute_layer_energies(df, gev_col)
layer_df = layer_df.loc[kept_events]

layer_df.reset_index().to_pickle(
    pkl_file.replace(".pkl", "_layer_energy.pkl")
)
print(
    f"[SAVE] Layer energies saved "
    f"({len(layer_df)} events)"
)

# -------------------------------------------------
# Final summary
# -------------------------------------------------
rejected_fraction = (
    100.0 * events_below_cut / total_events_seen
    if total_events_seen > 0 else 0.0
)

print("\n[DONE]")
print(f"  Total events processed : {total_events_seen}")
print(f"  Events below cut       : {events_below_cut}")
print(f"  Rejected fraction (%)  : {rejected_fraction:.2f}")

del df, cog_full, event_summary, layer_df
gc.collect()
