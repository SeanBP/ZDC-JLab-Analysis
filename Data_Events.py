#!/usr/bin/env python3

import os
import glob
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
base_dir = "/media/miguel/Expansion/ZDC_JLab_test_data"

run_folders = [
    "63-58-1-Beam",
    "63-58-2-Beam",
    "63-58-3-Beam",
    "63-58-4-Beam",
    "63-58-5-Beam",
    "63-58-6-Beam",
]

gev_MIP = 0.0012
sampling = 27.75
calibration = 1.5368
hit_threshold = 0.5 * gev_MIP
MIN_HITS = 10

energy_tags = ["full", "low", "high", "avg"]

# -----------------------------
# Event energy cut (GeV)
# -----------------------------
EVENT_ENERGY_CUT = gev_MIP * 10

# -------------------------------------------------
# Global counters
# -------------------------------------------------
global_event_offset = 0
total_events_seen = 0
events_below_cut = 0

# -------------------------------------------------
# Energy columns
# -------------------------------------------------
def add_energy_columns(df):
    df = df.copy()

    df["energy_GeV_full"] = df["energy_MIP_full"] * gev_MIP * sampling * calibration
    df["energy_GeV_avg"]  = df["energy_MIP_avg"]  * gev_MIP * sampling * calibration

    for col in ["energy_GeV_full", "energy_GeV_avg"]:
        df.loc[df[col] < hit_threshold, col] = 0.0

    df["energy_GeV_low"]  = 0.981 * df["energy_GeV_full"]
    df["energy_GeV_high"] = 1.019 * df["energy_GeV_full"]

    return df

# -------------------------------------------------
# Main loop
# -------------------------------------------------
for run in run_folders:
    run_path = os.path.join(base_dir, run)
    pkl_files = sorted(glob.glob(os.path.join(run_path, "*_calibrated.pkl")))
    print(f"[RUN] Found {len(pkl_files)} calibrated files in {run}")

    for i, pkl in enumerate(pkl_files, start=1):
        print(f"[LOAD] Processing file {i}/{len(pkl_files)}: {pkl}")
        df = pd.read_pickle(pkl)
        print(f"[LOAD] {len(df)} rows, columns: {list(df.columns)}")

        # ---------------------------------
        # Global event numbering
        # ---------------------------------
        df["event"] += global_event_offset
        global_event_offset = df["event"].max() + 1

        # ---------------------------------
        # Energy reconstruction
        # ---------------------------------
        df = add_energy_columns(df)
        df.to_pickle(pkl)
        print(f"[ENERGY] Added energy columns for {len(df)} hits")

        # ---------------------------------
        # Compute FULL event energy first
        # ---------------------------------
        cog_full = compute_event_cog(df, "energy_GeV_full")

        hit_mask = df["energy_GeV_full"] >= hit_threshold
        hit_mult = (
            df.loc[hit_mask]
            .groupby("event")
            .size()
        )
        hit_mult = hit_mult.reindex(cog_full.index, fill_value=0)

        n_events_file = len(cog_full)
        total_events_seen += n_events_file

        # Event-level energy mask
        event_mask = (
            (cog_full["event_energy"] >= EVENT_ENERGY_CUT) &
            (hit_mult >= MIN_HITS)
        )


        n_below = (~event_mask).sum()
        events_below_cut += n_below

        kept_events = cog_full.index[event_mask]

        print(
            f"[CUT] Events in file: {n_events_file}, "
            f"below cut: {n_below}, "
            f"kept: {len(kept_events)}"
        )

        # ---------------------------------
        # Event summary (masked)
        # ---------------------------------
        event_summary = pd.DataFrame(index=kept_events)
        event_summary.index.name = "event"

        for tag in energy_tags:
            gev_col = f"energy_GeV_{tag}"

            cog = compute_event_cog(df, gev_col)
            mm  = compute_moment_matrices(df, gev_col)
            ori = compute_orientations(mm)
            proj = project_to_z0(cog, ori)

            merged = cog.join(ori).join(proj)

            # Apply event cut
            merged = merged.loc[kept_events]

            merged = merged.add_suffix(f"_{tag}")
            event_summary = event_summary.join(merged, how="left")

        event_summary.reset_index().to_pickle(
            pkl.replace("_calibrated.pkl", "_event_summary.pkl")
        )
        print(
            f"[SAVE] Event summary saved "
            f"({len(event_summary)} events)"
        )

        # ---------------------------------
        # Layer energies (masked)
        # ---------------------------------
        layer_frames = []

        for tag in energy_tags:
            gev_col = f"energy_GeV_{tag}"
            le = compute_layer_energies(df, gev_col)

            # Apply same event cut
            le = le.loc[kept_events]

            le = le.add_prefix(f"{tag}_")
            layer_frames.append(le)

        layer_df = pd.concat(layer_frames, axis=1)

        layer_df.reset_index().to_pickle(
            pkl.replace("_calibrated.pkl", "_layer_energy.pkl")
        )
        print(
            f"[SAVE] Layer energies saved "
            f"({len(layer_df)} events)"
        )

        del df, cog_full, event_summary, layer_df
        gc.collect()

# -------------------------------------------------
# Final summary
# -------------------------------------------------
rejected_fraction = 100.0 * events_below_cut / total_events_seen if total_events_seen > 0 else 0.0

print("\n[DONE]")
print(f"  Total events processed : {total_events_seen}")
print(f"  Events below cut       : {events_below_cut}")
print(f"  Rejected fraction (%)  : {rejected_fraction:.2f}")

