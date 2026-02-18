import os
import glob
import gc
import numpy as np
import pandas as pd

# -------------------------------------------------
# Event-level calculations
# -------------------------------------------------
def compute_event_cog(df, gev_col):
    print("Computing cog")
    work = df[["event", "x", "y", "z", gev_col]].copy()
    work["xw"] = work["x"] * work[gev_col]
    work["yw"] = work["y"] * work[gev_col]
    work["zw"] = work["z"] * work[gev_col]

    grouped = work.groupby("event", sort=False)
    cog = grouped.agg(
        x_cog=("xw", "sum"),
        y_cog=("yw", "sum"),
        z_cog=("zw", "sum"),
        event_energy=(gev_col, "sum"),
    )

    mask = cog["event_energy"] > 0
    cog.loc[mask, ["x_cog", "y_cog", "z_cog"]] = (
        cog.loc[mask, ["x_cog", "y_cog", "z_cog"]].div(cog.loc[mask, "event_energy"], axis=0)
    )

    print(f"[compute_event_cog] {len(cog)} events processed for {gev_col}")
    return cog

def compute_moment_matrices(df, gev_col):
    print("Computing moment matricies")
    cols = ["x", "y", "z", gev_col, "event"]
    work = df[cols].copy()

    def compute(group):
        e = group[gev_col].to_numpy(np.float32)
        p = group[["x", "y", "z"]].to_numpy(np.float32)
        mask = e > 0
        if not np.any(mask):
            return np.full((3, 3), np.nan, np.float32)
        e = e[mask]
        p = p[mask]
        w = e / e.sum()
        cog = np.average(p, axis=0, weights=w)
        d = p - cog
        return np.einsum("i,ij,ik->jk", w, d, d)

    # Silence FutureWarning by selecting only data columns
    moment_matrices = work.groupby("event", sort=False)[["x", "y", "z", gev_col]].apply(compute)
    print(f"[compute_moment_matrices] Computed {len(moment_matrices)} moment matrices for {gev_col}")
    return moment_matrices

def compute_orientations(moment_matrices: pd.Series) -> pd.DataFrame:
    print("Computing orientations")
    records = []
    for event_id, matrix in moment_matrices.items():
        if matrix is None or np.isnan(matrix).any():
            records.append({
                "event": event_id,
                "theta": np.nan,
                "phi": np.nan,
            })
            continue
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            idx = np.argmax(eigenvalues)
            axis = eigenvectors[:, idx]
            axis /= np.linalg.norm(axis)
            if axis[2] < 0:
                axis = -axis
            x, y, z = axis
            theta = np.arccos(z)
            phi = np.arctan2(y, x)
            records.append({
                "event": event_id,
                "theta": theta,
                "phi": phi,
            })
        except Exception:
            records.append({
                "event": event_id,
                "theta": np.nan,
                "phi": np.nan,
            })

    df_orient = pd.DataFrame.from_records(records).set_index("event")
    print(f"[compute_orientations] Computed orientations for {len(df_orient)} events")
    return df_orient

def project_to_z0(cog, orient):
    print("Projecting to z0")
    merged = cog.join(orient, how="left")
    z0 = merged["z_cog"]
    vz = np.where(merged["theta"].notna(), np.cos(merged["theta"]), np.nan)
    t = -z0 / vz
    merged["x_proj"] = merged["x_cog"] + t * 0  # placeholder
    merged["y_proj"] = merged["y_cog"] + t * 0
    print(f"[project_to_z0] Computed projections for {len(merged)} events")
    return merged[["x_proj", "y_proj"]]

# -------------------------------------------------
# Layer energies
# -------------------------------------------------
def compute_layer_energies(df, gev_col):
    le = df.groupby(["event", "layer"], sort=False)[gev_col].sum().unstack("layer").fillna(0.0)
    print(f"[compute_layer_energies] Computed layer energies for {gev_col} with shape {le.shape}")
    return le
