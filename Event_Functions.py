import os
import glob
import gc
import numpy as np
import pandas as pd

# -------------------------------------------------
# Event-level calculations
# -------------------------------------------------
def compute_event_cog(df, gev_col):

    print(f"[COG] Computing for {gev_col}")

    work = df[["event", "x", "y", "z", gev_col]].copy()

    work["xw"] = work["x"] * work[gev_col]
    work["yw"] = work["y"] * work[gev_col]
    work["zw"] = work["z"] * work[gev_col]

    cog = work.groupby("event", sort=False).agg(
        x_cog=("xw", "sum"),
        y_cog=("yw", "sum"),
        z_cog=("zw", "sum"),
        event_energy=(gev_col, "sum"),
    )

    mask = cog["event_energy"] > 0

    cog.loc[mask, ["x_cog", "y_cog", "z_cog"]] = (
        cog.loc[mask, ["x_cog", "y_cog", "z_cog"]]
        .div(cog.loc[mask, "event_energy"], axis=0)
    )

    cog.index.name = "event"

    print(f"[COG] {len(cog)} events processed")

    return cog


# -------------------------------------------------
# Moment matrices
# -------------------------------------------------
def compute_moment_matrices(df, gev_col):

    print(f"[MOMENTS] Computing for {gev_col}")

    work = df[["event", "x", "y", "z", gev_col]].copy()

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

    moment_matrices = (
        work.groupby("event", sort=False)[["x", "y", "z", gev_col]]
        .apply(compute)
    )

    print(f"[MOMENTS] {len(moment_matrices)} moment matrices computed")

    return moment_matrices


# -------------------------------------------------
# Orientations (FIXED: stores full axis vector)
# -------------------------------------------------
def compute_orientations(moment_matrices: pd.Series) -> pd.DataFrame:

    print("[ORIENTATION] Computing principal axes")

    records = []

    for event_id, matrix in moment_matrices.items():

        if matrix is None or np.isnan(matrix).any():

            records.append({
                "event": event_id,
                "theta": np.nan,
                "phi": np.nan,
                "principal_axis_x": np.nan,
                "principal_axis_y": np.nan,
                "principal_axis_z": np.nan,
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
                "principal_axis_x": x,
                "principal_axis_y": y,
                "principal_axis_z": z,
            })

        except Exception:

            records.append({
                "event": event_id,
                "theta": np.nan,
                "phi": np.nan,
                "principal_axis_x": np.nan,
                "principal_axis_y": np.nan,
                "principal_axis_z": np.nan,
            })

    df_orient = pd.DataFrame.from_records(records).set_index("event")

    print(f"[ORIENTATION] {len(df_orient)} orientations computed")

    return df_orient


# -------------------------------------------------
# Projection to z = 0 (FIXED and VECTORIZED)
# -------------------------------------------------
def project_to_z0(cog, orient):

    print("[PROJECTION] Projecting to z=0")

    merged = cog.join(orient, how="left")

    vz = merged["principal_axis_z"].to_numpy()

    valid = np.isfinite(vz) & (vz != 0)

    t = np.full(len(merged), np.nan, dtype=np.float64)

    z0 = merged["z_cog"].to_numpy()

    t[valid] = -z0[valid] / vz[valid]

    merged["x_proj"] = merged["x_cog"] + t * merged["principal_axis_x"]
    merged["y_proj"] = merged["y_cog"] + t * merged["principal_axis_y"]

    print(f"[PROJECTION] {valid.sum()} valid projections computed")

    return merged[["x_proj", "y_proj"]]


# -------------------------------------------------
# Layer energies
# -------------------------------------------------
def compute_layer_energies(df, gev_col):

    le = (
        df.groupby(["event", "layer"], sort=False)[gev_col]
        .sum()
        .unstack("layer")
        .fillna(0.0)
    )

    print(f"[LAYER] {gev_col} layer energies shape {le.shape}")

    return le
