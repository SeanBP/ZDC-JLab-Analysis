import pandas as pd
import awkward as ak
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import pandas as pd
import uproot as ur
from scipy.optimize import curve_fit
import os
import re
import gc

def datTXT_to_DF(path, max_events=None):
    with open(path) as f:
        lines = f.read().split('\n')

    data = []
    tags = None
    event_count = 0
    collecting = True

    for line in lines:
        if line.strip() == "" or line.startswith("//"):
            continue

        split = line.split()

        # Count events by detecting lines with 7 items
        if len(split) == 7:
            if max_events is not None and event_count >= max_events:
                collecting = False
                break
            event_count += 1
            #continue  # This is just a separator/indicator line, skip it

        if not collecting or len(split) < 4:
            continue  # skip malformed lines or if we're done collecting

        # First non-comment, valid line sets column headers
        if tags is None:
            tags = split[-3:]  # e.g., LG, HG, etc.
            continue

        CAEN, CAEN_ch, LG, HG = split[:4]
        channel = int(CAEN) * 64 + int(CAEN_ch)

        data.append({
            "CAEN": int(CAEN),
            "CAEN_ch": int(CAEN_ch),
            "channel": channel,
            "LG": float(LG),
            "HG": float(HG)
        })

    return pd.DataFrame(data)

def datROOT_to_DF(events):
    
    # Extract all channel keys
    channel_keys = [key for key in events.keys() if key.startswith("ch_")]

    # Load all channels into an awkward array dict
    channels = {key: events[key].array(library="ak") for key in channel_keys}

    # Create a list of awkward arrays for each channel with extra fields
    channel_arrays = []
    for key, arr in channels.items():
        channel = int(key.split("_")[1])
        arr_with_fields = ak.zip({
            "event": ak.local_index(arr),
            "channel": channel,
            "HG": arr["HG"],
            "LG": arr["LG"],
        })
        channel_arrays.append(arr_with_fields)

    # Concatenate all channel arrays
    all_data = ak.concatenate(channel_arrays)

    # Convert to pandas DataFrame
    df = ak.to_dataframe(all_data).reset_index(drop=True)
    df["TS"] = np.array(events["TS"])[df["event"].values]
    df = df.sort_values(by=["event", "channel"]).reset_index(drop=True)
    df["CAEN_brd"] = df["channel"] // 64
    df["CAEN_ch"] = df["channel"] % 64
    
    return df

def apply_pedestal_corrections(df: pd.DataFrame, ped_df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    # Ensure 'channel' exists in df based on CAEN and CAEN_ch
    if 'channel' not in df.columns and {'CAEN', 'CAEN_ch'}.issubset(df.columns):
        df = df.copy()
        df["channel"] = df["CAEN"] * 64 + df["CAEN_ch"]

    # Drop duplicate columns from pedestal dataframe to avoid _x/_y confusion
    ped_df_clean = ped_df.drop(columns=["CAEN", "CAEN_ch"], errors="ignore")

    # Merge pedestal info
    merged = pd.merge(df, ped_df_clean, on="channel", how="left")

    # Compute pedestal-corrected values
    merged["HG_ped_corr"] = merged["HG"] - merged["HGPedMean"]
    merged["LG_ped_corr"] = merged["LG"] - merged["LGPedMean"]

    # Apply threshold cut
    merged["HG_ped_corr"] = merged["HG_ped_corr"].where(
        merged["HG_ped_corr"] >= threshold * merged["HGPedSigma"], 0
    )
    merged["LG_ped_corr"] = merged["LG_ped_corr"].where(
        merged["LG_ped_corr"] >= threshold * merged["LGPedSigma"], 0
    )

    # Drop the pedestal mean and sigma columns before returning
    columns_to_drop = [
    "HGPedMean", "HGPedSigma",
    "LGPedMean", "LGPedSigma",
    "HGPedMeanErr", "HGPedSigmaErr",
    "LGPedMeanErr", "LGPedSigmaErr",
    ]
    merged = merged.drop(columns=columns_to_drop, errors="ignore")


    return merged

def apply_geometry(df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    # Merge geometry information
    merged = pd.merge(df, geo_df[["channel", "x", "y", "z", "layer", "layer_ch"]], on="channel", how="left")
    
    return merged

def build_pedestal_dataframe(
    txt_path,
    n_caen_units=9,
    lg_max_val=400,
    hg_max_val=2000,
    num_bins=200,
):
    """
    Build a pedestal DataFrame from a CAEN TXT run list.

    Parameters
    ----------
    txt_path : str
        Path to the input TXT file listing runs.
    n_caen_units : int, optional
        Number of CAEN units (default: 9).
    lg_max_val : int, optional
        Upper ADC cut for LG fits (default: 400).
    hg_max_val : int, optional
        Upper ADC cut for HG fits (default: 2000).
    num_bins : int, optional
        Number of histogram bins (default: 200).

    Returns
    -------
    pedestal_df : pandas.DataFrame
        DataFrame containing HG/LG pedestal means, sigmas, and errors
        for each CAEN channel.
    """

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    data_DF = datTXT_to_DF(txt_path)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # --------------------------------------------------
    # Fit helper
    # --------------------------------------------------
    def fit_caen(
        dataframe,
        caen_unit,
        gain_type,
        max_val,
        Peds,
        PedStds,
        PedErrs,
        PedStdErrs,
    ):
        for ch in range(64):
            df_ch = dataframe[
                (dataframe["CAEN"] == caen_unit)
                & (dataframe["CAEN_ch"] == ch)
            ]

            if df_ch.empty:
                Peds.append(np.nan)
                PedStds.append(np.nan)
                PedErrs.append(np.nan)
                PedStdErrs.append(np.nan)
                continue

            raw_data = df_ch[gain_type].to_numpy()
            filtered = raw_data[(raw_data > 0) & (raw_data < max_val)]

            if len(filtered) < 10:
                Peds.append(np.nan)
                PedStds.append(np.nan)
                PedErrs.append(np.nan)
                PedStdErrs.append(np.nan)
                continue

            counts, edges = np.histogram(
                filtered, bins=num_bins, range=(0, max_val)
            )
            centers = 0.5 * (edges[1:] + edges[:-1])
            errors = np.sqrt(counts)

            mask = (counts > 0) & (counts < 5000)
            x = centers[mask]
            y = counts[mask]
            yerr = errors[mask]

            if len(x) < 3:
                Peds.append(np.nan)
                PedStds.append(np.nan)
                PedErrs.append(np.nan)
                PedStdErrs.append(np.nan)
                continue

            init = [np.max(y), np.mean(filtered), np.std(filtered)]

            lsq = LeastSquares(x, y, yerr, gauss)
            m = Minuit(lsq, A=init[0], mu=init[1], sigma=init[2])
            m.limits["sigma"] = (0, None)
            m.migrad()

            if not m.valid:
                Peds.append(np.nan)
                PedStds.append(np.nan)
                PedErrs.append(np.nan)
                PedStdErrs.append(np.nan)
                continue

            vals = m.values
            errs = m.errors

            Peds.append(vals["mu"])
            PedStds.append(vals["sigma"])
            PedErrs.append(errs["mu"])
            PedStdErrs.append(errs["sigma"])

    # --------------------------------------------------
    # Run LG and HG fits
    # --------------------------------------------------
    LGPeds, LGPedStd, LGPedsErr, LGPedStdErr = [], [], [], []
    HGPeds, HGPedStd, HGPedsErr, HGPedStdErr = [], [], [], []

    for i in range(n_caen_units):
        fit_caen(
            data_DF,
            caen_unit=i,
            gain_type="LG",
            max_val=lg_max_val,
            Peds=LGPeds,
            PedStds=LGPedStd,
            PedErrs=LGPedsErr,
            PedStdErrs=LGPedStdErr,
        )

    for i in range(n_caen_units):
        fit_caen(
            data_DF,
            caen_unit=i,
            gain_type="HG",
            max_val=hg_max_val,
            Peds=HGPeds,
            PedStds=HGPedStd,
            PedErrs=HGPedsErr,
            PedStdErrs=HGPedStdErr,
        )

    # --------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------
    num_channels = len(HGPeds)
    channels = np.arange(num_channels)
    CAEN = channels // 64
    CAEN_ch = channels % 64

    pedestal_df = pd.DataFrame(
        {
            "channel": channels,
            "CAEN": CAEN,
            "CAEN_ch": CAEN_ch,
            "HGPedMean": HGPeds,
            "HGPedSigma": HGPedStd,
            "HGPedMeanErr": HGPedsErr,
            "HGPedSigmaErr": HGPedStdErr,
            "LGPedMean": LGPeds,
            "LGPedSigma": LGPedStd,
            "LGPedMeanErr": LGPedsErr,
            "LGPedSigmaErr": LGPedStdErr,
        }
    )

    return pedestal_df


def apply_mip_calibrations(
    data_df: pd.DataFrame,
    mip_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge data with MIP and gain-ratio calibrations and compute
    full and average MIP-calibrated energies.
    """

    print("\n=== apply_mip_calibrations ===")

    # --------------------------------------------------
    # Rename MIPs column
    # --------------------------------------------------
    mip_df = mip_df.rename(columns={"MIPs": "MIPs_full"})

    # --------------------------------------------------
    # Drop conflicting columns
    # --------------------------------------------------
    conflicting_mip = set(data_df.columns).intersection(mip_df.columns) - {"channel"}
    conflicting_ratio = set(data_df.columns).intersection(ratio_df.columns) - {"channel"}

    mip_df = mip_df.drop(columns=conflicting_mip)
    ratio_df = ratio_df.drop(columns=conflicting_ratio)

    # --------------------------------------------------
    # Merge dataframes
    # --------------------------------------------------
    merged = data_df.merge(mip_df, on="channel", how="left")
    merged = merged.merge(ratio_df, on="channel", how="left")

    # --------------------------------------------------
    # Compute averages (nonzero only)
    # --------------------------------------------------
    mip_nonzero = merged["MIPs_full"] > 0
    gain_nonzero = merged["GainRatio"] > 0

    mip_avg = merged.loc[mip_nonzero, "MIPs_full"].mean()
    gain_avg = merged.loc[gain_nonzero, "GainRatio"].mean()


    merged["MIPs_avg"] = mip_avg

    # --------------------------------------------------
    # Compute calibrated energies
    # --------------------------------------------------
    merged["energy_MIP_full"] = np.where(
        merged["MIPs_full"] > 0,
        merged["LG_ped_corr"] * merged["GainRatio"] / merged["MIPs_full"],
        0.0,
    )

    merged["energy_MIP_avg"] = np.where(
        mip_avg > 0,
        merged["LG_ped_corr"] * gain_avg / mip_avg,
        0.0,
    )


    # --------------------------------------------------
    # Drop intermediate column
    # --------------------------------------------------
    merged = merged.drop(columns=["GainRatio"])

    print("=== END apply_mip_calibrations ===\n")
    return merged

def build_fully_calibrated_dataframe(
    root_path: str,
    geo_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    pedestal_df: pd.DataFrame,
    mip_df: pd.DataFrame,
    tree_name: str = "data",
    pedestal_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Build a fully calibrated dataframe starting from a ROOT file.
    """
    print("Opening ROOT file")
    # ROOT → raw dataframe
    with ur.open(root_path) as f:
        events = f[tree_name]
        raw_df = datROOT_to_DF(events)
    print("ROOT file loaded")

    # Pedestal corrections
    ped_corr_df = apply_pedestal_corrections(
        raw_df, pedestal_df, threshold=pedestal_threshold
    )
    print("Pedestal applied")

    # Geometry
    geo_df_applied = apply_geometry(ped_corr_df, geo_df)
    print("Geometry applied")

    # MIP + gain-ratio calibration
    calibrated_df = apply_mip_calibrations(
        geo_df_applied,
        mip_df=mip_df,
        ratio_df=ratio_df,
    )
    cols_to_drop = [
    "CAEN", "CAEN_ch", "CAEN_brd",
    "CAEN_x", "CAEN_y", "MIPs_full", "MIPs_avg"
    ]

    calibrated_df = calibrated_df.drop(columns=cols_to_drop, errors="ignore")
    print("MIP applied")

    return calibrated_df


def calibrate_run_folder(
    run_dir: str,
    calib_dir: str,
    tree_name: str = "raw",
    pedestal_threshold: float = 3.0,
):
    """
    Fully calibrate a run folder efficiently using build_fully_calibrated_dataframe.

    Workflow
    --------
    1. Build pedestal dataframe from Run0_list.txt -> pedestals.pkl
    2. Build gain dataframe from first pedestal-corrected ROOT file -> Ratios.pkl
    3. For each ROOT file, build fully calibrated dataframe and save -> RunX_calibrated.pkl
    """

    
    from Analysis_Functions import (
        build_pedestal_dataframe,
        apply_pedestal_corrections,
        build_gain_dataframe,
        build_fully_calibrated_dataframe
    )

    # --------------------------------------------------
    # Load static calibration inputs
    # --------------------------------------------------
    geo_df = pd.read_pickle(os.path.join(calib_dir, "geometry.pkl"))
    mip_df = pd.read_pickle(os.path.join(calib_dir, "MIPs.pkl"))

    # --------------------------------------------------
    # Step 1: build pedestal dataframe
    # --------------------------------------------------
    txt_path = os.path.join(run_dir, "Run0_list.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing pedestal TXT: {txt_path}")

    print("Building pedestal dataframe")
    pedestal_df = build_pedestal_dataframe(txt_path)
    pedestal_path = os.path.join(run_dir, "pedestals.pkl")
    pedestal_df.to_pickle(pedestal_path)

    # --------------------------------------------------
    # Identify ROOT files
    # --------------------------------------------------
    root_files = sorted(
        [
            f for f in os.listdir(run_dir)
            if f.endswith(".root")
            and f != "Run0.root"
            and re.match(r"Run\d+\.root", f)
        ],
        key=lambda x: int(re.findall(r"\d+", x)[0]),
    )

    if len(root_files) == 0:
        raise RuntimeError("No ROOT files found to process")

    # --------------------------------------------------
    # Step 2: build gain dataframe from first pedestal-corrected file
    # --------------------------------------------------
    first_root_path = os.path.join(run_dir, root_files[0])
    print(f"Building gain dataframe from {root_files[0]}")

    # Use uproot ":raw" syntax
    with ur.open(f"{first_root_path}:raw") as f:
        ped_corr_df = apply_pedestal_corrections(
            datROOT_to_DF(f), pedestal_df, threshold=pedestal_threshold
        )

    # Build gain dataframe using updated function that uses mip_df
    ratio_df = build_gain_dataframe(ped_corr_df, mip_df)
    ratio_path = os.path.join(run_dir, "Ratios.pkl")
    ratio_df.to_pickle(ratio_path)
    del ped_corr_df
    gc.collect()

    # --------------------------------------------------
    # Step 3: process each ROOT file using build_fully_calibrated_dataframe
    # --------------------------------------------------
    for fname in root_files:
        root_path = os.path.join(run_dir, fname)
        out_path = os.path.join(run_dir, fname.replace(".root", "_calibrated.pkl"))

        print(f"Processing and calibrating {fname} from {run_dir}")

        calibrated_df = build_fully_calibrated_dataframe(
            root_path=root_path,
            geo_df=geo_df,
            ratio_df=ratio_df,
            pedestal_df=pedestal_df,
            mip_df=mip_df,
            tree_name=tree_name,
            pedestal_threshold=pedestal_threshold
        )

        # Save fully calibrated dataframe
        calibrated_df.to_pickle(out_path)

        del calibrated_df
        gc.collect()

        print(f"  -> Saved {os.path.basename(out_path)}")

def linear(x, m, b):
    return m * x + b

def build_gain_dataframe(
    ped_corr_df: pd.DataFrame,
    mip_df: pd.DataFrame,
    num_caen: int = 6,
    channels_per_caen: int = 64,
    lg_max: float = 800.0,
    hg_max: float = 7500.0,
    max_gain_err: float = 0.5,
    debug: bool = False
) -> pd.DataFrame:
    """
    Build a gain-ratio dataframe from a pedestal-corrected dataframe,
    fully mimicking the selection used in the old plot_gain_ratios script.

    Parameters
    ----------
    ped_corr_df : pandas.DataFrame
        Must contain columns: ['channel', 'LG_ped_corr', 'HG_ped_corr']
    mip_df : pandas.DataFrame
        Must contain columns: ['channel', 'MIPs']
    num_caen : int
        Number of CAEN units (default: 6)
    channels_per_caen : int
        Channels per CAEN unit (default: 64)
    lg_max : float
        Maximum LG value allowed in fit
    hg_max : float
        Maximum HG value allowed in fit
    max_gain_err : float
        Maximum allowed slope uncertainty before replacing with average
    debug : bool
        Print debug information per channel

    Returns
    -------
    gain_df : pd.DataFrame
        Columns: ['channel', 'CAEN', 'CAEN_ch', 'GainRatio']
    """

    num_channels = num_caen * channels_per_caen
    GainRatios = []
    GainRatiosErr = []

    for ch in range(num_channels):
        data = ped_corr_df[ped_corr_df["channel"] == ch]
        if data.empty:
            # no hits in this channel
            GainRatios.append(np.nan)
            GainRatiosErr.append(np.nan)
            continue

        lg = data["LG_ped_corr"].values
        hg = data["HG_ped_corr"].values

        # per-channel MIP
        mip_row = mip_df[mip_df["channel"] == ch]
        mip_val = float(mip_row["MIPs"].iloc[0]) if not mip_row.empty else 0.0

        # Mask exactly like old plotting function
        mask = (hg > 0) & (lg > 0) & (hg < hg_max) & (lg < lg_max) & (hg > (11 * lg) - 400) & (hg > mip_val / 2)

        if np.count_nonzero(mask) < 5:
            # Not enough points to fit
            GainRatios.append(np.nan)
            GainRatiosErr.append(np.nan)
            if debug:
                print(f"Channel {ch}: Not enough points for fit, mask count = {np.count_nonzero(mask)}")
            continue

        try:
            popt, pcov = curve_fit(linear, lg[mask], hg[mask], p0=[12, 0])
            m_fit, _ = popt
            m_err = np.sqrt(np.diag(pcov))[0]

            GainRatios.append(m_fit)
            GainRatiosErr.append(m_err)

            if debug:
                print(f"Channel {ch}: Fit m={m_fit:.3f} err={m_err:.3f}, mask count={np.count_nonzero(mask)}")

        except Exception as e:
            GainRatios.append(np.nan)
            GainRatiosErr.append(np.nan)
            if debug:
                print(f"Channel {ch}: Fit failed ({e})")

    GainRatios = np.array(GainRatios)
    GainRatiosErr = np.array(GainRatiosErr)

    # Replace bad channels with average of valid slopes
    valid = (GainRatiosErr > 0) & (GainRatiosErr < max_gain_err)
    if np.any(valid):
        valid_avg = np.nanmean(GainRatios[valid])
        GainRatios[~valid] = valid_avg
    else:
        # fallback if all failed
        GainRatios[:] = 0.0

    channels = np.arange(num_channels)
    CAEN = channels // channels_per_caen
    CAEN_ch = channels % channels_per_caen

    gain_df = pd.DataFrame({
        "channel": channels,
        "CAEN": CAEN,
        "CAEN_ch": CAEN_ch,
        "GainRatio": GainRatios
    })

    return gain_df

