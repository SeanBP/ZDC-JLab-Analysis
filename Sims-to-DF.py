import uproot as ur
import awkward as ak
import pandas as pd
import numpy as np
import pickle
import glob
import os

# ------------------------------------------------------
# Input/output directory
# ------------------------------------------------------
input_dir  = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/"
output_dir = input_dir  # change if needed

root_files = sorted(glob.glob(os.path.join(input_dir, "*.root")))

print(f"Found {len(root_files)} ROOT files")

for input_file in root_files:

    print(f"\nProcessing: {input_file}")

    events = ur.open(f"{input_file}:events")

    # Load jagged arrays
    energy = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.energy"].array()
    x = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.position.x"].array()
    y = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.position.y"].array()
    z = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.position.z"].array()

    contrib_begin = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.contributions_begin"].array()
    time_all = events["HcalFarForwardZDCHitsContributions/HcalFarForwardZDCHitsContributions.time"].array()

    # First contribution time per hit
    first_time = time_all[contrib_begin]

    # Event numbers
    event_nums = ak.local_index(energy, axis=0)
    event_nums = ak.broadcast_arrays(energy, event_nums)[1]

    # Combine records
    hits = ak.zip({
        "event": event_nums,
        "energy_GeV": energy,
        "x": x,
        "y": y,
        "z": z - np.min(z),
        "t": first_time
    })

    flat_hits = ak.flatten(hits, axis=0)
    df = ak.to_dataframe(flat_hits).reset_index(drop=True)

    # Build full grid
    unique_events = df[['event']].drop_duplicates()
    unique_xyz = df[['x', 'y', 'z']].drop_duplicates()

    full_grid = unique_events.assign(dummy=1).merge(
        unique_xyz.assign(dummy=1), on='dummy'
    ).drop('dummy', axis=1)

    full_df = pd.merge(
        full_grid,
        df,
        on=['event', 'x', 'y', 'z'],
        how='left'
    )

    full_df['energy_GeV'] = full_df['energy_GeV'].astype(float)
    full_df = full_df.sort_values(by=["event", "z", "y", "x"]).reset_index(drop=True)

    # Assign layer index
    z_to_layer = {z_val: i for i, z_val in enumerate(sorted(full_df['z'].unique()))}
    full_df['layer'] = full_df['z'].map(z_to_layer)

    full_df = full_df.sort_values(by=["event", "layer", "y", "x"]).reset_index(drop=True)
    full_df['layer_ch'] = full_df.groupby(['event', 'layer']).cumcount()

    # ------------------------------------------------------
    # Save output
    # ------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.pkl")

    print(f"Saving: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(full_df, f)

print("\nDone.")
