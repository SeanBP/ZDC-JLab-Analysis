import uproot as ur
import awkward as ak
import pandas as pd
import itertools
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load file
input_file = "/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e5.3GeV_full_1.edm4hep.root"
events = ur.open(f"{input_file}:events")
print("Opened file")
# Load jagged arrays from branches
energy = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.energy"].array()
x = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.position.x"].array()
y = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.position.y"].array()
z = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.position.z"].array()

contrib_begin = events["HcalFarForwardZDCHits/HcalFarForwardZDCHits.contributions_begin"].array()
time_all = events["HcalFarForwardZDCHitsContributions/HcalFarForwardZDCHitsContributions.time"].array()
print("Loaded branches")
# Use the begin indices to select the first time for each hit
first_time = time_all[contrib_begin]

# This gives one event number per event
event_nums = ak.local_index(energy, axis=0)

# Broadcast to match the hits per event
event_nums = ak.broadcast_arrays(energy, event_nums)[1]


# Combine into a jagged array of records
hits = ak.zip({
    "event": event_nums,
    "energy_GeV": energy,
    "x": x,
    "y": y,
    "z": z - np.min(z),
    "t": first_time
})

# Flatten to get one row per hit
flat_hits = ak.flatten(hits, axis=0)

# Convert to pandas DataFrame
df = ak.to_dataframe(flat_hits).reset_index(drop=True)

unique_events = df[['event']].drop_duplicates()
unique_xyz = df[['x', 'y', 'z']].drop_duplicates()


# Cartesian product
full_grid = unique_events.assign(dummy=1).merge(
    unique_xyz.assign(dummy=1), on='dummy').drop('dummy', axis=1)

# Merge with original
full_df = pd.merge(
    full_grid,
    df,
    on=['event', 'x', 'y', 'z'],
    how='left'
)

# Optional: convert energy to float (if needed)
full_df['energy_GeV'] = full_df['energy_GeV'].astype(float)
full_df = full_df.sort_values(by=["event", "z", "y", "x"]).reset_index(drop=True)


# Step 1: Assign "layer" by ranking unique z values
z_to_layer = {z: i for i, z in enumerate(sorted(full_df['z'].unique()))}
full_df['layer'] = full_df['z'].map(z_to_layer)

# Step 2: Sort again to ensure correct order before assigning layer_ch
full_df = full_df.sort_values(by=["event", "layer", "y", "x"]).reset_index(drop=True)

# Step 3: Assign layer_ch within each event/layer group
full_df['layer_ch'] = full_df.groupby(['event', 'layer']).cumcount()
print("Saving dataframe")
with open('/media/miguel/Expansion/ZDC_JLab_test_data/Sim/e+5.3GeV_DF_1.pkl', 'wb') as f:
    pickle.dump(full_df, f)
