import json
import math
import numpy as np
import pandas as pd
pkl = "/media/miguel/Expansion/ZDC_JLab_test_data/63-58-1-Beam/Run1_calibrated.pkl"
df = pd.read_pickle(pkl)

# Generate the header section
header = {
    "version": "2.2.0",
    "experiment": "ZDC Engineering Test Module",
    "energy_unit": "GeV",
    "color_bar": "Log",
    "length_unit": "cm",
    "time_scale": 10,
    "particles": [
        {
            "angle_rad": [0.0, 0.0],
            "size": float(10),
            "color_rgba": [float(1), float(0), float(0), float(1)]
        }
    ],
    "tracker_settings": {
        "B_field_T": float(0),
        "tracker_boundary": [float(1), float(1), float(1)]
    }
}

num_events_to_include = 100  # adjust as needed

# Group by event
grouped = df.groupby('event')

# Use energy_GeV_full instead of energy_MIP
maxE = df['energy_GeV_full'].max()

events = []

for i, (evt_num, group) in enumerate(grouped):
    if i >= num_events_to_include:
        break

    event = {
        "event_data": {
            "info_text": f"Event #{evt_num}",
            "energy_scale": [0.1, float(maxE)]
        },
        "hits": [],
        "clusters": [],
        "tracks": [],
        "jets": [],
        "blocks": []
    }

    blocks = []

    for _, row in group.iterrows():
        energy = row['energy_GeV_full']

        # skip invalid energy
        if energy <= 0 or math.isnan(energy):
            continue

        x, y, z = row['x'], row['y'], row['z']

        # log-normalized color scaling
        normalized_energy = (
            np.log(energy) - np.log(0.1)
        ) / (
            np.log(maxE) - np.log(0.1)
        )

        normalized_energy = max(0, normalized_energy)

        red   = normalized_energy
        green = 0.0
        blue  = 1.0 - normalized_energy
        alpha = normalized_energy

        # convert z position to time
        c = 299.792  # mm/ns
        time = z / c

        block = {
            "position": [float(x), float(y), float(z)],
            "time_ns": float(time),
            "size": [48.8, 48.8, 4],
            "color_rgba": [
                float(red),
                float(green),
                float(blue),
                float(alpha)
            ]
        }

        blocks.append(block)

    event["blocks"] = blocks
    events.append(event)

# Final output
data = {
    "header": header,
    "events": events
}
eventPath = "/home/sean/JLab_Analysis/JLab_Beam.json"
with open(eventPath, "w") as outfile:
    json.dump(data, outfile, indent=4)
