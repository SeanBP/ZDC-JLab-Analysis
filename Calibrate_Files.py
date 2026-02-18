from Analysis_Functions import calibrate_run_folder

import os

base_path = "/media/miguel/Expansion/ZDC_JLab_test_data"
calib_dir = "/home/sean/JLab_Analysis"

run_folders = [
    "63-58-1-Beam",
    "63-58-2-Beam",
    "63-58-3-Beam",
    "63-58-4-Beam",
    "63-58-5-Beam",
    "63-58-6-Beam"
]

run_paths = [os.path.join(base_path, folder) for folder in run_folders]

for run_path in run_paths:
    print(f"\n=== Calibrating folder: {run_path} ===")
    calibrate_run_folder(run_dir=run_path, calib_dir=calib_dir)
