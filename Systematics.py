#!/usr/bin/env python3
import numpy as np

# ----------------------------
# Inputs
# ----------------------------
x = 15.5          # inches
y = 172.74        # inches

sigma_x_geom = 0.5     # inches
sigma_y_geom = 1.33    # inches

sigma_x_tile_mm = 3.06 # mm (10.6/sqrt(12))
sigma_x_tile = sigma_x_tile_mm / 25.4  # convert mm -> inches

beam_energy = 5.3      # GeV
dE_dtheta = 0.64       # GeV per degree
ps_resolution = 0.025  # GeV (PS hodoscope resolution)

# ----------------------------
# Functions
# ----------------------------
def theta_deg(x, y):
    return np.degrees(np.arctan(x / y))

def propagate_theta_uncertainty(x, y, sigma_x, sigma_y):
    denom = (x**2 + y**2)
    dtheta_dx = y / denom
    dtheta_dy = -x / denom
    sigma_theta_rad = np.sqrt(
        (dtheta_dx * sigma_x)**2 +
        (dtheta_dy * sigma_y)**2
    )
    return np.degrees(sigma_theta_rad)

def energy_from_angle_uncertainty(sigma_theta_deg):
    sigma_E = dE_dtheta * sigma_theta_deg
    percent = 100.0 * sigma_E / beam_energy
    return sigma_E, percent

# ----------------------------
# Nominal angle
# ----------------------------
theta0 = theta_deg(x, y)

# ----------------------------
# Geometric placement
# ----------------------------
sigma_theta_geom = propagate_theta_uncertainty(
    x, y, sigma_x_geom, sigma_y_geom
)
sigma_E_geom, percent_geom = energy_from_angle_uncertainty(
    sigma_theta_geom
)

E_low_geom  = beam_energy - sigma_E_geom
E_high_geom = beam_energy + sigma_E_geom

mult_low_geom  = E_low_geom  / beam_energy
mult_high_geom = E_high_geom / beam_energy

# ----------------------------
# Trigger tile contribution
# ----------------------------
sigma_theta_tile = propagate_theta_uncertainty(
    x, y, sigma_x_tile, 0.0
)
sigma_E_tile, percent_tile = energy_from_angle_uncertainty(
    sigma_theta_tile
)

# ----------------------------
# PS hodoscope (standalone)
# ----------------------------
percent_ps = 100.0 * ps_resolution / beam_energy

# ----------------------------
# Trigger + PS combined
# ----------------------------
sigma_E_combined = np.sqrt(
    sigma_E_tile**2 +
    ps_resolution**2
)
percent_combined = 100.0 * sigma_E_combined / beam_energy

# ----------------------------
# Output
# ----------------------------
print(f"Nominal angle: {theta0:.3f} deg\n")

print("Geometric placement uncertainty:")
print(f"  sigma_theta = {sigma_theta_geom:.3f} deg")
print(f"  sigma_E     = {sigma_E_geom:.3f} GeV")
print(f"  percent     = {percent_geom:.2f} %")
print(f"  Energy range: {E_low_geom:.3f} – {E_high_geom:.3f} GeV")
print(f"  Scale multipliers:")
print(f"     lower = {mult_low_geom:.4f}")
print(f"     upper = {mult_high_geom:.4f}\n")

print("Trigger tile angular contribution:")
print(f"  sigma_theta = {sigma_theta_tile:.3f} deg")
print(f"  sigma_E     = {sigma_E_tile:.3f} GeV")
print(f"  percent     = {percent_tile:.2f} %\n")

print("PS hodoscope energy resolution (standalone):")
print(f"  sigma_E     = {ps_resolution:.3f} GeV")
print(f"  percent     = {percent_ps:.2f} %\n")

print("Trigger + PS hodoscope (quadrature):")
print(f"  sigma_E     = {sigma_E_combined:.3f} GeV")
print(f"  percent     = {percent_combined:.2f} %")
