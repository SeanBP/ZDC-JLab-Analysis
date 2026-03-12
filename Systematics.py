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
print(f"  percent     = {percent_combined:.2f} %\n")

#!/usr/bin/env python3
import numpy as np

# ============================================================
# Generic total uncertainty calculator
# ============================================================

def total_uncertainty(central, central_err, variations):
    """
    central: central value (standard)
    central_err: statistical error of standard
    variations: list of (value, fit_error)
    """

    pos_terms = []
    neg_terms = []

    for val, err in variations:

        # shift relative to standard
        delta = val - central

        # propagate error on the shift
        delta_err = np.sqrt(err**2 + central_err**2)

        if delta > 0:
            pos_terms.append((delta, delta_err))
        elif delta < 0:
            neg_terms.append((delta, delta_err))

    def combine(terms):
        if not terms:
            return 0.0

        # quadratic sum of shifts
        S = np.sqrt(np.sum([d**2 for d, _ in terms]))

        return S

    S_pos = combine(pos_terms)
    S_neg = combine(neg_terms)

    # total = sqrt(stat^2 + syst^2)
    tot_pos = np.sqrt(central_err**2 + S_pos**2)
    tot_neg = np.sqrt(central_err**2 + S_neg**2)

    return tot_pos, tot_neg


# ============================================================
# Energy Resolution
# ============================================================

E_std, E_std_err = 11.199, 0.019

E_vars = [
    (11.199, 0.02),  # Low
    (11.2, 0.02),  # High
    (11.102, 0.02),  # Avg
]

E_pos, E_neg = total_uncertainty(E_std, E_std_err, E_vars)


# ============================================================
# Position Resolution
# ============================================================

# ---- X ----
X_std, X_std_err = 6.095, 0.017

X_vars = [
    (6.095, 0.017),  # Low
    (6.095, 0.017),  # High
    (6.59, 0.023),  # Avg
]

X_pos, X_neg = total_uncertainty(X_std, X_std_err, X_vars)

# ---- Y ----
Y_std, Y_std_err = 5.352, 0.022

Y_vars = [
    (5.352, 0.022),  # Low
    (5.352, 0.022),  # High
    (5.238, 0.022),  # Avg
]

Y_pos, Y_neg = total_uncertainty(Y_std, Y_std_err, Y_vars)

def percent_diff(var, ref):
    return 100.0 * (var - ref) / ref

# ============================================================
# Print Results
# ============================================================

print("Energy resolution:")
print(f"{E_std:.3f} %")
print(f"  +{E_pos:.3f} %")
print(f"  -{E_neg:.3f} %\n")

print("  Percent differences:")
for name, val, _ in E_vars:
    print(f"    {name}: {percent_diff(val, E_std):+.3f} %")
print()

print("X position resolution:")
print(f"{X_std:.3f} mm")
print(f"  +{X_pos:.3f} mm")
print(f"  -{X_neg:.3f} mm\n")

print("  Percent differences:")
for name, val, _ in X_vars:
    print(f"    {name}: {percent_diff(val, X_std):+.3f} %")
print()

print("Y position resolution:")
print(f"{Y_std:.3f} mm")
print(f"  +{Y_pos:.3f} mm")
print(f"  -{Y_neg:.3f} mm")

print("  Percent differences:")
for name, val, _ in Y_vars:
    print(f"    {name}: {percent_diff(val, Y_std):+.3f} %")
