"""Basic usage example for the edrose package."""

import numpy as np

from edrose import EllipticalWindRose, MixtureEllipticalWindRose

# --- 1. Create a wind rose from parameters ---
wr = EllipticalWindRose(a=0.8, f=0.3, theta_prev=210, n_sectors=12)
print(wr)
print("Directions:", wr.wind_directions)
print("Frequencies:", np.round(wr.sector_frequencies, 4))
print("Sum:", wr.sector_frequencies.sum())
print()

# --- 2. Fit to measured data ---
# Synthetic example: generate data from a known model, then fit
true_wr = EllipticalWindRose(a=1.0, f=0.4, theta_prev=180, n_sectors=12)
measured = true_wr.sector_frequencies

fit_wr = EllipticalWindRose.fit(measured)
print("True:   ", true_wr)
print("Fitted: ", fit_wr)
print(f"R² = {fit_wr.r_squared:.6f},  RMSE = {fit_wr.rmse:.6f}")
print()

# --- 3. PyWake-compatible arrays ---
arrays = wr.pywake_arrays(A=9.0, k=2.0)
print("PyWake arrays:")
for key, val in arrays.items():
    print(f"  {key}: {np.round(val, 4) if isinstance(val, np.ndarray) else val}")
print()

# --- 4. Mixture model ---
c1 = EllipticalWindRose(a=0.8, f=0.1, theta_prev=180, n_sectors=36)
c2 = EllipticalWindRose(a=1.1, f=1.0, theta_prev=260.5, n_sectors=36)
mix = MixtureEllipticalWindRose([c1, c2], weights=[0.85, 0.15])
print("Mixture:", mix)
print("Sum:", mix.sector_frequencies.sum())
