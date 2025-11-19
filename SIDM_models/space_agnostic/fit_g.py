import matplotlib
matplotlib.use("Agg")  # non-interactive backend for workstation

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------- LOAD DATA -------------------
data = torch.load("outputs/preprocessing/orography_bins.pt")
bin_centers = data["bin_centers"].numpy()
mean_per_bin = data["mean"].numpy()
std_per_bin = data["std"].numpy()
counts = data["counts"].numpy()

# ------------------- FILTER BINS: only bins with data -------------------
mask = counts > 0
x = bin_centers[mask]
y = mean_per_bin[mask]

# ------------------- ADDITIONAL FILTER: |ΔH| < 700 -------------------
mask = (130 < np.abs(x)) & (np.abs(x) < 700)
x = x[mask]
y = y[mask]

# ------------------- POLYNOMIAL FIT -------------------
deg = 1
coeffs = np.polyfit(x, y, deg=deg)
poly = np.poly1d(coeffs)

print("\nFitted cubic polynomial:")
print(poly)

# Smooth x-range for plotting
xs = np.linspace(x.min(), x.max(), 1200)
ys = poly(xs)

# ------------------- PLOT -------------------
plt.figure(figsize=(11, 7))

# Scatter of empirical means
plt.scatter(x, y, s=12, alpha=0.6, label="Empirical mean |ΔT|", edgecolor="k")

# Polynomial curve
plt.plot(xs, ys, "r-", linewidth=2, label=f"Cubic fit (deg={deg})")

plt.xlabel("|ΔH| (m)")
plt.ylabel("Mean |ΔT| (K)")
plt.title("Degree-3 Polynomial Fit to Empirical |ΔT| vs |ΔH|")
plt.grid(True)
plt.legend()

# ------------------- SAVE FIGURE -------------------
output_dir = "outputs/preprocessing"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "polyfit_curve.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved polynomial fit plot → {output_path}\n")
