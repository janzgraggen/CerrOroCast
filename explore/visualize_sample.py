import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe backend for scripts
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------
# Dataset root
# ------------------------------
ROOT = Path("../dataset/CERRA-534")
train_files = sorted((ROOT / "train").glob("*.npz"))
sample_file = train_files[0]

# ------------------------------
# Load sample NPZ
# ------------------------------
z = np.load(sample_file, allow_pickle=True)
lat = np.load(ROOT / "lat.npy")
lon = np.load(ROOT / "lon.npy")

# Pick first array with 2D+ data
for k in z.files:
    a = z[k]
    if a.ndim >= 2:
        arr = a
        key = k
        break

# Reduce to 2D: first timestep/channel
arr2 = np.squeeze(arr[0, ...]).astype(np.float32)
print("arr2.shape:", arr2.shape)

# ------------------------------
# Flip latitude if decreasing
# ------------------------------
# Works for both 1D or 2D lat
if lat.ndim == 1:
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arr2 = arr2[::-1, :]
elif lat.ndim == 2:
    if lat[0,0] > lat[-1,0]:
        lat = lat[::-1, :]
        arr2 = arr2[::-1, :]

# ------------------------------
# Plot using original 2D lat/lon grids
# ------------------------------
plt.figure(figsize=(10,5))
plt.pcolormesh(lon, lat, arr2, shading='auto')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"{sample_file.name} : key={key}")
plt.colorbar(label="value")
plt.tight_layout()

# Save to file
plt.savefig("sample_plot.png", dpi=150)
plt.close()
print("Plot saved as sample_plot.png")
