import numpy as np
import matplotlib.pyplot as plt

path = "/home/janz/dataset/CERRA-534/orography.npz"

# Load npz
data = np.load(path)
print("Available keys in .npz:", data.files)

key = data.files[0]  # take first key
oro = data[key]

print(f"Using key: {key}")
print("Orography raw shape:", oro.shape)

# Remove singleton dimension if present
oro2d = np.squeeze(oro)  # shape -> (534, 534)

print("Orography squeezed shape:", oro2d.shape)
print("dtype:", oro2d.dtype)
print("Min:", oro2d.min(), "Max:", oro2d.max(), "Mean:", oro2d.mean())

# Plot
plt.imshow(oro2d, cmap="terrain")
plt.colorbar(label="Elevation (m)")
plt.title("Orography Map")

out_path = "orography_preview.png"
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")