import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------- LOAD OROGRAPHY -------------------
oro_path = "dataset/CERRA-534/orography.npz"
oro = np.load(oro_path)["orography"]
oro = np.squeeze(oro).astype(np.float32)
H = torch.tensor(oro)  # shape (534, 534)

# ------------------- COMPUTE DIFFERENCES -------------------
# Horizontal differences
dH_x = torch.abs(H[:, 1:] - H[:, :-1])  # shape (534, 533)
dH_y = torch.abs(H[1:, :] - H[:-1, :])  # shape (533, 534)

# ------------------- PLOT -------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
im0 = axs[0].imshow(dH_x.numpy(), cmap="terrain")
axs[0].set_title("Horizontal differences (dH_x)")
axs[0].axis("off")
cbar0 = plt.colorbar(im0, ax=axs[0])
cbar0.set_label("Height difference (m)")

im1 = axs[1].imshow(dH_y.numpy(), cmap="terrain")
axs[1].set_title("Vertical differences (dH_y)")
axs[1].axis("off")
cbar1 = plt.colorbar(im1, ax=axs[1])
cbar1.set_label("Height difference (m)")

plt.suptitle("Orography Differences")
plt.tight_layout()

# ------------------- SAVE FIGURE -------------------
output_dir = "outputs/preprocessing"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "oro_diff_map.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot → {output_path}")
