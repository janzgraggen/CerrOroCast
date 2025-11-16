import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import os

# ------------------- LOAD BINS -------------------
data = torch.load("outputs/preprocessing/orography_bins.pt")
bin_centers = data["bin_centers"]
mean_per_bin = data["mean"]
std_per_bin = data["std"]
counts = data["counts"]

# Remove empty bins
mask_nz = counts > 0
bin_centers_nz = bin_centers[mask_nz]
mean_nz = mean_per_bin[mask_nz]
std_nz = std_per_bin[mask_nz]
counts_nz = counts[mask_nz]

# Scatter point settings
alpha = (counts_nz / counts_nz.max()).clamp(0.1,1.0)
colors = 1 - (std_nz / std_nz.max())
colors = np.clip(colors.numpy(),0,1)

# ------------------- BAR SETTINGS -------------------
bar_height_max = 0.5
bar_cmap = plt.cm.Blues

# Root scaling for count bars
count_root = np.sqrt(counts_nz.numpy())
count_root_norm = count_root / count_root.max()
count_root_norm = np.clip(count_root_norm, 0, 0.95)
bar_heights = count_root_norm * bar_height_max

# Compute bin widths
bin_widths = np.diff(np.concatenate([[bin_centers_nz[0] - (bin_centers_nz[1]-bin_centers_nz[0])/2],
                                     0.5*(bin_centers_nz[1:] + bin_centers_nz[:-1]),
                                     [bin_centers_nz[-1] + (bin_centers_nz[-1]-bin_centers_nz[-2])/2]]))

# ------------------- FUNCTION TO PLOT -------------------
def plot_scatter_with_bars(bin_centers, mean, std, counts, title, filename,xlim=0):
    # Remove empty bins
    mask = counts > 0
    bin_centers_nz = bin_centers[mask]
    mean_nz = mean[mask]
    std_nz = std[mask]
    counts_nz = counts[mask]

    alpha = (counts_nz / counts_nz.max()).clamp(0.1,1.0)
    colors = 1 - (std_nz / std_nz.max())
    colors = np.clip(colors.numpy(),0,1)

    # Root-scaled bars
    count_root = np.sqrt(counts_nz.numpy())
    count_root_norm = count_root / count_root.max()
    count_root_norm = np.clip(count_root_norm, 0, 0.95)
    bar_heights = count_root_norm * bar_height_max

    # Bin widths
    bin_widths = np.diff(np.concatenate([[bin_centers_nz[0] - (bin_centers_nz[1]-bin_centers_nz[0])/2],
                                         0.5*(bin_centers_nz[1:] + bin_centers_nz[:-1]),
                                         [bin_centers_nz[-1] + (bin_centers_nz[-1]-bin_centers_nz[-2])/2]]))

    # Figure
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(bin_centers_nz.numpy(), mean_nz.numpy(),
                    s=30, c=colors, cmap='viridis_r', alpha=alpha.numpy(), edgecolors='k')
    ax.errorbar(bin_centers_nz.numpy(), mean_nz.numpy(),
                yerr=std_nz.numpy(), fmt='none', ecolor='gray', alpha=0.4, capsize=2)
    ax.set_xlabel("|ΔH| (m)")
    ax.set_ylabel("Mean |ΔT| (K)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim(left=xlim)

    # Bars below y=0
    for x,h,w in zip(bin_centers_nz.numpy(), bar_heights, bin_widths):
        ax.add_patch(Rectangle((x-w/2, -h), w, h, color=bar_cmap(h/bar_height_max)))
    ax.set_ylim(bottom=-bar_height_max, top=ax.get_ylim()[1])

    # Vertical colorbar for std (right side)
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label("Normalized std (uncertainty)")

    # Horizontal colorbar for count bars
    cbar_ax = fig.add_axes([0.12, -0.02, 0.7, 0.02])  # lower position
    norm = Normalize(vmin=counts_nz.min(), vmax=counts_nz.max())
    sm = cm.ScalarMappable(cmap=bar_cmap, norm=norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar2.set_label("Bin occurrence (bar height ∝ sqrt(counts))")

    # Save
    os.makedirs("outputs/preprocessing", exist_ok=True)
    plt.savefig(f"outputs/preprocessing/{filename}", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------- FULL PLOT -------------------
plot_scatter_with_bars(
    bin_centers, mean_per_bin, std_per_bin, counts,
    "Empirical |ΔT| vs |ΔH| with root-scaled count bars",
    "empirical_diff_map_full_root_bars_std.png"
)

# ------------------- HIGH ΔH PLOT (≥700) -------------------
high_threshold = 700
mask_high = bin_centers >= high_threshold
plot_scatter_with_bars(
    bin_centers[mask_high], mean_per_bin[mask_high], std_per_bin[mask_high], counts[mask_high],
    f"Empirical |ΔT| vs |ΔH| (high ΔH ≥ {high_threshold})",
    "empirical_diff_map_high_root_bars_std.png",
    xlim=high_threshold
)

print("Saved full and high ΔH scatter plots with root-scaled count bars and std colorbars.")
