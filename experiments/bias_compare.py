#!/usr/bin/env python3
"""
biascompare.py

Stepwise comparison of 4 pixelwise absolute bias maps.
Red = worse (bias increased), Green = improvement (bias decreased).
Land mask is overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ---------- Hardcoded paths to bias maps ----------


path_name_dict = {
    "outputs/mean_abs_bias/vit/initial_runs/mean_abs_bias.npz":                             "ViT (Baseline)",
    "outputs/mean_abs_bias/vitginr/vitginr/mean_abs_bias.npz":                              "vitGiNR",      
    "outputs/mean_abs_bias/vitginr/CONV_PE_working/mean_abs_bias.npz":                      "vitGINR(eecR:cnvPE)",
    "outputs/mean_abs_bias/vitginr/vitginr_oroloss_conv_0.1_1/mean_abs_bias.npz":           "vitGINR(eecR:cnv)",
    "outputs/mean_abs_bias/vitginr/vitginr_oroloss_test_9/mean_abs_bias.npz":               "vitGINR(eecR:HIST)",
    "outputs/mean_abs_bias/vitcc/concat_RERUN/mean_abs_bias.npz":                           "ViT (Concat)",
}

name_path_dict = {v: k for k, v in path_name_dict.items()}

bias_files = [
    name_path_dict["ViT (Baseline)"],
    name_path_dict["ViT (Concat)"],
    name_path_dict["vitGiNR"],
    name_path_dict["vitGINR(eecR:cnvPE)"],
    name_path_dict["vitGINR(eecR:HIST)"],
]

# ---------- Hardcoded land mask path ----------
land_mask_path = "/home/janz/dataset/CERRA-534/landsea.npz"  # must contain 2D array 'lsm'

# ---------- Output directory ----------
out_dir = Path("outputs/bias_comparison")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Functions ----------
def load_bias_map(file_path):
    data = np.load(file_path)
    return data["mean_abs_bias"], data["lat"], data["lon"]

def load_land_mask(path):
    data = np.load(path)
    mask = data["lsm"]
    # Ensure 2D
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Land mask must be 2D after squeeze, got {mask.shape}")
    return mask


def make_red_green_cmap():
    """Red = worse, White = no change, Green = improvement"""
    colors = ["#d73027", "#ffffff", "#1a9850"]  # red → white → green
    return LinearSegmentedColormap.from_list("red_green_diff", colors)

def plot_bias_diff(diff_map, lat, lon, land_mask, title, save_path):
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    fig, ax = plt.subplots(figsize=(8,6))

    cmap = make_red_green_cmap()
    vmax = min(5, np.max(np.abs(diff_map)))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    


    im = ax.imshow(diff_map, extent=extent, origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest", aspect='auto')

    # Overlay land mask contour
    cs = ax.contour(
        np.linspace(extent[0], extent[1], land_mask.shape[1]),
        np.linspace(extent[2], extent[3], land_mask.shape[0]),
        land_mask,
        levels=[0.5],
        colors="k",
        linewidths=0.6
    )

     # --- NEW: compute and display average bias difference ---
    avg_bias_diff = np.mean(diff_map)
    ax.set_title(f"{title} \n [Avg bias diff: {avg_bias_diff:.3f}]", fontweight='bold')


    # Add zoom inset if coordinates are provided–––––––––
    zoom_coords=[172, 202, 41, 46]  # x1, x2, y1, y2
    zoom_size=(2.5, 2.5)
    if zoom_coords is not None:
        x1, x2, y1, y2 = zoom_coords
        axins = inset_axes(ax, width=zoom_size[0], height=zoom_size[1], loc='upper right')
        axins.imshow(diff_map, extent=extent, origin="upper",
                     cmap=cmap, norm=norm, interpolation="nearest", aspect='auto')
        axins.contour(
            np.linspace(extent[0], extent[1], land_mask.shape[1]),
            np.linspace(extent[2], extent[3], land_mask.shape[0]),
            land_mask,
            levels=[0.5],
            colors="k",
            linewidths=1
        )

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        # Draw rectangle + lines connecting to inset
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k", lw=0.7)
    # Add zoom inset if coordinates are provided–––––––––



    plt.colorbar(im, ax=ax, pad=0.02, label="Bias difference (prev - current)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")

# ---------- Main stepwise comparison ----------
def main():
    # Load land mask
    land_mask = load_land_mask(land_mask_path)

    # Load all bias maps
    bias_maps = []
    lat, lon = None, None
    for bf in bias_files:
        bmap, lat, lon = load_bias_map(bf)
        bias_maps.append(bmap)
    MAB = [np.mean(bm) for bm in bias_maps]
    for bf, mab in zip(bias_files, MAB):
        print(f"{path_name_dict[bf]}: Mean Abs Bias = {mab:.4f} K")

    # Stepwise comparison
    for i in range(len(bias_maps)-1):
        baseline = bias_maps[i]
        current = bias_maps[i+1]
        diff_map = baseline - current  # positive = improvement

        title = f"|Avg Bias|-improvement:\n{path_name_dict[bias_files[i+1]]} over {path_name_dict[bias_files[i]]}"
        save_path = out_dir / f"bias_diff_{i}_{i+1}.png"

        plot_bias_diff(diff_map, lat, lon, land_mask, title, save_path)

if __name__ == "__main__":
    main()
