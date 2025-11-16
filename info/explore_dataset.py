
## Imports Libraries
from pathlib import Path
import numpy as np


## Set the root path to the dataset
ROOT = Path("dataset/CERRA-534/")

for split in ("train", "val", "test"):
    files = sorted((ROOT / split).glob("*.npz"))
    print(f"{split}: {len(files)} files; first 5: {[f.name for f in files[:5]]}")

print("\nRoot files:", [f.name for f in ROOT.iterdir() if f.is_file()])

lat = np.load(ROOT / "lat.npy")
lon = np.load(ROOT / "lon.npy")
print("lat.shape:", lat.shape, "lon.shape:", lon.shape,
      "lat min/max:", lat.min(), lat.max(), "lon min/max:", lon.min(), lon.max())

for nf in ("normalize_mean.npz", "normalize_std.npz", "orthography.npz"):
    p = ROOT / nf
    if p.exists():
        z = np.load(p, allow_pickle=True)
        print(nf, "-> keys:", z.files)

sample_file = sorted((ROOT / "train").glob("*.npz"))[0]
print("\nInspecting", sample_file.name)
z = np.load(sample_file, allow_pickle=True)
print("Keys:", z.files)
for k in z.files:
    a = z[k]
    print(k, "shape:", a.shape, "dtype:", a.dtype,
          "min/max:", float(a.min()), float(a.max()))

# Check the shape of arrays in all files (use train/val/test as needed)
# split = "train"  # or "val", "test"
# shapes = {}

# for f in sorted((ROOT / split).glob("*.npz")):
#     z = np.load(f, allow_pickle=True)
#     for k in z.files:
#         a = z[k]
#         shapes.setdefault(k, set()).add(a.shape)
#     z.close()

# for k, s in shapes.items():
#     print(f"{k}: found {len(s)} unique shapes → {s}")