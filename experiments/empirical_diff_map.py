import torch
from tqdm import tqdm
import climate_learn as cl
import numpy as np

# -------------------------
# 1. Load orography
# -------------------------
oro_path = "dataset/CERRA-534/orography.npz"
oro = np.load(oro_path)["orography"]
oro = np.squeeze(oro).astype(np.float32)
H = torch.tensor(oro)  # shape (534, 534)

# horizontal differences
dH_x = torch.abs(H[:, 1:] - H[:, :-1])
dH_y = torch.abs(H[1:, :] - H[:-1, :])
dH = torch.cat([dH_x.flatten(), dH_y.flatten()])  # 1D vector
max_dH = dH.max()
print(f"Max orography horizontal diff: {max_dH}")

# 2. Setup DataModule (batch_size=1)
dm = cl.data.IterDataModule(
    "direct-forecasting",
    "dataset/CERRA-534/",
    "dataset/CERRA-534/",
    ["2m_temperature"],
    ["2m_temperature"],
    src="era5",
    history=1, # no multiple history
    window=1, # no window trivially
    pred_range=1,
    batch_size=1,  # single sample per batch
    num_workers=4,
)
dm.setup()
train_loader = dm.train_dataloader()

# 3. Prepare bins and accumulators
num_bins = 300
dH_split = torch.linspace(0, max_dH, num_bins + 1)
dH_split_centers = 0.5 * (dH_split[:-1] + dH_split[1:]) 

idx = torch.bucketize(dH, dH_split) - 1
idx = idx.clamp(0, num_bins - 1)  # ensures valid bin index

bin_sums = torch.zeros(num_bins)
bin_sq_sums = torch.zeros(num_bins)
bin_counts = torch.zeros(num_bins)
  


# 4. Loop over training batches
print("Starting empirical statistics computation...")
with torch.no_grad():
    for  batch in tqdm(train_loader):
        T, y, _, _ = batch
        T = T.squeeze().float()  # shape (H, W), batch_size=1

        # compute differences along x and y
        dT_x = (T[:, 1:] - T[:, :-1]).abs() # shape (H, W-1)
        dT_y = (T[1:, :] - T[:-1, :]).abs() # shape (H-1, W)
        dT = torch.cat([dT_x.flatten(), dT_y.flatten()])  # flatten
        
        # accumulate sums, squares, counts
        bin_sums.scatter_add_(0, idx, dT)
        bin_sq_sums.scatter_add_(0, idx, dT * dT)
        bin_counts.scatter_add_(0, idx, torch.ones_like(dT))

# 5. Compute mean and std per bin
mean_per_bin = bin_sums / bin_counts.clamp_min(1)
var_per_bin = (bin_sq_sums / bin_counts.clamp_min(1)) - mean_per_bin**2
std_per_bin = torch.sqrt(var_per_bin.clamp_min(0))

# 6. Save results
torch.save({
    "bin_centers": dH_split_centers,
    "mean": mean_per_bin,
    "std": std_per_bin,
    "counts": bin_counts
}, "outputs/preprocessing/orography_bins.pt")

print("Empirical binned statistics saved to outputs/preprocessing/orography_bins.pt")
