# ---------------------- IMPORTS ----------------------
import torch
import torch.nn as nn
from tqdm import tqdm
import climate_learn as cl
import numpy as np
import os

from climate_learn.models.hub.sidm import dH_to_dT_conv
from climate_learn.metrics.metrics import DISA_abs_horizontal_vertical_differences

# ---------------------- DATASET ----------------------
dm = cl.data.IterDataModule(
    "direct-forecasting",
    "dataset/CERRA-534/",
    "dataset/CERRA-534/",
    ["2m_temperature"],
    ["2m_temperature"],
    src="era5",
    history=1,
    window=1,
    pred_range=1,
    batch_size=1,
    num_workers=4,
)
dm.setup()
train_loader = dm.train_dataloader()


# ---------------------- STATIC TERRAIN ----------------------
oro_path = "dataset/CERRA-534/orography.npz"
oro = np.load(oro_path)["orography"].astype(np.float32)
H = torch.tensor(oro) # Ensure shape is (1, 534, 534)
dH_stack = DISA_abs_horizontal_vertical_differences(H, output="concat",soft=False) # (B,2, H, W)

# ---------------------- MODEL ----------------------
model = dH_to_dT_conv().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()


# ---------------------- TRAINING ----------------------
for batch in tqdm(train_loader, total=8640):
    T, y, _, _ = batch
    T = T.squeeze(1,2).float().cuda()  # (H, W)
    
    with torch.no_grad(): # Compute target dT (NO grad tracking)
        dT_stack = DISA_abs_horizontal_vertical_differences(T, output="concat",soft=False) # (B,2, H, W)

    # Forward & Loss
    dT_stack_pred = model(dH_stack)
    loss = loss_fn(dT_stack_pred, dT_stack)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Training complete.")


# ---------------------- SAVE --------------------
save_path = "outputs/SIDM_models/SIDM_convolution_based/dH_to_dT_conv.pt"
save_dir = os.path.dirname(save_path)

os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
torch.save(model.state_dict(), save_path)
print("Model saved to", save_path)
