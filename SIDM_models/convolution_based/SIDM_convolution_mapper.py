import torch
import torch.nn as nn
from tqdm import tqdm
import climate_learn as cl
import numpy as np
import os

from climate_learn.models.hub.sidm import dH_to_dT_conv
from climate_learn.metrics.metrics import DISA_abs_horizontal_vertical_differences
# class dH_to_dT_conv(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # local receptive field
#             nn.ReLU(),
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
#         )

#     def forward(self, x):
#         return self.net(x)
    

# class dH_to_dT_conv_PositionalEncodingPretrained(nn.Module):


#     def __init__(self, in_channels=2, out_channels=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # local receptive field
#             nn.ReLU(),
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
#         )

#     def forward(self, x):
#         return self.net(x)
    

# class dH_to_dT_conv_PositionalEncodingJointTrained(nn.Module):


#     def __init__(self, in_channels=2, out_channels=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # local receptive field
#             nn.ReLU(),
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
#         )

#     def forward(self, x):
#         return self.net(x)
    
    



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
print(H.shape)
dH_stack = DISA_abs_horizontal_vertical_differences(H, output="concat",soft=False)
print(dH_stack.shape)

# ---------------------- MODEL ----------------------

model = dH_to_dT_conv().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()


# ---------------------- TRAINING ----------------------

for batch in tqdm(train_loader, total=8640):
    T, y, _, _ = batch
    T = T.squeeze(1,2).float().cuda()  # (H, W)
    # Compute target gradients (NO grad tracking)
    with torch.no_grad():
        dT_stack = DISA_abs_horizontal_vertical_differences(T, output="concat",soft=False)
        print("dT_stack shape:", dT_stack.shape)

    # Forward & Loss
    dT_stack_pred = model(dH_stack)
    loss = loss_fn(dT_stack_pred, dT_stack)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Training complete.")


# ---------------------- SAVE ----------------------

save_path = "outputs/SIDM_models/SIDM_convolution_based/dH_to_dT_conv.pt"
save_dir = os.path.dirname(save_path)

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), save_path)
print("Model saved to", save_path)
