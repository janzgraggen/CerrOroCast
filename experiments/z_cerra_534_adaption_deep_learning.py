
# IMPORTS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Standard library
TOP_LVL_DEBUG = True

from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

# END IMPORTS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# PARSER ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
if TOP_LVL_DEBUG: 
    print("[line:28] Start parsing")
parser = ArgumentParser()

parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--logname", type=str, default=None)


subparsers = parser.add_subparsers(
    help="Whether to perform direct, iterative, or continuous forecasting.",
    dest="forecast_type",
    required=True,
)
direct = subparsers.add_parser("direct")
iterative = subparsers.add_parser("iterative")
continuous = subparsers.add_parser("continuous")

direct.add_argument("cerra534_dir")
direct.add_argument("model", choices=["resnet", "unet", "vit","vit","vitcc", "geofar","geofar_v2"])
direct.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

iterative.add_argument("cerra534_dir")
iterative.add_argument("model", choices=["resnet", "uneta", "vit","vitcc", "geofar","geofar_v2"])
iterative.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

continuous.add_argument("cerra534_dir")
continuous.add_argument("model", choices=["resnet", "unet", "vit","vitcc", "geofar","geofar_v2"])

args = parser.parse_args()

if TOP_LVL_DEBUG: 
    print("[line:63] parsing complete")
    print("[line:64]: Start Variable assignment" )
# END PARSER ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# VARIABLES ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# Set up data
variables = [
    "2m_temperature"
    #"lattitude",
]
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)
if args.forecast_type in ("direct", "continuous"):
    out_variables = ["2m_temperature"] #, "geopotential_500", "temperature_850"]
elif args.forecast_type == "iterative":
    out_variables = variables
out_vars = []
for var in out_variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            out_vars.append(var + "_" + str(level))
    else:
        out_vars.append(var)


if TOP_LVL_DEBUG: 
    print("[line:94] var assignment complete")
    print("[line:95]: Start Data module loading" )
# END VARIABLES ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DATA MODULE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
if args.forecast_type in ("direct", "iterative"):
    dm = cl.data.IterDataModule(
        f"{args.forecast_type}-forecasting",
        args.cerra534_dir,
        args.cerra534_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=args.bs, ## reduce for memory (-> test) #32 or 128
        num_workers=4, #16
    )

elif args.forecast_type == "continuous":
    dm = cl.data.IterDataModule(
        "continuous-forecasting",
        args.cerra534_dir,
        args.cerra534_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=1,
        max_pred_range=120,
        random_lead_time=True,
        hrs_each_step=1,
        subsample=6,
        batch_size=128,
        buffer_size=2000,
        num_workers=8, 
    )
dm.setup()


if TOP_LVL_DEBUG: 
    print("[line:137] Data Loader module loading complete")
    print("[line:138]: Start Model setup")
# END DATA MODULE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# LEARNING MODEL ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Set up deep learning model
in_channels = 1 
if args.forecast_type == "continuous":
    in_channels += 1  # time dimension
if args.forecast_type == "iterative":  # iterative predicts every var
    out_channels = in_channels
else:
    out_channels = 1
if args.model == "resnet":
    model_kwargs = {  # override some of the defaults
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "n_blocks": 5, #28
    }
elif args.model == "unet":
    model_kwargs = {  # override some of the defaults
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    model_kwargs = {  # override some of the defaults
        "img_size": (534, 534),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3, 
        "patch_size": 6, #2
        "embed_dim": 64, #128
        "depth": 4, # 8
        "decoder_depth": 2, #2
        "learn_pos_emb": True,
        "num_heads": 4,
    }

elif args.model == "vitcc":
    model_kwargs = {  # override some of the defaults
        "img_size": (534, 534),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 6, #2
        "embed_dim": 64, #128
        "depth": 4, # 8
        "decoder_depth": 2, #2
        "learn_pos_emb": True,
        "num_heads": 4,
        ### Aditional Params for VisionTransformerCC
        "oro_path": f"{args.cerra534_dir}/orography.npz" ## <- !!!!!
    }


elif args.model == "geofar":
    model_kwargs = {  # override some of the defaults
        "img_size": (534, 534),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 6, #2
        "embed_dim": 64, #128
        "depth": 4, # 8
        "decoder_depth": 2, #2
        "learn_pos_emb": True,
        "num_heads": 4,
        ### Aditional Params for GeoFAR
        "oro_path": f"{args.cerra534_dir}/orography.npz", ## <- !!!!!
        "n_coeff": 16, ## <- !!!!! 64
        "n_sh_coeff": 16, ## <- !!!!! 64
        "conv_start_size": 16, ## <- !!!!!
        "siren_hidden": 32, ## <- !!!!! 
    }

elif args.model == "geofar_v2":
    model_kwargs = {  # override some of the defaults
        "img_size": (534, 534),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 6, #2
        "embed_dim": 64, #128
        "depth": 4, # 8
        "decoder_depth": 2, #2
        "learn_pos_emb": True,
        "num_heads": 4,
        ### Aditional Params for GeoFAR
        "oro_path": f"{args.cerra534_dir}/orography.npz", ## <- !!!!!
        "n_coeff": 64, ## <- !!!!! 64
        "n_sh_coeff": 64, ## <- !!!!! 64
        "conv_start_size": 64, ## <- !!!!!
        "siren_hidden": 128, ## <- !!!!! 
    }


optim_kwargs = {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
sched_kwargs = {
    "warmup_epochs": 5,
    "max_epochs": 20,
    "warmup_start_lr": 1e-8,
    "eta_min": 1e-8,
}
model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
    train_loss="mse",
    val_loss=["rmse"],
    test_loss=["rmse"],
    train_target_transform=None,
    val_target_transform=["denormalize"],
    test_target_transform=["denormalize"],
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.model}_{args.forecast_type}_forecasting_{args.pred_range}"
if args.logname == None:
    args.logname = default_root_dir
#tb_logger = TensorBoardLogger(save_dir=f"{args.logname}/logs")
wandb_logger = WandbLogger(project="cerra_534", name=args.logname, save_dir=f"logs/{args.logname}")

loggers = [wandb_logger] #, tb_logger,

early_stopping = "val/rmse:aggregate" ## available: `train/mse:aggregate`, `val/rmse:2m_temperature`, `val/rmse:aggregate`
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, patience=args.patience),
    ModelCheckpoint(
        dirpath=f"checkpoints/{default_root_dir}",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
]
trainer = pl.Trainer(
    logger=loggers,
    callbacks=callbacks,
    default_root_dir=default_root_dir, 
    max_epochs=args.max_epochs,
    accelerator="gpu" if args.gpu != -1 else None,
    devices=[args.gpu] if args.gpu != -1 else None,
    strategy="ddp",
    precision="32",
)
if TOP_LVL_DEBUG:
    print("[line:292]: Model and Trainer Setup complete" )
    print("[line:293]: Start Training" )
###––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– Iterative
### alternative to trainer.test(model, datamodule=dm) for iter


# Define testing regime for iterative forecasting
def iterative_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        n_iters = lead_time // args.pred_range
        model.set_mode("iter")
        model.set_n_iters(n_iters)
        test_dm = cl.data.IterDataModule(
            "iterative-forecasting",
            args.cerra534_dir,
            args.cerra534_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            subsample=1,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")

###–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– Continous
### alternative to trainer.test(model, datamodule=dm) for cont


# Define testing regime for continuous forecasting
def continuous_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        test_dm = cl.data.IterDataModule(
            "continuous-forecasting",
            args.cerra534_dir,
            args.cerra534_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            max_pred_range=lead_time,
            random_lead_time=False,
            hrs_each_step=1,
            subsample=1,
            batch_size=128,
            buffer_size=2000,
            num_workers=8,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


# Train and evaluate model from scratch –––––––––––––-––––-–––––––––

if args.checkpoint is None:
    ### TRAINING MODEL FROM SCRATCH
    trainer.fit(model, datamodule=dm) 

    ### EVALUATING MODEL
    if args.forecast_type == "direct":
        if TOP_LVL_DEBUG:
            print("[line:361]: \{Training\} enter direct training mode and call: trainer.test() in if args.checkpoint is none [A]")
        trainer.test(model, datamodule=dm, ckpt_path="best")
        if TOP_LVL_DEBUG:
            print("[line:364]: \{Training\} enter direct training mode and call: trainer.test() [A]")
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args)

# Evaluate saved model checkpoint ––––-––––––––––––––––––––––––––––––
else:
    model = cl.LitModule.load_from_checkpoint(
        args.checkpoint,
        net=model.net,
        optimizer=model.optimizer,
        lr_scheduler=None,
        train_loss=None,
        val_loss=None,
        test_loss=model.test_loss,
        test_target_tranfsorms=model.test_target_transforms,
    )
    if args.forecast_type == "direct":
        if TOP_LVL_DEBUG:
            print("[line:384]: \{Training\} enter direct training mode and call: trainer.test() in FROM checkpint [B]")
        trainer.test(model, datamodule=dm)
        if TOP_LVL_DEBUG:
            print("[line:387]: \{Training\} enter direct training mode and call: trainer.test() [B]")
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args, from_checkpoint=True)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args, from_checkpoint=True)

'''
class Geo_INR_v2(nn.Module):
    def __init__(
            self,
            n_sh_coeff, 
            basis, 
            oro_path=None, 
            in_channels=1,
            conv_start_size=64,
            siren_hidden=128
        ):  # [n, H, W]
        super().__init__()
        eps = 1e-6
        if isinstance(oro_path, str):
            oro = np.load(oro_path)['orography']
            oro = (oro - oro.mean()) / (oro.std() + eps)
            oro = torch.tensor(oro, dtype=torch.float32)
        else:  
            oro = oro_path.to(torch.float32)
            std = oro.std(unbiased=False).clamp_min(eps)
            oro = (oro - oro.mean()) / std
        self.register_buffer("oro", oro.unsqueeze(0))  # [1, H, W]
        self.register_buffer("basis", basis.unsqueeze(0))  # 
        self.oro_encoder = nn.Sequential(
            PeriodicConv2D(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            PeriodicConv2D(32, n_sh_coeff, kernel_size=3, padding=1),
        )
        self.siren = SirenNet(dim_in=n_sh_coeff, dim_hidden=siren_hidden, num_layers=2, dim_out=n_sh_coeff)
        self.projection = PeriodicConv2D(n_sh_coeff*in_channels, conv_start_size, kernel_size=1)
        self.conv = nn.Sequential(
            PeriodicConv2D(conv_start_size, 2*conv_start_size, kernel_size=3, padding=1),
            nn.SiLU(),   # new, remove two layers to reduce computation
            PeriodicConv2D(2*conv_start_size, n_sh_coeff, kernel_size=1, padding=0), # # new, remove (two layers, and the *) to reduce computation
        )
        self.n_sh_coeff=n_sh_coeff
        self.in_channels = in_channels

    def forward(self, A):  # [B, n, H, W]
        print(f"A shape in Geo_INR_v2: {A.shape}") # torch.Size([16, 192, 534, 534])
        B, _, H, W = A.size() 
        C = self.in_channels
        loc_basis = self.basis.view(-1, self.n_sh_coeff, H, W) # [1, 64, H, W]
        # Orography
        oro = self.oro.view(-1, H, W)  # [B, H, W]
        oro = oro.unsqueeze(1)  # [B, 1, H, W]
        # Compute gradients 
        dx = oro[:, :, :, 1:] - oro[:, :, :, :-1]  # [B, 1, H, W-1]
        last_col = dx[:, :, :, -1:].clone()        # replicate last column
        grad_x = torch.cat([dx, last_col], dim=-1)  # [B, 1, H, W
        dy = oro[:, :, 1:, :] - oro[:, :, :-1, :]  # [B, 1, H-1, W]
        last_row = dy[:, :, -1:, :].clone()
        grad_y = torch.cat([dy, last_row], dim=-2)  # [B, 1, H, W]
        oro_feat = torch.cat([oro, grad_x, grad_y], dim=1)  # [B, 3, H, W]
        # Encode orography features
        oro_basis = self.oro_encoder(oro_feat)  # [B, n_sh, H, W]
        geo_basis = loc_basis + oro_basis # [1, n_sh, H, W]
        geo_basis = self.siren(geo_basis.permute(0,2,3,1))#.view(self.n_sh_coeff, H, W)
        geo_basis = geo_basis.permute(0, 3, 1, 2) # [B, n_sh, H, W]
        multi_geo_basis = geo_basis.repeat(1, C, 1, 1)
        
        fused = multi_geo_basis * A + multi_geo_basis #A: torch.Size([16, 192, 534, 534])
        fused = self.projection(fused)
        out = self.conv(fused)
        
        return out  # [B, 1, H, W]
class GeoNoFAR_INR(nn.Module):
    def __init__(
            self,
            n_sh_coeff,
            basis,
            oro_path=None,
            in_channels=1,
            conv_start_size=64,
            siren_hidden=128
        ):  # [n, H, W]
        super().__init__()
        eps = 1e-6
        if isinstance(oro_path, str):
            oro = np.load(oro_path)['orography']
            oro = (oro - oro.mean()) / (oro.std() + eps)
            oro = torch.tensor(oro, dtype=torch.float32)
        else:
            oro = oro_path.to(torch.float32)
            std = oro.std(unbiased=False).clamp_min(eps)
            oro = (oro - oro.mean()) / std
        self.register_buffer("oro", oro.unsqueeze(0))  # [1, H, W]
        self.register_buffer("basis", basis.unsqueeze(0))  #
        self.oro_encoder = nn.Sequential(
            PeriodicConv2D(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            PeriodicConv2D(32, n_sh_coeff, kernel_size=3, padding=1),
        )
        self.siren = SirenNet(dim_in=n_sh_coeff, dim_hidden=siren_hidden, num_layers=2, dim_out=n_sh_coeff)
        self.conv = nn.Sequential(
            PeriodicConv2D(conv_start_size, 2*conv_start_size, kernel_size=3, padding=1),
            nn.SiLU(),   # new, remove two layers to reduce computation
            PeriodicConv2D(2*conv_start_size, n_sh_coeff, kernel_size=1, padding=0), # # new, remove (two layers, and the *) to reduce computation
        )
        self.projection = PeriodicConv2D(n_sh_coeff, conv_start_size, kernel_size=1)
        self.projection_A = PeriodicConv2D(in_channels, n_sh_coeff, kernel_size=1)
        self.n_sh_coeff = n_sh_coeff
        self.in_channels = in_channels
    def forward(self, A):  # [B, C, H, W] 
        B, _,_, H, W = A.size()
        C = self.in_channels
        loc_basis = self.basis.view(-1, self.n_sh_coeff, H, W) # [1, 64, H, W]
        # Orography
        oro = self.oro.view(-1, H, W)  # [B, H, W]
        oro = oro.unsqueeze(1)  # [B, 1, H, W]
        # Compute gradients
        dx = oro[:, :, :, 1:] - oro[:, :, :, :-1]  # [B, 1, H, W-1]
        last_col = dx[:, :, :, -1:].clone()        # replicate last column
        grad_x = torch.cat([dx, last_col], dim=-1)  # [B, 1, H, W
        dy = oro[:, :, 1:, :] - oro[:, :, :-1, :]  # [B, 1, H-1, W]
        last_row = dy[:, :, -1:, :].clone()
        grad_y = torch.cat([dy, last_row], dim=-2)  # [B, 1, H, W]
        oro_feat = torch.cat([oro, grad_x, grad_y], dim=1)  # [B, 3, H, W]
        # Encode orography features
        oro_basis = self.oro_encoder(oro_feat)  # [B, n_sh, H, W]
        geo_basis = loc_basis + oro_basis # [1, n_sh, H, W]
        geo_basis = self.siren(geo_basis.permute(0,2,3,1))#.view(self.n_sh_coeff, H, W)
        geo_basis = geo_basis.permute(0, 3, 1, 2) # [B, n_sh, H, W]
        #multi_geo_basis = geo_basis.repeat(1,C , 1, 1)
        print(f"A shape in GeoNoFAR_INR: {A.shape}")
        A = A.squeeze(2)  # [B, C, H, W] = [16, 3,534,534]
        print(f"A shape  after squeeze: {A.shape}")
        A = self.projection_A(A)
        print(f"A shape  after projection {A.shape}")
        
        
        fused = geo_basis * A + geo_basis
        fused = self.projection(fused)
        out = self.conv(fused)
        return out  # [B, 1, H, W]
'''
