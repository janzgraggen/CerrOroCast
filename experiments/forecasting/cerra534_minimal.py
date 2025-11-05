
# IMPORTS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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

# PARSER ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

parser = ArgumentParser()

## OPTIONAL ARGUMENTS
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--logname", type=str, default=None)

## POSITIONAL ARGUMENTS
parser.add_argument("cerra534_dir")
parser.add_argument("model", choices=["resnet", "unet", "vit","vit","vitcc", "geofar","geofar_v2"])
parser.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

args = parser.parse_args()
# END PARSER ––––––––––––––––––––––––––––––––––––––––––––––––––––––––


# VARIABLES ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
in_vars = ["2m_temperature"]
out_vars = ["2m_temperature"]
# END VARIABLES ––––––––––––––––––––––––––––––––––––––––––––––––––––––––


# DATA MODULE –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
dm = cl.data.IterDataModule(
    "direct-forecasting",
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
dm.setup()

# END DATA MODULE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# LEARNING MODEL ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Set up deep learning model
in_channels = 1 
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

# Train and evaluate model from scratch –––––––––––––-––––-–––––––––

if args.checkpoint is None:
    ### TRAINING MODEL FROM SCRATCH
    trainer.fit(model, datamodule=dm) 

    ### EVALUATING MODEL
    trainer.test(model, datamodule=dm, ckpt_path="best")
    

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
        trainer.test(model, datamodule=dm)
    