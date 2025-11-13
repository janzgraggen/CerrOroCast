
# IMPORTS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# Third party
import climate_learn as cl
import datetime
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
import json
import os
import argparse
import numpy as _np
PRINTS = True
# PARSER ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

parser = argparse.ArgumentParser()

## OPTIONAL ARGUMENTS
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--logname", type=str, default=None)

## MANDATORY OPTIONAL ARGS
parser.add_argument("--cerra534_dir",type=str,default="dataset/CERRA-534/")
parser.add_argument("--pred_range", type=int, choices=[6, 24, 72, 120, 240],default=6)

## POSITIONAL ARGUMENTS
parser.add_argument("model", choices=["vit","vit","vitcc", "geofar","geofar_v2", "geonofar"])

parser.add_argument("--vis",type=str, default=None,help="If given, visualize the model from the given checkpoint name (without .ckpt) instead of training.")
args = parser.parse_args()
LOG_DIR = f"outputs/{args.model}/{args.logname}" #no Slash logs as it creates a wandb folder anyways
CKPT_DIR = f"outputs/{args.model}/{args.logname}/checkpoints"
PRINT_DIR = f"outputs/{args.model}/{args.logname}/vis"
INFO_DIR = f"outputs/{args.model}/{args.logname}/info"

if args.vis: ## check that chekpoint exists
    ckpt_path = os.path.join(CKPT_DIR, f"{args.vis}.ckpt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path for visualization does not exist: {ckpt_path}")
    args.checkpoint = ckpt_path

# END PARSER ––––––––––––––––––––––––––––––––––––––––––––––––––––––––


# VARIABLES ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
in_vars =  ["2m_temperature"]
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
if PRINTS:print("DataModule ready.")
# END DATA MODULE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# LEARNING MODEL ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Set up deep learning model
in_channels = 1 
out_channels = 1

if args.model == "vit":
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
        "n_coeff": 16, ## <- !!!!! 64
        "n_sh_coeff": 16, ## <- !!!!! 64
        "conv_start_size": 16, ## <- !!!!! 64
        "siren_hidden": 32, ## <- !!!!!  128
    }


elif args.model == "geonofar":
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
if PRINTS: print("Model ready.")

# END LEARNING MODEL ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# SAVE CONFIG IN CASE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def _make_serializable(obj):
    if isinstance(obj, argparse.Namespace):
        return _make_serializable(vars(obj))
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    try:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj

os.makedirs(INFO_DIR, exist_ok=True)
config = {
    "args": _make_serializable(args),
    "model_kwargs": _make_serializable(model_kwargs),
    "optim_kwargs": _make_serializable(optim_kwargs),
    "sched_kwargs": _make_serializable(sched_kwargs),
}
with open(os.path.join(INFO_DIR, "config.json"), "w") as fh:
    json.dump(config, fh, indent=2)

# END SAVE CONFIG IN CASE –––––––––––––––––––––––––––––––––––––––––––––
# START DEFINE OUTPUTS and Trainer ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

pl.seed_everything(0)
if args.logname is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.logname = f"run_{timestamp}"

wandb_logger = WandbLogger(
    project="cerra_534",
    name=args.logname,
    save_dir=LOG_DIR
)
loggers = [wandb_logger]

early_stopping_metric = "val/rmse:aggregate"
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping_metric, patience=args.patience),
    ModelCheckpoint(
        dirpath=CKPT_DIR,
        monitor=early_stopping_metric,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
]
trainer = pl.Trainer(
    logger=loggers,
    callbacks=callbacks,
    default_root_dir="outputs/fallback", 
    max_epochs=args.max_epochs,
    accelerator="gpu" if args.gpu != -1 else None,
    devices=[args.gpu] if args.gpu != -1 else None,
    strategy="ddp",
    precision="32",
)
if PRINTS: print("Trainer ready.")
# START DEFINE OUTPUTS and Trainer ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Train and evaluate model from scratch –––––––––––––-––––-–––––––––

if args.vis is None:
    ### TRAINING MODEL FROM SCRATCH
    if PRINTS: print("Start training.")
    trainer.fit(model, datamodule=dm) 

    ### EVALUATING MODEL
    if PRINTS: print("Training Done. Start evaluation.")
    trainer.test(model, datamodule=dm, ckpt_path="best")
    

# Evaluate saved model checkpoint and visualize ––––-––––––––––––––––––––––––––––––
else:
    if PRINTS: print("Load Model from checkpoint.")
    model = cl.LitModule.load_from_checkpoint(
        args.checkpoint,
        net=model.net,
        optimizer=model.optimizer,
        lr_scheduler=None,
        train_loss=None,
        val_loss=None,
        test_loss=model.test_loss,
        test_target_transforms=model.test_target_transforms,
    )

    #trainer.test(model, datamodule=dm)

    denorm = model.test_target_transforms[0]
    if PRINTS: print("Visualize...")
    cl.utils.visualize_sphere_at_index_save(
        model,
        dm,
        in_transform=denorm,
        out_transform=denorm,
        out_path = PRINT_DIR,
        variable="2m_temperature", #
        src="cerra",
        index=0, # the index of the frame in the dataset you want to visualize
        is_global=False,
        ) 

    