import os
import json
import numpy as np
import argparse

def _make_serializable(obj):
    if isinstance(obj, argparse.Namespace):
        return _make_serializable(vars(obj))
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj

def write_info_json(args, model_kwargs, optim_kwargs, sched_kwargs, loss_kwargs, WRITE_DIR):
    os.makedirs(WRITE_DIR, exist_ok=True)
    config = {
        "args": _make_serializable(args),
        "model_kwargs": _make_serializable(model_kwargs),
        "optim_kwargs": _make_serializable(optim_kwargs),
        "sched_kwargs": _make_serializable(sched_kwargs),
        "loss_kwargs": _make_serializable(loss_kwargs),
    }
    script_path = os.path.abspath(__file__)
    train_cmd = (
        f"python {script_path} {args.model}"
        f" --cerra534_dir {args.cerra534_dir}"
        f" --pred_range {args.pred_range}"
        f" --bs {args.bs}"
        f" --max_epochs {args.max_epochs}"
        f" --patience {args.patience}"
        f" --gpu {args.gpu}"
        f" --summary_depth {args.summary_depth}"
        f" --logname {args.logname}"
    )
    vis_cmd = train_cmd + " --vis epoch_019"
    config["run_commands"] = {"train": train_cmd, "visualize": vis_cmd}
    with open(os.path.join(WRITE_DIR, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)