"""Config loading and result-path helpers shared by every script."""
import argparse
import os
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    return p


def results_dir(cfg, *parts):
    d = REPO_ROOT / cfg["paths"]["results_dir"] / cfg["model"]["slug"]
    d = d.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(cfg, variant):
    """Where the checkpoint for `variant` lives (fp16/nf4 are loaded from the hub id)."""
    return REPO_ROOT / cfg["paths"]["models_dir"] / cfg["model"]["slug"] / variant


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hf_token():
    return os.environ.get("HF_TOKEN") or None
