"""Helper utilities for seed setting and parameter counting."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["WANDB_DISABLED"] = "true"


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Calculates total and trainable parameters of a PyTorch module.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100) if total > 0 else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "pct_trainable": round(pct, 4)
    }
