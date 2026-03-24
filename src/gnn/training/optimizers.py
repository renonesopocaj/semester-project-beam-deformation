"""
optimizers.py
Contains helper functions for creating optimizers.
"""

import torch

def optimizer_factory(model, train_cfg):
    if train_cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_cfg.initial_lr, fused=True)
    elif train_cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_cfg.initial_lr, fused=True)
    else:
        raise NotImplementedError(f"Optimizer {train_cfg.optimizer} not implemented")