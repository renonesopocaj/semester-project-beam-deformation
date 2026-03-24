"""
optimizers.py
Contains helper functions for creating optimizers.
"""

import torch

def optimizer_factory(model, train_cfg):
    if train_cfg.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=train_cfg.initial_lr)
    elif train_cfg.optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=train_cfg.initial_lr)
    else:
        raise NotImplementedError(f"Optimizer {train_cfg.optimizer} not implemented")