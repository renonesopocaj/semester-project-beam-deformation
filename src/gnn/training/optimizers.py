"""
optimizers.py

Contains helper functions for creating optimizers.
"""

import torch

def optimizer_factory(model, train_cfg):
    if train_cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_cfg.initial_lr, fused=train_cfg.fused_adam,
                                foreach=train_cfg.foreach_optim)
    elif train_cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_cfg.initial_lr,
                                 weight_decay=train_cfg.adamw_weight_decay,
                                 betas=(train_cfg.adamw_beta1, train_cfg.adamw_beta2),
                                 eps=train_cfg.adamw_eps, amsgrad=train_cfg.adamw_amsgrad,
                                 fused=train_cfg.fused_adam, foreach=train_cfg.foreach_optim)
    else:
        raise NotImplementedError(f"Optimizer {train_cfg.optimizer} not implemented")