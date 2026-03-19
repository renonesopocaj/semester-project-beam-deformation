"""
custom_pbar.py
Custom progress bar for Lightning to shorten displayed names.
"""

from lightning.pytorch.callbacks import TQDMProgressBar

class ShortMetricProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate, process_position, leave):
        """
        Args:
            refresh_rate: int
                tqdm refresh frequency in batches.
            process_position: int
                Line offset for tqdm display.
            leave: bool
                Whether to keep finished bars visible.
            return: None
                Initializes the custom progress bar.
        """
        super().__init__(refresh_rate=refresh_rate, process_position=process_position, leave=leave)
        self.metric_name_map = {
            "val/final_roll_err_pos": "val/roll_err",
            "val/overall_relative_mae": "val/rel_mae",
            "train/overall_relative_mae": "train/rel_mae",
        }

    def get_metrics(self, trainer, pl_module):
        """
        Args:
            trainer: lightning.pytorch.Trainer
                Active Lightning trainer.
            pl_module: lightning.pytorch.LightningModule
                Active Lightning module.
            return: dict
                Progress-bar metrics with shortened display names.
        """
        items = super().get_metrics(trainer, pl_module)

        renamed_items = {}
        for metric_name, metric_value in items.items():
            short_name = self.metric_name_map.get(metric_name, metric_name)
            renamed_items[short_name] = metric_value

        return renamed_items