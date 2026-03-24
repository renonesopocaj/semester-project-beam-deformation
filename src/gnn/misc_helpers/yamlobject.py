"""
yamlobject.py
Maps yaml keys to instance attributes, via flattening the cfg.
"""

from omegaconf import OmegaConf, DictConfig

def _extract_nested(cfg, key):
    """
    Find the first occurrence of `key` in a (possibly nested) dict/DictConfig
    and return its value as a plain Python object (not flattened).

    Args:
        cfg: dict or omegaconf.DictConfig
            The config tree to search.
        key: str
            The key to look for.
        return: object or None
            The value associated with `key`, or None if not found.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    for k, v in cfg.items():
        if k == key:
            return v
        if isinstance(v, dict):
            result = _extract_nested(v, key)
            if result is not None:
                return result
    return None

class EdaYaml:
    raw_data_dir: str
    dataset_type: str
    eda_split: str
    gif_generation_mode: str
    gif_sample_percentage: float
    generate_distributions: bool
    normalization_runs: list
    run_colors: dict
    vector_run_names: list
    scalar_run_names: list
    scalar_feature_names: list

    def __init__(self, cfg):
        """
        Args:
            cfg: dict or omegaconf.DictConfig
                Configuration dictionary loaded from YAML or provided by Hydra.
            return: None
                Creates attributes on self dynamically.
        """
        flattened_cfg = _to_dict_flatten(cfg)
        self.raw_data_dir = flattened_cfg["raw_data_dir"]
        self.dataset_type = flattened_cfg["dataset_type"]
        self.eda_split = flattened_cfg["eda_split"]
        self.gif_generation_mode = flattened_cfg["gif_generation_mode"]
        self.gif_sample_percentage = flattened_cfg["gif_sample_percentage"]
        self.generate_distributions = flattened_cfg["generate_distributions"]
        self.normalization_runs = _extract_nested(cfg, "normalization_runs")
        self.run_colors = _extract_nested(cfg, "run_colors")
        self.vector_run_names = _extract_nested(cfg, "vector_run_names")
        self.scalar_run_names = _extract_nested(cfg, "scalar_run_names")
        self.scalar_feature_names = _extract_nested(cfg, "scalar_feature_names")


class PreprocessYaml:
    """
    Dynamically maps all yaml keys from a nested config dictionary to instance attributes, ignoring nesting structure.
    Accepts either a plain dict (legacy) or an OmegaConf DictConfig (Hydra).
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: dict or omegaconf.DictConfig
                Configuration dictionary loaded from YAML or provided by Hydra.
            return: None
                Creates attributes on self dynamically.
        """

        flattened_cfg = _to_dict_flatten(cfg)
        self.dataset_size = flattened_cfg["dataset_size"]
        self.device = flattened_cfg["device"]
        self.noise_scale_train = flattened_cfg["noise_scale_train"]
        self.dt = flattened_cfg["dt"]
        self.noise_scale_test = flattened_cfg["noise_scale_test"]
        self.seed = flattened_cfg["seed"]
        self.raw_data_dir = flattened_cfg["raw_data_dir"]
        self.base_dir = flattened_cfg["base_dir"]
        self.save_dir = flattened_cfg["save_dir"]
        self.limit_timesteps = flattened_cfg["limit_timesteps"]
        self.mises_log = flattened_cfg["mises_log"]
        self.isotropic_normalizer = flattened_cfg["isotropic_normalizer"]
        self.normalizer_method = flattened_cfg["normalizer_method"]
        self.displacement = flattened_cfg["displacement"]
        self.dataset_type = flattened_cfg["dataset_type"]
        if "edge_channels" in flattened_cfg:
            self.edge_channels = flattened_cfg["edge_channels"]
        if "node_channels" in flattened_cfg:
            self.node_channels = flattened_cfg["node_channels"]
        if "train_dir" in flattened_cfg:
            self.train_dir = flattened_cfg["train_dir"]
        if "test_dir" in flattened_cfg:
            self.test_dir = flattened_cfg["test_dir"]
        if "save_dir_val" in flattened_cfg:
            self.save_dir_val = flattened_cfg["save_dir_val"]


class TrainYaml:
    """
    Dynamically maps all yaml keys from a nested config dictionary to instance attributes, ignoring nesting structure.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: dict
                Configuration dictionary loaded from YAML.
            return: None
                Creates attributes on self dynamically.
        """
        flattened_cfg = _to_dict_flatten(cfg)
        self.seed = flattened_cfg["seed"]
        self.use_wandb = flattened_cfg["use_wandb"]
        self.batch_size = flattened_cfg["batch_size"]
        self.num_workers = flattened_cfg["num_workers"]
        self.model_name = flattened_cfg["model_name"]
        self.num_messages = flattened_cfg["num_messages"]
        self.latent_dim = flattened_cfg["latent_dim"]
        self.initial_lr = float(flattened_cfg["initial_lr"])
        self.scheduler_cfg = _extract_nested(cfg, "lr_scheduler")
        self.total_epochs = flattened_cfg["total_epochs"]
        self.loss_type = flattened_cfg["loss_type"]
        self.enable_val = flattened_cfg["enable_val"]
        self.validation_interval = flattened_cfg["validation_interval"]
        self.dataset_dir = flattened_cfg["dataset_dir"]
        self.train_dir = flattened_cfg["train_dir"]
        self.val_dir = flattened_cfg["val_dir"]
        self.test_dir = flattened_cfg["test_dir"]
        self.stats_path = flattened_cfg["stats_path"]
        self.preprocess_config_path = flattened_cfg["preprocess_config_path"]
        self.resume_from = flattened_cfg.get("resume_from")
        self.checkpoint_interval = flattened_cfg["checkpoint_interval"]
        self.training_type = flattened_cfg["training_type"]
        self.torch_compile = flattened_cfg["torch_compile"]
        self.profiling = flattened_cfg["profiling"]
        self.profile_loss_stress_vs_acceleration = flattened_cfg["profile_loss_stress_vs_acceleration"]
        self.enable_wandb_watch = flattened_cfg["enable_wandb_watch"]
        self.enable_rollout_val = flattened_cfg["enable_rollout_val"]
        self.log_test_media = flattened_cfg["log_test_media"]
        self.devices = flattened_cfg["devices"]
        self.residual_list = flattened_cfg["residual_list"]
        self.plot_diagnostics_advanced = flattened_cfg["plot_diagnostics_advanced"]
        self.selected_validation_trajectories = flattened_cfg["selected_validation_trajectories"]
        self.selected_test_trajectories = flattened_cfg["selected_test_trajectories"]
        self.curvature_loss_weight = flattened_cfg["curvature_loss_weight"]
        self.optimizer = flattened_cfg["optimizer"]
        self.fused_adam = flattened_cfg["fused_adam"]
        self.num_layers_graphnetblock = flattened_cfg["num_layers_graphnetblock"]
        self.layer_norm_enable_encoder = flattened_cfg["layer_norm_enable_encoder"]
        self.layer_norm_enable_processor = flattened_cfg["layer_norm_enable_processor"]
        self.layer_norm_enable_decoder = flattened_cfg["layer_norm_enable_decoder"]

def _to_dict_flatten(data):
    """
    Args:
        data: dict
            Possibly nested dictionary.
        return: dict
            Flattened dictionary containing only leaf key-value pairs.
    """
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested_flat = _to_dict_flatten(value)
            for nested_key, nested_value in nested_flat.items():
                flat[nested_key] = nested_value
        else:
            flat[key] = value
    return flat
