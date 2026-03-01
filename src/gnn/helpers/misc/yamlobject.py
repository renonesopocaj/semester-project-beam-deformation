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
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(cfg, DictConfig):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            pass
        flattened_cfg = _flatten_dict(cfg)
        self.dataset_size = flattened_cfg["dataset_size"]
        self.split = flattened_cfg["split"]
        self.device = flattened_cfg["device"]
        self.noise_scale_train = flattened_cfg["noise_scale_train"]
        self.dt = flattened_cfg["dt"]
        self.noise_scale_test = flattened_cfg["noise_scale_test"]
        self.seed = flattened_cfg["seed"]
        self.graphs_dir = flattened_cfg["graphs_dir"]
        self.raw_data_dir = flattened_cfg["raw_data_dir"]
        self.base_dir = flattened_cfg["base_dir"]
        self.test_graphs_dir = flattened_cfg["test_graphs_dir"]
        self.save_dir = flattened_cfg["save_dir"]
        self.train_save_dir = flattened_cfg["train_save_dir"]
        self.limit_timesteps = flattened_cfg["limit_timesteps"]
        self.mises_log = flattened_cfg["mises_log"]
        self.isotropic_norm = flattened_cfg["isotropic_norm"]

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
        flattened_cfg = _flatten_dict(cfg)
        self.batch_size = flattened_cfg["batch_size"]
        self.num_workers = flattened_cfg["num_workers"]
        self.path_val = flattened_cfg["path_val"]
        self.path_train = flattened_cfg["path_train"]
        self.seed = flattened_cfg["seed"]
        self.name = flattened_cfg["name"]
        self.num_messages = flattened_cfg["num_messages"]
        self.latent_dim = flattened_cfg["latent_dim"]
        self.lr = flattened_cfg["lr"]
        self.total_epochs = flattened_cfg["total_epochs"]
        self.validation_interval = flattened_cfg["validation_interval"]
        self.loss_type = flattened_cfg["loss_type"]
        self.use_wandb = flattened_cfg["use_wandb"]

def _flatten_dict(data):
    """
    Args:
        data: dict
            Possibly nested dictionary.
        return: dict
            Flattened dictionary containing only leaf key-value pairs.
    """
    flat = {}

    for key, value in data.items():
        if isinstance(value, dict):
            nested_flat = _flatten_dict(value)
            for nested_key, nested_value in nested_flat.items():
                flat[nested_key] = nested_value
        else:
            flat[key] = value

    return flat