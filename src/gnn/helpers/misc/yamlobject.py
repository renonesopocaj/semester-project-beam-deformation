from omegaconf import OmegaConf, DictConfig

class EdaYaml:

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
        self.rest_minus_nonrest_pos = flattened_cfg["rest_minus_nonrest_pos"]
        if "edge_channels" in flattened_cfg:
            self.edge_channels = flattened_cfg["edge_channels"]
        if "node_channels" in flattened_cfg:
            self.node_channels = flattened_cfg["node_channels"]

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
        self.name = flattened_cfg["name"]
        self.num_messages = flattened_cfg["num_messages"]
        self.latent_dim = flattened_cfg["latent_dim"]
        self.lr = float(flattened_cfg["lr"])
        self.lr_scheduler = flattened_cfg["lr_scheduler"]
        self.total_epochs = flattened_cfg["total_epochs"]
        self.loss_type = flattened_cfg["loss_type"]
        self.enable_val = flattened_cfg["enable_val"]
        self.validation_interval = flattened_cfg["validation_interval"]
        self.dataset_dir = flattened_cfg["dataset_dir"]
        self.train_dir = flattened_cfg["train_dir"]
        self.val_dir = flattened_cfg["val_dir"]
        self.stats_path = flattened_cfg["stats_path"]
        self.preprocess_config_path = flattened_cfg["preprocess_config_path"]

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