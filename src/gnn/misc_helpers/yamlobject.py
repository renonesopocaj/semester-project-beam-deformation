"""
yamlobject.py
Maps yaml keys to instance attributes, via flattening the cfg.
"""

from omegaconf import OmegaConf, DictConfig
from src.gnn.preprocessing.edge_augmentation import validate_edge_augmentation_config

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
        self.dt = flattened_cfg["dt"]
        self.seed = flattened_cfg["seed"]
        self.raw_data_dir = flattened_cfg["raw_data_dir"]
        self.base_dir = flattened_cfg["base_dir"]
        self.save_dir = flattened_cfg["save_dir"]
        self.limit_timesteps = flattened_cfg["limit_timesteps"]
        self.mises_log = flattened_cfg["mises_log"]
        self.isotropic_normalizer = flattened_cfg["isotropic_normalizer"]
        self.normalizer_method = flattened_cfg["normalizer_method"]
        self.dataset_type = flattened_cfg["dataset_type"]
        self.time_discretization_method = flattened_cfg["time_discretization_method"]
        # Target layout: "von_mises" (y=[acc, vm], 4 ch) or "full_stress" (y=[acc, sig6], 9 ch).
        # Defaults to "von_mises" for backward compatibility with existing processed datasets.
        self.target_mode = flattened_cfg["target_mode"]
        if self.target_mode not in ("von_mises", "full_stress"):
            raise ValueError(
                f"Unknown preprocess.target_mode '{self.target_mode}'. Expected 'von_mises' or 'full_stress'.")
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
        self.extra_info_dir = flattened_cfg["extra_info_dir"]
        # Features enablers
        self.name_features = flattened_cfg["name_features"]
        self.enable_edge_velocity = flattened_cfg["enable_edge_velocity"]
        self.enable_absolute_pos = flattened_cfg["enable_absolute_pos"]
        self.enable_node_norms = flattened_cfg["enable_node_norms"]
        self.enable_stretch_edge = flattened_cfg["enable_stretch_edge"]
        self.enable_past_history = flattened_cfg["enable_past_history"]
        self.enable_force_edge = flattened_cfg["enable_force_edge"]
        self.enable_rotation_angle = flattened_cfg["enable_rotation_angle"]
        self.enable_displ_edge = flattened_cfg["enable_displ_edge"]
        self.enable_abs_pos_edge = flattened_cfg["enable_abs_pos_edge"]
        self.enable_abs_rest_pos_edge = flattened_cfg["enable_abs_rest_pos_edge"]
        self.enable_abs_pos_edge_norm = flattened_cfg["enable_abs_pos_edge_norm"]
        self.enable_rest_pos_edge_norm = flattened_cfg["enable_rest_pos_edge_norm"]
        self.enable_force_edge_norm = flattened_cfg["enable_force_edge_norm"]
        self.enable_edge_velocity_norm = flattened_cfg["enable_edge_velocity_norm"]
        # Per-node feature granular toggles
        self.enable_velocity_node = flattened_cfg["enable_velocity_node"]
        self.enable_displacement_node = flattened_cfg["enable_displacement_node"]
        self.enable_force_node = flattened_cfg["enable_force_node"]
        self.enable_rest_pos_node = flattened_cfg["enable_rest_pos_node"]

        edge_aug = _extract_nested(cfg, "edge_augmentation")
        if edge_aug is None:
            self.edge_aug_k_hop = 0
            self.edge_aug_random_offline = 0.0
            self.edge_aug_add_wall_edges = None
            self.edge_aug_wall_border_percentage = 0.0
            self.edge_aug_wall_internal_percentage = 0.0
        else:
            if isinstance(edge_aug, DictConfig):
                edge_aug = OmegaConf.to_container(edge_aug, resolve=True)
            kh = edge_aug.get("k_hop", 0)
            if isinstance(kh, str):
                self.edge_aug_k_hop = kh
            else:
                self.edge_aug_k_hop = int(kh)
            self.edge_aug_random_offline = float(edge_aug.get("random_offline", 0.0))
            raw_wall = edge_aug.get("add_wall_edges", None)
            if raw_wall is None or raw_wall == "null":
                self.edge_aug_add_wall_edges = None
            else:
                self.edge_aug_add_wall_edges = str(raw_wall)
            self.edge_aug_wall_border_percentage = float(edge_aug.get("wall_border_percentage", 0.0))
            self.edge_aug_wall_internal_percentage = float(edge_aug.get("wall_internal_percentage", 0.0))
        validate_edge_augmentation_config(self.edge_aug_k_hop, self.edge_aug_random_offline,
                                          self.edge_aug_add_wall_edges, self.edge_aug_wall_border_percentage,
                                          self.edge_aug_wall_internal_percentage)

class TrainYaml:
    """
    Dynamically maps all yaml keys from a nested config dictionary to instance attributes, ignoring nesting structure.
    """

    wandb_run_name: str | None
    wandb_group: str | None
    wandb_project: str
    wandb_offline: bool
    wandb_init_timeout: int
    random_online: float

    def __init__(self, cfg):
        """
        Args:
            cfg: dict
                Configuration dictionary loaded from YAML.
            return: None
                Creates attributes on self dynamically.
        """
        flattened_cfg = _to_dict_flatten(cfg)
        self.wandb_run_name = flattened_cfg.get("wandb_run_name")
        self.wandb_group = flattened_cfg.get("wandb_group")
        self.wandb_project = flattened_cfg.get("wandb_project", "beam_deformation")
        offline_raw = flattened_cfg.get("wandb_offline")
        self.wandb_offline = True if offline_raw is None else bool(offline_raw)
        self.wandb_init_timeout = int(flattened_cfg.get("wandb_init_timeout"))
        self.seed = flattened_cfg["seed"]
        self.use_wandb = flattened_cfg["use_wandb"]
        self.batch_size = flattened_cfg["batch_size"]
        self.num_workers = flattened_cfg["num_workers"]
        self.model_name = flattened_cfg["model_name"]
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
        self.precision = flattened_cfg["precision"]
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
        self.val_monitor_metric = flattened_cfg.get("val_monitor_metric")
        self.use_best = bool(flattened_cfg.get("use_best"))
        self.curvature_loss_weight = flattened_cfg["curvature_loss_weight"]
        self.volume_loss_weight = float(flattened_cfg.get("volume_loss_weight", 1.0))
        self.optimizer = flattened_cfg["optimizer"]
        self.fused_adam = flattened_cfg["fused_adam"]
        # Notes various
        self.pin_memory = flattened_cfg["pin_memory"]
        self.rollout_batch_size = flattened_cfg["rollout_batch_size"]
        self.extra_info_dir = flattened_cfg["extra_info_dir"]
        # Noise
        self.noise_scale_train = flattened_cfg["noise_scale_train"]
        self.noise_gamma = flattened_cfg["noise_gamma"]
        self.noise_seed = flattened_cfg["noise_seed"]
        self.noise_also_prev = flattened_cfg["noise_also_prev"]
        self.noise_prev_scale_train = flattened_cfg["noise_prev_scale_train"]
        self.noise_feature_update_mode = flattened_cfg["noise_feature_update_mode"]
        self.noise_per_epoch = flattened_cfg["noise_per_epoch"]
        self.random_online = float(flattened_cfg.get("random_online"))
        if self.random_online < 0.0 or self.random_online > 1.0:
            raise ValueError("random_online must be in [0, 1].")
        # Exponentiate
        self.exp_mises = flattened_cfg["exp_mises"]
        self.stress_weight = flattened_cfg["stress_weight"]
        # Full-stress loss flags (no-op when preprocess.target_mode == "von_mises").
        # "components" -> plain weighted MSE/MAE/Huber on [acc(3), sig6(6)] target.
        # "components_plus_vm" -> same plus a differentiable VM regularizer of weight `full_stress_vm_weight`.
        self.full_stress_loss_mode = flattened_cfg.get("full_stress_loss_mode")
        self.full_stress_vm_weight = float(flattened_cfg.get("full_stress_vm_weight"))
        # Misc
        self.protect_from_oom = flattened_cfg["protect_from_oom"]
        self.activation_checkpointing = flattened_cfg["activation_checkpointing"]
        self.foreach_optim = flattened_cfg["foreach_optim"]
        # Models
        if self.model_name == "MGN":
            self.num_layers_graphnetblock = flattened_cfg["num_layers_graphnetblock"]
            self.layer_norm_enable_encoder = flattened_cfg["layer_norm_enable_encoder"]
            self.layer_norm_enable_processor = flattened_cfg["layer_norm_enable_processor"]
            self.layer_norm_enable_decoder = flattened_cfg["layer_norm_enable_decoder"]
            self.num_messages = flattened_cfg["num_messages"]
            self.hidden_processor_dim = flattened_cfg["latent_dim"]
            self.num_layers_heads = flattened_cfg["num_layers_heads"]
            self.num_layers_encoder = flattened_cfg["num_layers_encoder"]
            self.dropout_encoder = flattened_cfg["dropout_encoder"]
            self.dropout_head = flattened_cfg["dropout_head"]
            self.dropout_processor = flattened_cfg["dropout_processor"]
        elif self.model_name == "gunet":
            self.hidden_channels = flattened_cfg["hidden_channels"]
            self.layer_norm = flattened_cfg["layer_norm"]
            self.depth = flattened_cfg["depth"]
            self.pool_ratios = flattened_cfg["pool_ratios"]
            self.sum_res = flattened_cfg["sum_res"]
            self.enable_embedder = flattened_cfg["enable_embedder"]
            self.encoder_layers = flattened_cfg["encoder_layers"]
            self.decoder_layers = flattened_cfg["decoder_layers"]
            self.dropout_encoder = flattened_cfg["dropout_encoder"]
            self.dropout_head = flattened_cfg["dropout_head"]
        elif self.model_name == "magnet":
            self.hidden_channels = flattened_cfg["hidden_channels"]
            self.num_poolings = flattened_cfg["num_poolings"]
            self.layers_per_level = flattened_cfg["layers_per_level"]
            self.window_size = flattened_cfg["window_size"]
            self.pool_seed = flattened_cfg["pool_seed"]
            self.pool_aggr = flattened_cfg["pool_aggr"]
            self.activation = flattened_cfg["activation"]
            self.dropout_encoder = flattened_cfg["dropout_encoder"]
            self.dropout_head = flattened_cfg["dropout_head"]
            self.dropout_processor = flattened_cfg["dropout_processor"]
        elif self.model_name == "clofnet":
            self.clofnet_variant = flattened_cfg["clofnet_variant"]
            self.hidden_dim = flattened_cfg["hidden_dim"]
            self.n_layers = flattened_cfg["n_layers"]
            self.coords_weight = flattened_cfg["coords_weight"]
            self.recurrent = flattened_cfg["recurrent"]
            self.enable_norm_diff = flattened_cfg["enable_norm_diff"]
            self.tanh = flattened_cfg["tanh"]
            self.attention_enable = flattened_cfg["attention_enable"]
            self.clamp_enable = flattened_cfg["clamp_enable"]
            self.out_coord_mlp_dim = flattened_cfg["out_coord_mlp_dim"]
            self.embedding_edge_dim = flattened_cfg["embedding_edge_dim"]
            self.clofnet_activation = flattened_cfg["clofnet_activation"]
            self.leakyrelu_negative_slope = flattened_cfg["leakyrelu_negative_slope"]
            self.use_neighbor_diff_norms = flattened_cfg["use_neighbor_diff_norms"]
            self.frame_source = flattened_cfg["frame_source"]
            self.include_u_tm1 = flattened_cfg["include_u_tm1"]
            self.rescalarize_per_layer = flattened_cfg["rescalarize_per_layer"]
            self.include_v_branch = flattened_cfg["include_v_branch"]
            self.include_force = flattened_cfg["include_force"]
            self.m_ij_init = flattened_cfg["m_ij_init"]
            self.n_layers_gate_mlp = flattened_cfg["n_layers_gate_mlp"]
            self.n_layers_node_mlp = flattened_cfg["n_layers_node_mlp"]
            self.n_layers_message_mlp = flattened_cfg["n_layers_message_mlp"]
            self.eps_mode = flattened_cfg["eps_mode"]
            self.normalize_delta_a = flattened_cfg["normalize_delta_a"]
            self.mlp_layer_norm = flattened_cfg["mlp_layer_norm"]
        elif self.model_name == "transolver":
            self.transolver_n_layers = flattened_cfg["transolver_n_layers"]
            self.transolver_n_hidden = flattened_cfg["transolver_n_hidden"]
            self.transolver_dropout = flattened_cfg["transolver_dropout"]
            self.transolver_n_head = flattened_cfg["transolver_n_head"]
            self.transolver_time_input = flattened_cfg["transolver_time_input"]
            self.transolver_act = flattened_cfg["transolver_act"]
            self.transolver_mlp_ratio = flattened_cfg["transolver_mlp_ratio"]
            self.transolver_slice_num = flattened_cfg["transolver_slice_num"]
            self.transolver_ref = flattened_cfg["transolver_ref"]
            self.transolver_unified_pos = flattened_cfg["transolver_unified_pos"]
        elif self.model_name == "gatv2":
            self.hidden_dim = flattened_cfg["hidden_dim"]
            self.num_layers = flattened_cfg["num_layers"]
            self.heads = flattened_cfg["heads"]
            self.dropout_attn = flattened_cfg["dropout_attn"]
            self.negative_slope = flattened_cfg["negative_slope"]
            self.add_self_loops = flattened_cfg["add_self_loops"]
            self.share_weights = flattened_cfg["share_weights"]
            self.residual = flattened_cfg["residual"]
            self.layer_norm_encoder = flattened_cfg["layer_norm_encoder"]
            self.layer_norm_processor = flattened_cfg["layer_norm_processor"]
            self.layer_norm_decoder = flattened_cfg["layer_norm_decoder"]
            self.num_layers_encoder = flattened_cfg["num_layers_encoder"]
            self.num_layers_heads = flattened_cfg["num_layers_heads"]
            self.dropout_encoder = flattened_cfg["dropout_encoder"]
            self.dropout_head = flattened_cfg["dropout_head"]
        elif self.model_name == "gatmgn":
            self.hidden_dim = flattened_cfg["hidden_dim"]
            self.num_layers = flattened_cfg["num_layers"]
            self.num_layers_graphnetblock = flattened_cfg["num_layers_graphnetblock"]
            self.heads = flattened_cfg["heads"]
            self.dropout_attn = flattened_cfg["dropout_attn"]
            self.negative_slope = flattened_cfg["negative_slope"]
            self.add_self_loops = flattened_cfg["add_self_loops"]
            self.share_weights = flattened_cfg["share_weights"]
            self.residual = flattened_cfg["residual"]
            self.layer_norm_encoder = flattened_cfg["layer_norm_encoder"]
            self.layer_norm_processor = flattened_cfg["layer_norm_processor"]
            self.layer_norm_decoder = flattened_cfg["layer_norm_decoder"]
            self.num_layers_encoder = flattened_cfg["num_layers_encoder"]
            self.num_layers_heads = flattened_cfg["num_layers_heads"]
            self.dropout_encoder = flattened_cfg["dropout_encoder"]
            self.dropout_head = flattened_cfg["dropout_head"]
            self.dropout_processor = flattened_cfg["dropout_processor"]

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

def cfg_for_wandb(cfg):
    """
    Resolved train config as plain nested dict, without Hydra metadata (for W&B).
    Helper for train.py

    Args:
        cfg: omegaconf.DictConfig
            Full Hydra-composed training config.
    return: dict
        JSON-friendly nested mapping suitable for wandb.config.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(container, dict) and "hydra" in container:
        container = {key: value for key, value in container.items() if key != "hydra"}
    return container


def train_cfg_yaml_for_wandb(cfg):
    """
    Same content as the resolved YAML printed in main(), minus the hydra: block.
    Helper for train.py

    Args:
        cfg: omegaconf.DictConfig
            Full Hydra-composed training config.
    return: str
        Multi-line YAML string for wandb.summary.
    """
    return OmegaConf.to_yaml(OmegaConf.create(cfg_for_wandb(cfg)), resolve=True)


def assert_target_mode_consistent(cfg, preprocess_dict, preprocess_config_path):
    """
    Raises ValueError when the train config declares a target_mode that does not match the
    target_mode baked into the processed dataset.  The training pipeline reads target_mode
    exclusively from preprocess_cfg (the on-disk dataset config), so a mismatched train-side
    override would be silently ignored, causing every sweep trial to train identically on the
    wrong mode.

    Args:
        cfg: omegaconf.DictConfig
            Resolved Hydra train config (may have a top-level target_mode key).
        preprocess_dict: dict
            Preprocess config loaded from the dataset's config.yaml.
        preprocess_config_path: str or Path
            Path displayed in the error message for easier debugging.
    """
    cfg_target_mode = OmegaConf.select(cfg, "target_mode")
    if cfg_target_mode is None:
        return
    dataset_target_mode = preprocess_dict.get("target_mode", "von_mises")
    if str(cfg_target_mode) != str(dataset_target_mode):
        raise ValueError(
            f"train config has target_mode='{cfg_target_mode}' but the dataset at "
            f"'{preprocess_config_path}' was preprocessed with "
            f"target_mode='{dataset_target_mode}'. "
            f"Point data.dataset_dir (or use data=full_stress / data=slurm_full_stress) at the "
            f"matching processed dataset. Full-stress datasets are stored under a directory "
            f"whose name ends with '_fs' (e.g. z_isotr_centr_nonlin_fs)."
        )
