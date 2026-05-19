"""
eda_features_helpers.py

Feature names for EDA plots must stay aligned with build_node_features / build_edge_features in
src/gnn/preprocessing/build_features.py. Use verify_eda_feature_names_match_builders() (runs
probe_feature_layout) to fail fast on drift. Scalar routing for distribution plots uses
build_scalar_feature_names() so it stays aligned with the same builders (single-channel columns only).
"""


def _xyz_names(prefix):
    """
    Args:
        prefix: str
            Feature name prefix.
        return: list[str]
            xyz-expanded feature names.
    """
    return [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]


def build_node_feature_names(cfg):
    """
    Args:
        cfg: object
            EDA/preprocess config.
        return: list[str]
            Node valued feature names in the exact order returned by build_node_features().
    """
    enable_velocity_node = cfg.enable_velocity_node
    enable_displacement_node = cfg.enable_displacement_node
    enable_force_node = cfg.enable_force_node
    enable_rest_pos_node = cfg.enable_rest_pos_node
    names = []
    if enable_velocity_node:
        names.extend(_xyz_names("velocity"))
    if enable_displacement_node:
        names.extend(_xyz_names("displacement_pos"))
    if enable_force_node:
        names.extend(_xyz_names("force"))
    if cfg.enable_past_history:
        if cfg.enable_absolute_pos:
            names.extend(_xyz_names("position_past"))
        if enable_displacement_node:
            names.extend(_xyz_names("displacement_pos_past"))
    if cfg.enable_absolute_pos:
        names.extend(_xyz_names("position"))
        if enable_rest_pos_node:
            names.extend(_xyz_names("rest_pos"))
    if cfg.enable_node_norms:
        if enable_displacement_node:
            names.append("displacement_norm")
        if enable_force_node:
            names.append("force_norm")
        if enable_velocity_node:
            names.append("velocity_norm")
        # position_norm / rest_pos_norm intentionally omitted (same as build_features)
        if cfg.enable_past_history and enable_displacement_node:
            names.append("displacement_pos_past_norm")
    return names


def build_edge_feature_names(cfg):
    """
    Args:
        cfg: object
            EDA/preprocess config.
        return: list[str]
            Edge feature names in the exact order returned by build_edge_features().
    """
    names = []
    # Vector features (xyz triplets), same order as build_edge_features
    if cfg.enable_displ_edge:
        names.extend(_xyz_names("edge_displacement_pos"))
    if cfg.enable_edge_velocity:
        names.extend(_xyz_names("edge_velocity_pos"))
    if cfg.enable_force_edge:
        names.extend(_xyz_names("edge_force_pos"))
    if cfg.enable_abs_pos_edge:
        names.extend(_xyz_names("edge_relative_pos"))
    if cfg.enable_abs_rest_pos_edge:
        names.extend(_xyz_names("edge_rest_relative_pos"))
    if cfg.enable_past_history:
        if cfg.enable_displ_edge:
            names.extend(_xyz_names("edge_displacement_pos_past"))
        if cfg.enable_abs_pos_edge:
            names.extend(_xyz_names("edge_relative_pos_past"))
        if cfg.enable_abs_pos_edge_norm:
            names.append("edge_relative_pos_past_norm")
    # Scalar features (norms)
    if cfg.enable_abs_pos_edge_norm:
        names.append("edge_relative_pos_norm")
    if cfg.enable_rest_pos_edge_norm:
        names.append("edge_rest_relative_pos_norm")
    if cfg.enable_force_edge_norm:
        names.append("edge_force_norm")
    if cfg.enable_edge_velocity_norm:
        names.append("edge_velocity_norm")
    if cfg.enable_stretch_edge:
        names.append("edge_stretch_norm")
        if cfg.enable_past_history:
            names.append("edge_stretch_norm_past")
    if cfg.enable_rotation_angle:
        names.append("edge_direction_cosine")
        if cfg.enable_past_history:
            names.append("edge_direction_cosine_past")
    return names


def build_scalar_feature_names(cfg):
    """
    Single-channel EDA feature names for plot routing (scalar vs vector normalization runs).
    von_mises is the scalar stress target; node/edge scalars are every build_* name that is not an xyz triplet.

    Args:
        cfg: object
            EDA/preprocess config with the same feature toggles as build_features.

    return: list[str]
        Stable order: von_mises, then node scalar columns, then edge scalar columns.
    """
    names = ["von_mises"]
    for n in build_node_feature_names(cfg):
        if n.endswith("_x") or n.endswith("_y") or n.endswith("_z"):
            continue
        names.append(n)
    for n in build_edge_feature_names(cfg):
        if n.endswith("_x") or n.endswith("_y") or n.endswith("_z"):
            continue
        names.append(n)
    return names


def verify_eda_feature_names_match_builders(preprocess_cfg, device):
    """
    Ensure name list lengths match tensor widths from build_node_features / build_edge_features.

    Args:
        preprocess_cfg: object
            Config with feature toggles (e.g. PreprocessYaml).
        device: torch.device
            Device for probe tensors.

    return: None
        Raises ValueError if EDA name lists disagree with probe_feature_layout.
    """
    from src.gnn.preprocessing.build_features import probe_feature_layout

    node_ch, edge_ch, _, _ = probe_feature_layout(preprocess_cfg, device)
    valued_node_dim = node_ch - 1
    n_node_names = len(build_node_feature_names(preprocess_cfg))
    n_edge_names = len(build_edge_feature_names(preprocess_cfg))
    if n_node_names != valued_node_dim:
        raise ValueError(
            "EDA node feature name count (%s) != valued node width from build_node_features (%s). "
            "Sync build_node_feature_names with build_features.build_node_features."
            % (n_node_names, valued_node_dim))
    if n_edge_names != edge_ch:
        raise ValueError(
            "EDA edge feature name count (%s) != width from build_edge_features (%s). "
            "Sync build_edge_feature_names with build_features.build_edge_features."
            % (n_edge_names, edge_ch))


class EdaObj:
    feature_names: list
    node_feature_names: list
    edge_feature_names: list

    def __init__(self, node_feature_names, edge_feature_names):
        """
        Args:
            node_feature_names: list[str]
                Node feature names in valued-node order (without bc).
            edge_feature_names: list[str]
                Edge feature names in edge feature order.
        """
        self.node_feature_names = list(node_feature_names)
        self.edge_feature_names = list(edge_feature_names)
        self.feature_names = ["acceleration_x", "acceleration_y", "acceleration_z", "von_mises",
                              *self.node_feature_names, *self.edge_feature_names]
        for name in self.feature_names:
            setattr(self, name, [])

    def append_timestep(self, node_features, edge_features, target):
        """
        Args:
            node_features: torch.Tensor
                Node input tensor `[bc, valued_node_features]`.
            edge_features: torch.Tensor
                Edge feature tensor.
            target: torch.Tensor
                Target tensor `[acc_x, acc_y, acc_z, von_mises]`.
            return: None
                Appends the current timestep to the internal EDA lists.
        """
        self.acceleration_x.append(target[:, 0])
        self.acceleration_y.append(target[:, 1])
        self.acceleration_z.append(target[:, 2])
        self.von_mises.append(target[:, 3])

        valued_node_features = node_features[:, 1:]
        if valued_node_features.shape[1] != len(self.node_feature_names):
            raise ValueError(f"Node feature mismatch: tensor has {valued_node_features.shape[1]} columns but "
                             f"{len(self.node_feature_names)} names were configured.")
        if edge_features.shape[1] != len(self.edge_feature_names):
            raise ValueError(f"Edge feature mismatch: tensor has {edge_features.shape[1]} columns but "
                             f"{len(self.edge_feature_names)} names were configured.")

        for column_idx, feature_name in enumerate(self.node_feature_names):
            getattr(self, feature_name).append(valued_node_features[:, column_idx])
        for column_idx, feature_name in enumerate(self.edge_feature_names):
            getattr(self, feature_name).append(edge_features[:, column_idx])