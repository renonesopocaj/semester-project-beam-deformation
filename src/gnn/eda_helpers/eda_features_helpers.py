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
    names = []
    names.extend(_xyz_names("position"))
    names.extend(_xyz_names("rest_pos"))
    names.extend(_xyz_names("velocity"))
    names.extend(_xyz_names("displacement_pos"))
    names.extend(_xyz_names("force"))
    if cfg.enable_past_pos_displ:
        names.extend(_xyz_names("position_past"))
        names.extend(_xyz_names("displacement_pos_past"))
    if cfg.enable_node_norms:
        names.append("position_norm")
        names.append("displacement_norm")
        names.append("force_norm")
        names.append("velocity_norm")
        names.append("rest_pos_norm")
        if cfg.enable_past_pos_displ:
            names.extend(_xyz_names("position_past_norm"))
            names.extend(_xyz_names("displacement_pos_past_norm"))
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
    names.extend(_xyz_names("edge_relative_pos"))
    names.extend(_xyz_names("edge_rest_relative_pos"))
    names.extend(_xyz_names("edge_displacement_pos"))
    if cfg.enable_edge_velocity:
        names.extend(_xyz_names("edge_velocity_pos"))
    if cfg.enable_force_edge:
        names.extend(_xyz_names("edge_force_pos"))
    names.append("edge_relative_pos_norm")
    names.append("edge_rest_relative_pos_norm")
    names.append("edge_displacement_norm")
    if cfg.enable_past_pos_displ:
        names.append("edge_displacement_pos_past")
        names.append("edge_displacement_pos_past_norm")
    if cfg.enable_force_edge:
        names.append("edge_force_norm")
    if cfg.enable_edge_velocity:
        names.append("edge_velocity_norm")
    if cfg.enable_stretch_edge:
        names.append("edge_stretch_norm")
    if cfg.enable_rotation_angle:
        names.append("edge_direction_cosine")
    return names

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