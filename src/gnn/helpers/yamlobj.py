class YamlObj:
    """
    Dynamically maps all leaf keys from a nested config dictionary
    to instance attributes, ignoring nesting structure.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: dict
                Configuration dictionary loaded from YAML.
            return: None
                Creates attributes on self dynamically.
        """
        flattened_cfg = self._flatten_dict(cfg)

        for key, value in flattened_cfg.items():
            setattr(self, key, value)

    def _flatten_dict(self, data):
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
                nested_flat = self._flatten_dict(value)
                for nested_key, nested_value in nested_flat.items():
                    flat[nested_key] = nested_value
            else:
                flat[key] = value

        return flat