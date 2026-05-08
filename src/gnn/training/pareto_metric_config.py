def resolve_pareto_metric_keys(pareto_metric_source, pareto_x_metric, pareto_y_metric):
    """
    Resolves Pareto metric keys from either a high-level source selector or explicit keys.

    Args:
        pareto_metric_source: str | None
            Supported values: test, validation, custom, explicit, or empty.
        pareto_x_metric: str | None
            Explicit x-axis metric key used for custom/explicit mode.
        pareto_y_metric: str | None
            Explicit y-axis metric key used for custom/explicit mode.
    return: tuple
            (pareto_x_metric_key, pareto_y_metric_key)
    """
    source_norm = ""
    if pareto_metric_source is not None:
        source_norm = str(pareto_metric_source).strip().lower()
    source_to_keys = {
        "test": ("final_test/acc_all_rel_mae", "final_test/stress_rel_mae"),
        "validation": ("final_validation/acc_all_rel_mae", "final_validation/stress_rel_mae"),
    }
    if source_norm in source_to_keys:
        return source_to_keys[source_norm]
    if source_norm in ("", "custom", "explicit"):
        if pareto_x_metric is None or pareto_y_metric is None:
            raise ValueError("pareto_x_metric and pareto_y_metric must be configured when pareto_metric_source is custom/explicit/empty.")
        return str(pareto_x_metric), str(pareto_y_metric)
    raise ValueError(
        "Unsupported pareto_metric_source=%s (use test, validation, or custom)." % source_norm
    )
