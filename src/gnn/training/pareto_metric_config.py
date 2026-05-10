def resolve_pareto_metric_pairs(pareto_metric_source, pareto_x_metric, pareto_y_metric):
    """
    Resolves Pareto metric pairs from either a high-level source selector or explicit keys/lists.

    When pareto_x_metric and pareto_y_metric are both provided (as strings or lists), the
    pareto_metric_source shorthand is ignored. Each index i defines one (x, y) pair, producing
    one Pareto plot. Lists must have the same length.

    Supported pareto_metric_source shortcuts (used only when explicit metrics are absent):
        test             → [(final_test/acc_all_rel_mae, final_test/stress_rel_mae)]
        test_abs         → [(final_test/acc_all_abs_mae, final_test/stress_abs_mae)]
        validation       → [(final_validation/acc_all_rel_mae, final_validation/stress_rel_mae)]
        validation_abs   → [(final_validation/acc_all_abs_mae, final_validation/stress_abs_mae)]

    Args:
        pareto_metric_source: str | None
            Shorthand source selector; ignored when explicit metrics are provided.
        pareto_x_metric: str | list | None
            X-axis metric key(s).
        pareto_y_metric: str | list | None
            Y-axis metric key(s).
    return: list of tuple
        List of (x_key, y_key) pairs; one Pareto plot per pair.
    """
    x_raw = pareto_x_metric
    y_raw = pareto_y_metric

    if x_raw is not None and y_raw is not None:
        x_list = _to_str_list(x_raw)
        y_list = _to_str_list(y_raw)
        if len(x_list) != len(y_list):
            raise ValueError(
                "pareto_x_metric and pareto_y_metric must have the same length "
                "(got %d vs %d)." % (len(x_list), len(y_list))
            )
        return list(zip(x_list, y_list))

    source_norm = ""
    if pareto_metric_source is not None:
        source_norm = str(pareto_metric_source).strip().lower()

    source_to_pairs = {
        "test": [("final_test/acc_all_rel_mae", "final_test/stress_rel_mae")],
        "test_abs": [("final_test/acc_all_abs_mae", "final_test/stress_abs_mae")],
        "validation": [("final_validation/acc_all_rel_mae", "final_validation/stress_rel_mae")],
        "validation_abs": [("final_validation/acc_all_abs_mae", "final_validation/stress_abs_mae")],
    }
    if source_norm in source_to_pairs:
        return source_to_pairs[source_norm]

    raise ValueError(
        "Cannot resolve Pareto metrics: provide pareto_x_metric + pareto_y_metric lists, "
        "or set pareto_metric_source to one of: %s." % ", ".join(source_to_pairs.keys())
    )


def _to_str_list(value):
    """
    Coerces a string, OmegaConf ListConfig, or plain list to a list of str.

    Args:
        value: str | list | ListConfig
    return: list of str
    """
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]
