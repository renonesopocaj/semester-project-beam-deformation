"""
sklearn_metrics_helpers.py
R^2 helpers using scikit-learn for plot annotations.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def r2_prediction_vs_target(y_true, y_pred):
    """
    Coefficient of determination for direct prediction vs ground truth (same as sklearn r2_score).

    Args:
        y_true: np.ndarray
            Ground-truth values.
        y_pred: np.ndarray
            Predicted values.
        return: float
            R^2, or nan if undefined (too few points or zero target variance).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape[0] < 2:
        return float("nan")
    if np.var(y_true) < 1e-30:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def r2_linear_fit_x_to_y(x, y):
    """
    R^2 from ordinary least squares of y on x (single explanatory variable).

    Args:
        x: np.ndarray
            Independent variable (e.g. distance, curvature).
        y: np.ndarray
            Dependent variable (e.g. absolute error).
        return: float
            In-sample R^2 of the linear fit, or nan if undefined.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.shape[0] < 2:
        return float("nan")
    if np.var(x) < 1e-30:
        return float("nan")
    x_col = x.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x_col, y)
    y_hat = reg.predict(x_col)
    return float(r2_score(y, y_hat))


def format_r2_text(r2_value):
    """
    Args:
        r2_value: float
            R^2 or nan.
        return: str
            Short string for plot annotation.
    """
    if r2_value != r2_value:
        return r"$R^2$ = n/a"
    return rf"$R^2$ = {r2_value:.4f}"
