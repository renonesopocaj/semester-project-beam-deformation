"""
metrics_base.py

Abstract Metric strategy interface and result containers.

A Metric receives a TrajectorySample and a signal name, and returns either a
CurveResult (x-values + y-values, for line plots) or a ScalarResult (a single
float per trajectory, for violin/bar plots).

All concrete metrics follow this interface; the plotter dispatches purely by
result kind without any external if/else.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class CurveResult:
    """
    Result of a curve metric.

    Attributes:
        x: np.ndarray  1-D, the x-axis values (e.g. hop distance, PC index, frequency)
        y: np.ndarray  1-D, same length as x, the y-axis values
        xlabel: str
        ylabel: str
    """
    x: np.ndarray
    y: np.ndarray
    xlabel: str
    ylabel: str


@dataclass
class ScalarResult:
    """
    Result of a scalar metric (one number per trajectory).

    Attributes:
        value: float
        label: str  short description for axis labels
    """
    value: float
    label: str


class Metric(ABC):
    """
    Strategy interface for a single locality or frequency metric.

    Sub-classes implement `compute(traj, signal_name)` which returns either a
    CurveResult or a ScalarResult.  The `kind` property tells the plotter which
    branch to use.
    """

    @property
    @abstractmethod
    def name(self):
        """
        return: str
            Short machine-friendly name (used for filenames).
        """

    @property
    @abstractmethod
    def kind(self):
        """
        return: str
            Either "curve" or "scalar".
        """

    @abstractmethod
    def compute(self, traj, signal_name):
        """
        Args:
            traj: TrajectorySample
            signal_name: str
                One of SIGNAL_NAMES.
        return: CurveResult or ScalarResult
        """


class CurveMetric(Metric):
    """Base class for metrics that return a CurveResult."""

    @property
    def kind(self):
        return "curve"


class ScalarMetric(Metric):
    """Base class for metrics that return a ScalarResult."""

    @property
    def kind(self):
        return "scalar"
