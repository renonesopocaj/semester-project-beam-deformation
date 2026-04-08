"""
time_discretizer.py

Contains time discretization (differentiation and integration) helper functions to obtain velocity, acceleration from
position and viceversa. Implements a strategy pattern.
"""

from abc import ABC, abstractmethod
import torch

def create_time_discretizer(time_discretizer_str):
    if time_discretizer_str == "second_order_central":
        return SecondOrderCentralDiscretizer()
    elif time_discretizer_str == "newmark":
        return NewmarkDiscretizer()
    else:
        raise ValueError(f"Unknown time discretizer: {time_discretizer_str}")

class TimeDiscretizer(ABC):

    @abstractmethod
    def integrate_acceleration(self, a, v_t, dt, x_t):
        """
        Integrate acceleration takes as input a single time step, different from differentiate position

        Args:
            a: torch.Tensor
                input prediction acceleration
            v_t: torch.Tensor
                previous velocity
            dt: float
                delta t
            x_t: torch.Tensor
                previous position
        :return: (x_predict_tp1, v_predict_tp1)

        """
        raise NotImplementedError

    @abstractmethod
    def differentiate_position(self, x, dt):
        """
        Differentiate position takes as input the whole tensors for all timesteps and returns the velocity and
        acceleration

        Args:
            x: torch.Tensor (T, N, 3)
            dt: float
        :return: (v, a)
            v: torch.Tensor (T, N, 3)
            a: torch.Tensor (T-1, N, 3)
        """
        raise NotImplementedError

    def reset(self):
        """Reset any internal state (e.g. for autoregressive rollout). Default no-op."""
        pass

class SecondOrderCentralDiscretizer(TimeDiscretizer):

    def integrate_acceleration(self, a, v_t, dt, x_t):
        v_predict_tp1 = v_t + a * dt
        x_predict_tp1 = x_t + v_predict_tp1 * dt
        return x_predict_tp1, v_predict_tp1

    def differentiate_position(self, x, dt):
        v0 = torch.zeros(*x[0].shape, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, N, 3)
        # first order backwards
        dx = x[1:] - x[:-1]  # (T-1, N, 3) computes dx by shifting by 1, first excluding t=0 then excluding t=last
        v_steps = dx / dt  # (T-1, N, 3)
        v = torch.cat([v0, v_steps], dim=0)  # (T, N, 3)
        # NOTE: but why isn't it brought back to (T, N, 3)? Because then anyway I save single timesteps so no need to do
        # it now. Note also this is a forward difference wrt v --> second order centered difference wrt x
        dv = v[1:] - v[:-1]  # same as velocity  (T-1, N, 3)
        a = dv / dt
        return v, a

class NewmarkDiscretizer(TimeDiscretizer):
    """
    Newmark average-acceleration (trapezoidal) time discretizer matching
    the generalized-alpha scheme in beam_fea_solver.py with alpha_m=0, alpha_f=0.

    Derived constants (fixed, not configurable):
        gamma = 0.5 + alpha_f - alpha_m = 0.5
        beta  = (gamma + 0.5)^2 / 4     = 0.25

    With these values the general Newmark formulas simplify:
        - 1 / (beta * dt^2) = 4 / dt^2
        - (1 - 2*beta) / (2*beta) = 1  -->  a_old coefficient becomes 1
        - (0.5 - beta) = beta = 0.25   -->  symmetric weight on a_old, a_new
        - (1 - gamma)  = gamma = 0.5   -->  symmetric weight on a_old, a_new

    Simplified update equations (all still depend on a_old):
        a_new   = 4 * (x_new - x_old - dt * v_old) / dt^2 - a_old
        v_new   = v_old + 0.5 * dt * (a_old + a_new)
        x_new   = x_old + dt * v_old + 0.25 * dt^2 * (a_old + a_new)

    For integrate_acceleration (rollout), internal state _a_prev tracks the
    previous acceleration since the displacement update couples a[t] and a[t+1].

    Call reset() before starting a new rollout sequence.
    """

    # Fixed Newmark constants derived from alpha_m=0, alpha_f=0.
    BETA = 0.25
    GAMMA = 0.5

    def __init__(self):
        self._a_prev = None

    def reset(self):
        """Reset internal state before a new rollout."""
        self._a_prev = None

    def integrate_acceleration(self, a_new, v_t, dt, x_t):
        """
        Newmark forward integration (single step).

        Computes position and velocity at t+1 given predicted acceleration a_new
        at t+1, and (x_t, v_t) at time t.

        Uses internal _a_prev (acceleration at time t) for the coupled update.
        On first call (or after reset), _a_prev defaults to zero.

        Simplified from beam_fea_solver.py (beta=0.25, gamma=0.5):
            x_{t+1} = x_t + dt*v_t + 0.25*dt^2*(a_old + a_new)
            v_{t+1} = v_t + 0.5*dt*(a_old + a_new)

        Args:
            a_new: torch.Tensor (N, 3)
                Predicted acceleration at t+1 (model output).
            v_t: torch.Tensor (N, 3)
                Velocity at time t.
            dt: float
                Time step size.
            x_t: torch.Tensor (N, 3)
                Position at time t.
        Returns:
            (x_tp1, v_tp1): tuple of torch.Tensor
        """
        a_old = self._a_prev if self._a_prev is not None else torch.zeros_like(a_new)
        a_sum = a_old + a_new
        x_tp1 = x_t + dt * v_t + self.BETA * dt ** 2 * a_sum
        v_tp1 = v_t + self.GAMMA * dt * a_sum
        self._a_prev = a_new.clone()
        return x_tp1, v_tp1

    def differentiate_position(self, x, dt):
        """
        Recover velocity and acceleration from a position trajectory using
        the Newmark inverse (same as beam_fea_solver.update_a / update_v).

        Starting from rest (v[0]=0, a[0]=0), iteratively applies the simplified
        formulas (beta=0.25, gamma=0.5):
            a[t+1] = 4 * (x[t+1] - x[t] - dt*v[t]) / dt^2 - a[t]
            v[t+1] = v[t] + 0.5 * dt * (a[t] + a[t+1])

        Returns acceleration array of shape (T-1, N, 3) where a[t] is the
        Newmark acceleration computed at step t -> t+1 (stored as a_track[t+1]).
        This is the target the GNN learns to predict at time step t.

        Args:
            x: torch.Tensor (T, N, 3)
                Position trajectory.
            dt: float
                Time step size.
        Returns:
            v: torch.Tensor (T, N, 3)
                Velocity at each time step.
            a: torch.Tensor (T-1, N, 3)
                Acceleration targets for each time step.
        """
        T = x.shape[0]
        dt2 = dt ** 2
        v = torch.zeros_like(x) # (T, N, 3)
        a_track = torch.zeros_like(x) # (T, N, 3), a_track[0] = 0 (rest)
        for t in range(T - 1):
            # Newmark inverse with beta=0.25: (1-2*beta)/(2*beta) = 1
            a_new = 4.0 * (x[t + 1] - x[t] - dt * v[t]) / dt2 - a_track[t]
            # Newmark velocity with gamma=0.5: average of a_old and a_new
            v_new = v[t] + self.GAMMA * dt * (a_track[t] + a_new)
            a_track[t + 1] = a_new
            v[t + 1] = v_new
        # a_track[t+1] is the acceleration computed at step t→t+1.
        # Return a_track[1:] (T-1 entries) so that a[t] is the target for timestep t.
        return v, a_track[1:]