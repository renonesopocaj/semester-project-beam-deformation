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
        :return: (v_predict_tp1, x_predict_tp1)

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

    def integrate_acceleration(self, a, v_t, dt, x_t):
        pass

    def differentiate_position(self, x, dt):
        pass