from abc import ABC, abstractmethod

class TimeDiscretizer(ABC):
    pass

class SecondOrderCentralDiscretizer(TimeDiscretizer):
    def integrate_acceleration(self):
        pass
    # FIXME: compute everything (differentiator) from x, and then compute everything (integrator) from y
    def differentiate_position(self):
        pass

class NewmarkDiscretizer(TimeDiscretizer):
    def integrate_acceleration(self):
        pass

    def differentiate_position(self):
        pass
    pass
    # FIXME: compute everything (differentiator) from x, and then compute everything (integrator) from y