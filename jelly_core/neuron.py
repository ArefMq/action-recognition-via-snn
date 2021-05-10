from abc import abstractmethod
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate


class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.monitor = monitor_state
        self.reset()
        self.unparallelizable = True


    @abstractmethod
    def neuronal_charge(self, dv: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        if self.monitor:
            if self.monitor['h'].__len__() == 0:
                # 补充在0时刻的电压
                if self.v_reset is None:
                    self.monitor['h'].append(self.v.data.cpu().numpy().copy() * 0)
                else:
                    self.monitor['h'].append(self.v.data.cpu().numpy().copy() * self.v_reset)

        self.spike = self.surrogate_function(self.v - self.v_threshold)
        if self.monitor:
            self.monitor['s'].append(self.spike.data.cpu().numpy().copy())

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1 - spike) * self.v + spike * self.v_reset

        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())


    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def set_monitor(self, monitor_state=True):
        if monitor_state:
            self.monitor = {'h': [], 'v': [], 's': []}
        else:
            self.monitor = False


    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

    def reset(self):
        if self.v_reset is None:
            self.v = 0.0
        else:
            self.v = self.v_reset

        self.spike = None

        if self.monitor:
            self.monitor = {'h': [], 'v': [], 's': []}


class IFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False, monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)

    def neuronal_charge(self, dv: torch.Tensor):
        self.v += dv


class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.Sigmoid(), detach_reset=False,
                 monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        self.tau = tau

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau}'

    def neuronal_charge(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) / self.tau
        else:
            self.v += (dv - (self.v - self.v_reset)) / self.tau
