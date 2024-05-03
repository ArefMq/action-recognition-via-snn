from enum import Enum
import numpy as np
import torch
from spikenet.layers.spiking_base import SpikingNeuron
from spikenet.tools.configs import EPSILON


class SpikingDenseLayer(SpikingNeuron):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mem_clamp = kwargs.get("mem_clamp", True)

    def initialize_parameters(self) -> None:
        self.w = torch.nn.Parameter(
            torch.empty((self.in_features, self.out_features)), requires_grad=True
        )
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.out_features), requires_grad=True)

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / self.in_features),
        )
        torch.nn.init.normal_(
            self.beta, mean=self.beta_init_mean, std=self.beta_init_std
        )
        torch.nn.init.normal_(self.b, mean=self.b_init_mean, std=self.b_init_std)

    def spike_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, nb_steps = x.shape[0], x.shape[1]
        h = torch.einsum("abc,cd->abd", x, self.w)

        # membrane potential
        mem = torch.zeros(
            (batch_size, self.out_features), dtype=x.dtype, device=x.device
        )
        spk = torch.zeros(
            (batch_size, self.out_features), dtype=x.dtype, device=x.device
        )

        # output spikes recording
        mem_rec = torch.zeros(
            (batch_size, nb_steps, self.out_features), dtype=x.dtype, device=x.device
        )
        spk_rec = torch.zeros(
            (batch_size, nb_steps, self.out_features), dtype=x.dtype, device=x.device
        )

        for t in range(nb_steps):
            # reset term
            rst = spk * self.b * self.w_norm

            # input current
            input_ = h[:, t, :]

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            # if self.mem_clamp:
            #     mem = torch.clamp(mem, 0.0, 1.0)
            mem_rec[:, t, :] = mem

            # spike generation
            mthr = torch.einsum("ab,b->ab", mem, 1.0 / (self.w_norm + EPSILON)) - self.b
            spk = self.spike_fn(mthr)
            spk_rec[:, t, :] = spk

        return spk_rec, mem_rec

    def plot_mem(self, ax=None, batch_id: int = 0) -> None:
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()
        mem = self.mem_rec[batch_id].detach().cpu().numpy()
        ax.plot(mem)
        ax.set_title("Membrane potential")
        ax.set_xlabel("Time")
        ax.set_ylabel("Membrane potential")

    def plot_spk(self, ax=None, batch_id: int = 0) -> None:
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()
        spk = self.spike_rec[batch_id].detach().cpu().numpy()
        if len(spk.shape) == 1:
            ax.bar(range(len(spk)), spk, color="black")
            ax.set_title("Output spikes")
            ax.set_xlabel("Output Neurons")
            ax.set_ylabel(f"{self.time_reduction_method}")
        else:
            ax.imshow(spk.T, aspect="auto", cmap="gray_r")
            ax.set_title("Output spikes (spike = white)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Output spikes")
