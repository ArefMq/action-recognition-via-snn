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
            torch.empty((self.input_dim, self.output_dim)), requires_grad=True
        )
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.output_dim), requires_grad=True)

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / self.input_dim),
        )
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)

    def spike_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, nb_steps = x.shape[0], x.shape[1]
        h = torch.einsum("abc,cd->abd", x, self.w)

        # membrane potential
        mem = torch.zeros((batch_size, self.output_dim), dtype=x.dtype, device=x.device)
        spk = torch.zeros((batch_size, self.output_dim), dtype=x.dtype, device=x.device)

        # output spikes recording
        mem_rec = torch.zeros((batch_size, nb_steps, self.output_dim), dtype=x.dtype, device=x.device)
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_dim), dtype=x.dtype, device=x.device)

        for t in range(nb_steps):
            # reset term
            rst = spk * self.b * self.w_norm

            # input current
            input_ = h[:, t, :]

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            if self.mem_clamp:
                mem = torch.clamp(mem, 0.0, 1.0)
            mem_rec[:, t, :] = mem

            # spike generation
            mthr = torch.einsum("ab,b->ab", mem, 1.0 / (self.w_norm + EPSILON)) - self.b
            spk = self.spike_fn(mthr)
            spk_rec[:, t, :] = spk

        return spk_rec, mem_rec
