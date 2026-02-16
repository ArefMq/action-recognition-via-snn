import numpy as np
import torch
from torch import Tensor

from spikenet.constants import EPSILON
from spikenet.layers.spiking_base import SpikingNeuron


class SpikingDenseLayer(SpikingNeuron):
    """
    Spiking Dense Layer: This layer is used to create a dense layer of spiking neurons.

    Args:
        name (str): Name of the layer (default: "SpikingDenseLayer")
        in_features (int | None): Number of input features. None means defer specification to compile time.
        out_features (int | None): Number of output features. None means defer specification to compile time.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
        spike_fn (Callable): The spike function to use (default: SurrogateHeaviside.apply)
        time_reduction (Callable | None): The time reduction method to use (default: TimeReduction.NoTimeReduction)
        beta_init_std (float): Standard deviation of the normal distribution used to initialize the beta parameter.
        beta_init_mean (float): Mean of the normal distribution used to initialize the beta parameter.
        b_init_std (float): Standard deviation of the normal distribution used to initialize the b parameter.
        b_init_mean (float): Mean of the normal distribution used to initialize the b parameter.
        clamp_membrane (bool): Whether to clamp the membrane potential between 0 and 1 (default: True)
    """

    def __init__(self, out_features: int | None = None, **kwargs) -> None:
        super().__init__(out_features=out_features, name=kwargs.pop("name", "SpikingDenseLayer"), **kwargs)
        self.clamp_membrane = kwargs.get("clamp_membrane", True)

    def initialize_parameters(self) -> None:
        """Initializes the weights and parameters of the layer."""
        self.w = torch.nn.Parameter(torch.empty((self.in_features, self.out_features)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.out_features), requires_grad=True)

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / self.in_features),
        )
        torch.nn.init.normal_(self.beta, mean=self.beta_init_mean, std=self.beta_init_std)
        torch.nn.init.normal_(self.b, mean=self.b_init_mean, std=self.b_init_std)

    def spike_forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        batch_size, nb_steps = x.shape[0], x.shape[1]
        h = torch.einsum("abc,cd->abd", x, self.w)

        # membrane potential
        mem = torch.zeros((batch_size, self.out_features), dtype=x.dtype, device=x.device)
        spk = torch.zeros((batch_size, self.out_features), dtype=x.dtype, device=x.device)

        # output spikes recording
        mem_rec = torch.zeros((batch_size, nb_steps, self.out_features), dtype=x.dtype, device=x.device)
        spk_rec = torch.zeros((batch_size, nb_steps, self.out_features), dtype=x.dtype, device=x.device)

        for t in range(nb_steps):
            # reset term
            rst = spk * self.b * self.w_norm

            # input current
            input_ = h[:, t, :]

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            if self.clamp_membrane:
                mem = torch.clamp(mem, 0.0, 1.0)
            mem_rec[:, t, :] = mem

            # spike generation
            m_threshold = torch.einsum("ab,b->ab", mem, 1.0 / (self.w_norm + EPSILON)) - self.b
            spk = self.spike_fn(m_threshold)
            spk_rec[:, t, :] = spk

        return spk_rec, mem_rec
