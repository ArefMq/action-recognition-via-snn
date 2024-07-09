import torch
import numpy as np

from spikenet.functions import PollingReduction
from spikenet.layers.spiking_base import SpikingNeuron


class SpikingPoolingLayer(SpikingNeuron):
    """
    Spiking Pooling Layer: This layer is used to create a pooling layer of spiking neurons.

    Args:
        name (str): Name of the layer (default: <id>_<neuron_type>)
        in_features (int): Number of input features. None for getting the value from previous layer or input tensor.
        out_features (int): Number of output features.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
        spike_fn (Callable): The spike function to use (default: SurrogateHeaviside.apply)
        time_reduction (str or TimeReduction): The time reduction method to use (default: TimeReduction.NoTimeReduction)
        beta_init_std (float): Standard deviation of the normal distribution used to initialize the beta parameter.
        beta_init_mean (float): Mean of the normal distribution used to initialize the beta parameter.
        b_init_std (float): Standard deviation of the normal distribution used to initialize the b parameter.
        b_init_mean (float): Mean of the normal distribution used to initialize the b parameter.
        stride (int or np.ndarray): The stride of the pooling operation (default: np.array((1, 2, 2)))
        kernel_size (int or np.ndarray): The size of the pooling kernel (default: np.array((1, 2, 2)))
        reduction (PollingReduction): The reduction method to use (default: PollingReduction.MaxSpikeRate)

    WARNING: Not to confuse reduction with time_reduction!
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stride: np.array = kwargs.get("stride", np.array((1, 2, 2)))
        kernel_size = kwargs.get("kernel_size", np.array((1, 2, 2)))
        if isinstance(kernel_size, int) or kernel_size.size == 1:
            kernel_size = np.array((1, kernel_size, kernel_size))
        self.kernel_size: np.ndarray = kernel_size
        self.reduction: PollingReduction = kwargs.get(
            "reduction", PollingReduction.MaxSpikeRate
        )

    @property
    def is_conv(self) -> bool:
        return True

    def initialize_parameters(self) -> None:
        pass

    def clamp(self) -> None:
        pass

    def spike_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.w is not None, "Parameters w are not initialized"
        assert self.beta is not None, "Parameters beta are not initialized"
        assert self.b is not None, "Parameters b are not initialized"

        (batch_size, nb_in_channels, nb_steps, *in_shape) = x.shape
        out_shape = (
            (np.array((nb_steps, *in_shape)) - self.kernel_size) // self.stride + 1
        ).astype(int)

        spk_rec = torch.zeros(
            (batch_size, self.out_features, nb_steps, *out_shape),
            dtype=x.dtype,
            device=x.device,
        )

        if self.reduction == PollingReduction.MaxSpikeRate:
            spk_rec = torch.nn.functional.max_pool3d(
                x,
                kernel_size=tuple(self.kernel_size),
                stride=tuple(self.stride),
            )
        elif self.reduction == PollingReduction.AvgSpikeRate:
            spk_rec = torch.nn.functional.avg_pool3d(
                x,
                kernel_size=tuple(self.kernel_size),
                stride=tuple(self.stride),
            )
        elif self.reduction == PollingReduction.SpikeTime:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return spk_rec, None

    def details(self) -> str:
        txt = super().details()
        ks = "x".join(map(str, self.kernel_size))
        return f"pooling_{self.reduction.name}_{txt} {ks=}"

    def plot_mem(*args, **kwargs) -> None:
        pass

    def plot_spk(*args, **kwargs) -> None:
        pass
