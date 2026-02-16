import numpy as np
import torch
from torch import Tensor

from spikenet.constants import EPSILON
from spikenet.layers.spiking_dense import SpikingDenseLayer
from spikenet.tools.window import window_to_and_array


class SpikingConv2D(SpikingDenseLayer):
    """
    Spiking Convolutional Layer: This layer is used to create a convolutional layer of spiking neurons.


    Args:
        name (str): Name of the layer (default: "Conv2D")
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
        stride (int | np.ndarray): The stride of the convolution operation (default: [1, 1, 1])
        padding (int | np.ndarray): The padding of the convolution operation (default: [0, 0, 0])
        dilation (int | np.ndarray): The dilation of the convolution operation (default: [1, 1, 1])
        kernel (int | np.ndarray): The size of the convolution kernel (default: [1, 3, 3])
    """

    def __init__(self, out_features: int | None = None, **kwargs) -> None:
        super().__init__(out_features=out_features, **kwargs)
        self.stride = window_to_and_array(kwargs.get("stride", (1, 1, 1)))
        self.padding = window_to_and_array(kwargs.get("padding", (0, 0, 0)))
        self.dilation = window_to_and_array(kwargs.get("dilation", (1, 1, 1)))
        self.kernel = window_to_and_array(kwargs.get("kernel", (1, 3, 3)))

    def initialize_parameters(self) -> None:
        """Initializes the weights and parameters of the layer."""
        self.w = torch.nn.Parameter(
            torch.empty((self.out_features, self.in_features, *self.kernel)),
            requires_grad=True,
        )
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.out_features), requires_grad=True)

        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / self.in_features),
        )
        torch.nn.init.normal_(self.beta, mean=self.beta_init_mean, std=self.beta_init_std)
        torch.nn.init.normal_(self.b, mean=self.b_init_mean, std=self.b_init_std)

    def _apply_conv(self, x: Tensor) -> Tensor:
        assert self.w is not None, "Parameters w are not initialized"
        return torch.nn.functional.conv3d(
            x,
            self.w,
            padding=tuple(self.padding),
            dilation=tuple(self.dilation),
            stride=tuple(self.stride),
        )

    def _crop_output(self, x: Tensor, out_shape: tuple[int, int]) -> Tensor:
        return x[
            :,  # batch_size
            :,  # out_channels
            :,  # time_frames
            self.padding[0] : self.padding[0] + out_shape[0],
            self.padding[1] : self.padding[1] + out_shape[1],
        ]

    def spike_forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        (batch_size, nb_in_channels, nb_steps, *_) = x.shape
        conv_x = self._apply_conv(x)
        out_shape = conv_x.shape[3:]

        # membrane potential
        mem = torch.zeros(
            (batch_size, self.out_features, *out_shape),
            dtype=x.dtype,
            device=x.device,
        )
        # output spikes
        spk = torch.zeros(
            (batch_size, self.out_features, *out_shape),
            dtype=x.dtype,
            device=x.device,
        )

        # output spikes recording
        mem_rec = torch.zeros(
            (batch_size, self.out_features, nb_steps, *out_shape),
            dtype=x.dtype,
            device=x.device,
        )
        spk_rec = torch.zeros(
            (batch_size, self.out_features, nb_steps, *out_shape),
            dtype=x.dtype,
            device=x.device,
        )

        flatten_b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *out_shape))
        norm = (self.w**2).sum((1, 2, 3, 4))

        for t in range(nb_steps):
            # reset term
            rst = torch.einsum("abcd,b,b->abcd", spk, self.b, norm)

            # input current
            input_ = conv_x[:, :, t, :, :]

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            mem_rec[:, :, t, :] = mem

            m_threshold = torch.einsum("abcd,b->abcd", mem, 1.0 / (norm + EPSILON)) - flatten_b
            spk = self.spike_fn(m_threshold)
            spk_rec[:, :, t, :, :] = spk

        return spk_rec, mem_rec
