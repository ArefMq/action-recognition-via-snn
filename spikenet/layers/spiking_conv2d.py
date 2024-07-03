import numpy as np
import torch
from spikenet.layers.spiking_dense import SpikingDenseLayer
from spikenet.tools.configs import EPSILON


class SpikingConv2D(SpikingDenseLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stride: np.array = kwargs.get("stride", np.array((1, 1, 1)))
        self.padding: np.array = kwargs.get("padding", np.array((0, 0, 0)))
        self.dilation: np.array = kwargs.get("dilation", np.array((1, 1, 1)))

        kernel_size: np.array = kwargs.get("kernel_size", np.array((1, 3, 3)))
        if isinstance(kernel_size, int) or kernel_size.size == 1:
            kernel_size = np.array((1, kernel_size, kernel_size))
        self.kernel_size = kernel_size

    @property
    def is_conv(self) -> bool:
        return True

    def initialize_parameters(self) -> None:
        self.w = torch.nn.Parameter(
            torch.empty((self.out_features, self.in_features, *self.kernel_size)),
            requires_grad=True,
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

    def _apply_conv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv3d(
            x,
            self.w,
            padding=tuple(
                np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int)
            ),
            dilation=tuple(self.dilation),
            stride=tuple(self.stride),
        )

    def _crop_output(self, x: torch.Tensor, out_shape: tuple[int, int]) -> torch.Tensor:
        return x[
            :,  # batch_size
            :,  # out_channels
            :,  # time_frames
            self.padding[0] : self.padding[0] + out_shape[0],
            self.padding[1] : self.padding[1] + out_shape[1],
        ]

    def spike_forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIXME: fix this for all layers (make it shorter)
        (batch_size, nb_in_channels, nb_steps, *out_shape) = x.shape
        conv_x = self._apply_conv(x)
        conv_x = self._crop_output(conv_x, out_shape)

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

            mthr = torch.einsum("abcd,b->abcd", mem, 1.0 / (norm + EPSILON)) - flatten_b
            spk = self.spike_fn(mthr)
            spk_rec[:, :, t, :, :] = spk

        return spk_rec, mem_rec

    def details(self) -> str:
        txt = super().details()
        ks = "x".join(map(str, self.kernel_size))
        return f"conv_{txt} {ks=}"
    
    def plot_mem(*args, **kwargs):
        pass
    def plot_spk(*args, **kwargs):
        pass
        

