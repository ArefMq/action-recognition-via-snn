import torch
import numpy as np

from .default_configs import *


class SpikingPool2DLayer(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = False

    def __init__(self, input_shape, input_channels, kernel_size=(2, 2), stride=None,
                 output_shape=None, output_channels=None, flatten_output=False):

        super(SpikingPool2DLayer, self).__init__()

        if output_channels is None:
            output_channels = input_channels
        if stride is None:
            stride = kernel_size
        if output_shape is None:
            output_shape = [1+(i-k)/s for i, k, s in zip(input_channels, kernel_size, stride)]

        self.kernel_size = np.array(kernel_size)
        self.stride = np.array(stride)
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.flatten_output = flatten_output
        self.spk_rec_hist = None

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[2]
        spk_rec = torch.zeros((batch_size, self.out_channels, *self.output_shape), dtype=x.dtype, device=x.device)

        for t in range(nb_steps):
            print('fix me please >>>>> ', x.shape)
            pool_x_t = torch.nn.functional.pool(x[t, :, :, :, :], kernel_size=tuple(self.kernel_size), stride=tuple(self.stride))
            print('and here >>>>> ', pool_x_t.shape)
            spk_rec[t, :, :, :, :] = pool_x_t

        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * np.prod(self.output_shape))
        else:
            output = spk_rec

        return output
