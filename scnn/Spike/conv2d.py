import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scnn.Spike.conv1d import SpikingConv1DLayer
from scnn.default_configs import *


class SpikingConv2DLayer(SpikingConv1DLayer):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'Conv2D'
        if 'output_shape' not in kwargs:
            kwargs['output_shape'] = kwargs['input_shape']

        if 'stride' not in kwargs:
            kwargs['stride'] = (1, 1)

        if 'kernel_size' not in kwargs:
            kwargs['kernel_size'] = (3, 3)

        if 'dilation' not in kwargs:
            kwargs['dilation'] = (1, 1)

        super(SpikingConv2DLayer, self).__init__(*args, **kwargs)

    def forward_function(self, x):
        # TODO: refactor this
        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(
            np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int)),
                                            dilation=tuple(self.dilation),
                                            stride=tuple(self.stride))
        conv_x = conv_x[:, :, :, :self.output_shape]
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels, self.output_shape), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, self.output_shape), dtype=x.dtype,
                              device=x.device)

        if self.lateral_connections:
            d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)
        b = self.b.unsqueeze(1).repeat((1, self.output_shape))

        norm = (self.w ** 2).sum((1, 2, 3))

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abc,bd ->adc", spk, d)
            else:
                rst = torch.einsum("abc,b,b->abc", spk, self.b, norm)

            input_ = conv_x[:, :, t, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("abc,bd->adc", spk, self.v)

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            mthr = torch.einsum("abc,b->abc", mem, 1. / (norm + EPSILON)) - b

            spk = self.spike_fn(mthr)

            spk_rec[:, :, t, :] = spk

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        loss = 0.5 * (spk_rec ** 2).mean()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels * self.output_shape)
        else:
            output = spk_rec
        return output#, loss
