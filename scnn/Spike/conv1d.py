import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scnn.default_configs import *

from scnn.Spike.spiking_neuron_base import SpikingNeuronBase


class SpikingConv1DLayer(SpikingNeuronBase):
    IS_CONV = True
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, *args, **kwargs):
        if 'output_shape' not in kwargs:
            kwargs['output_shape'] = kwargs['input_shape']

        if 'stride' not in kwargs:
            kwargs['stride'] = 1

        if 'kernel_size' not in kwargs:
            kwargs['kernel_size'] = 3

        if 'dilation' not in kwargs:
            kwargs['dilation'] = 1

        super(SpikingConv1DLayer, self).__init__(*args, **kwargs)

        self.w = torch.nn.Parameter(torch.empty((self.output_channels, self.input_channels, *self.kernel_size)), requires_grad=True)
        if self.recurrent:
            self.v = torch.nn.Parameter(torch.empty((self.output_channels, self.output_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.output_channels), requires_grad=True)

    def trainable(self):
        if self.recurrent:
            return [self.w, self.b, self.beta, self.v]
        else:
            return [self.w, self.b, self.beta]

    def forward_function(self, x):
        # TODO : refactor this
        batch_size = x.shape[0]

        conv_x = torch.nn.functional.conv1d(x, self.w, padding=(
        np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int),),
                                            dilation=(self.dilation,),
                                            stride=(self.stride,))
        nb_steps = conv_x.shape[2]

        # membrane potential
        mem = torch.zeros((batch_size, self.out_channels), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps), dtype=x.dtype, device=x.device)

        if self.lateral_connections:
            d = torch.einsum("abc, ebc -> ae", self.w, self.w)
        b = self.b

        norm = (self.w ** 2).sum((1, 2))

        for t in range(nb_steps):

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bd ->ad", spk, d)
            else:
                rst = torch.einsum("ab,b,b->ab", spk, self.b, norm)

            input_ = conv_x[:, :, t]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bd->ad", spk, self.v)

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + EPSILON)) - b

            spk = self.spike_fn(mthr)

            spk_rec[:, :, t] = spk

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / (self.input_channels * np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean,
                                  std=self.w_init_std * np.sqrt(1. / self.output_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)

