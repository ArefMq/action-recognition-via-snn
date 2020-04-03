import torch
import numpy as np

from .default_configs import *


class SpikingConv1DLayer(torch.nn.Module):
    IS_CONV=True
    IS_SPIKING=True

    def __init__(self, input_channels, output_channels, kernel_size, dilation,
                 spike_fn, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD, recurrent=False,
                 lateral_connections=True,
                 eps=EPSILON, stride=1, flatten_output=False):

        super(SpikingConv1DLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = input_channels
        self.out_channels = output_channels

        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((output_channels, input_channels, kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_channels, output_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):
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
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - b

            spk = self.spike_fn(mthr)

            spk_rec[:, :, t] = spk

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        loss = 0.5 * (spk_rec ** 2).mean()

        if self.flatten_output:

            output = torch.transpose(spk_rec, 1, 2).contiguous()
        else:
            output = spk_rec

        return output, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / (self.in_channels * np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.out_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


