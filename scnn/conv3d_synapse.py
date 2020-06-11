import torch
import numpy as np

from .default_configs import *


class SpikingConv3DLayer(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, input_shape, input_channels, output_shape=None,
                 output_channels=1, kernel_size=3, dilation=1,
                 spike_fn=None, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD, recurrent=False,
                 lateral_connections=True,
                 eps=EPSILON, stride=(1, 1, 1), flatten_output=False):

        super(SpikingConv3DLayer, self).__init__()

        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.input_channels = input_channels
        self.input_shape = input_shape

        self.output_channels = output_channels
        self.output_shape = output_shape if output_shape is not None else input_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps

        self.flatten_output = flatten_output

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((output_channels, input_channels, *kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_channels, output_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        self.mem_rec_hist = None
        self.training = True

        # TODO : check this
        tau_mem = 10e-3
        tau_syn = 5e-3
        time_step = 1e-3
        self._alpha = float(np.exp(-time_step / tau_syn))
        self._beta = float(np.exp(-time_step / tau_mem))

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        res = [
            {'params': self.w},
            {'params': self.b},
            {'params': self.beta},
        ]

        if self.recurrent:
            res.append({'params': self.v})
        if lr is not None:
            for r in res:
                r['lr'] = lr
        if weight_decay is not None:
            res[0]['weight_decay'] = weight_decay
        return res

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[2]
        
        stride = tuple(self.stride)
        padding = tuple(np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int))
        conv_x = torch.nn.functional.conv3d(x, self.w, padding=padding,
                                            dilation=tuple(self.dilation),
                                            stride=stride)
        conv_x = conv_x[:, :, :, :self.output_shape[0], :self.output_shape[1]]

        mem = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x.dtype, device=x.device)
        syn = torch.zeros((batch_size, self.output_channels, *self.output_shape), device=x.device, dtype=x.dtype)  # FIXME
        spk = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x.dtype, device=x.device)

        spk_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype, device=x.device)
        mem_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype, device=x.device)

        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

        norm = (self.w ** 2).sum((1, 2, 3, 4))

        for t in range(nb_steps):
            # spike term
            # mthr = torch.einsum("abcd,b->abcd", mem, 1. / (norm + self.eps)) - b
            # spk = self.spike_fn(mthr)

            if self.lateral_connections:
                rst = torch.einsum("abcd,be ->aecd", spk, d)
            else:
                rst = torch.einsum("abcd,b,b->abcd", spk, self.b, norm)

            input_ = conv_x[:, :, t, :, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)

            # TODO : check to see if this is actually works
            new_syn = self._alpha * syn + input_
            new_mem = self._beta * mem + syn - rst
            mem = new_mem
            syn = new_syn
            # mem = (mem - rst) * self.beta + input_ * (1. - self.beta)

            mthr = torch.einsum("abcd,b->abcd", mem, 1. / (norm + self.eps)) - b
            spk = self.spike_fn(mthr)

            spk_rec[:, :, t, :, :] = spk
            mem_rec[:, :, t, :, :] = mem

        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
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
