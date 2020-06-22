# FIXME: refactor this file

import torch
import numpy as np

from scnn.default_configs import *


class SpikingConv2DStream(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, input_shape, input_channels, output_shape=None,
                 output_channels=1, kernel_size=3, dilation=1,
                 spike_fn=None, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD, recurrent=False,
                 lateral_connections=True,
                 eps=EPSILON, stride=(1, 1), flatten_output=False):

        super(SpikingConv2DStream, self).__init__()

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

        self.w = torch.nn.Parameter(torch.empty((output_channels, input_channels, *kernel_size)),
                                    requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_channels, output_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_channels), requires_grad=True)

        self.reset_parameters()
        self.clamp()
        self.training = True

        # Variables
        self.mem = None
        self.spk = None
        self.spk_rec_hist = None
        self.mem_rec_hist = None
        self.history_counter = 0

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

    def serialize(self):
        return {
            'type': 'conv2d_stream',
            'params': {
                'kernel_size': self.kernel_size,
                'dilation': self.dilation,
                'stride': self.stride,
                'output_channels': self.output_channels,

                'recurrent': self.recurrent,
                'lateral_connections': self.lateral_connections,

                'w_init_mean': self.w_init_mean,
                'w_init_std': self.w_init_std,
            }
        }

    def serialize_to_text(self):
        # FIXME: re-write this
        return 'C2.St(' + str(self.output_channels) \
               + ',k' + str(self.kernel_size[0]) \
               + (',l' if self.lateral_connections else '') \
               + (',r' if self.recurrent else '') \
               + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        if self.mem is None or self.spk is None:
            self.reset_mem(batch_size, x.device, x.dtype)

        stride = tuple(self.stride)
        padding = tuple(np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int))
        conv_x = torch.nn.functional.conv2d(x, self.w, padding=padding,
                                            dilation=tuple(self.dilation),
                                            stride=stride)
        conv_x = conv_x[:, :, :self.output_shape[0], :self.output_shape[1]]

        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

        norm = (self.w ** 2).sum((1, 2, 3))

        if self.lateral_connections:
            rst = torch.einsum("abcd,be ->aecd", self.spk, d)
        else:
            rst = torch.einsum("abcd,b,b->abcd", self.spk, self.b, norm)

        if self.recurrent:
            conv_x = conv_x + torch.einsum("abcd,be->aecd", self.spk, self.v)

        self.mem = (self.mem - rst) * self.beta + conv_x * (1. - self.beta)
        mthr = torch.einsum("abcd,b->abcd", self.mem, 1. / (norm + self.eps)) - b
        self.spk = self.spike_fn(mthr)

        self.spk_rec_hist[:, :, self.history_counter, :, :] = self.spk.detach().cpu()
        self.mem_rec_hist[:, :, self.history_counter, :, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= HISTOGRAM_MEMORY_SIZE:
            self.history_counter = 0

        if self.flatten_output:
            #             output = torch.transpose(self.spk, 1, 2).contiguous()
            output = self.spk.view(batch_size, self.output_channels * np.prod(self.output_shape))
        else:
            output = self.spk

        return output

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.mem = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x_dtype, device=x_device)
        self.spk = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x_dtype, device=x_device)

        self.spk_rec_hist = torch.zeros(
            (batch_size, self.output_channels, HISTOGRAM_MEMORY_SIZE, *self.output_shape), dtype=x_dtype)
        self.mem_rec_hist = torch.zeros(
            (batch_size, self.output_channels, HISTOGRAM_MEMORY_SIZE, *self.output_shape), dtype=x_dtype)

        self.history_counter = 0

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / (self.input_channels * np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)
