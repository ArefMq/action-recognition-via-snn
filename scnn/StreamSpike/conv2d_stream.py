import torch
import numpy as np

from scnn.StreamSpike.conv1d_stream import SpikingConv1DStream
from scnn.default_configs import *


class SpikingConv2DStream(SpikingConv1DStream):
    def __init__(self, *args, **kwargs):
        super(SpikingConv2DStream, self).__init__(*args, **kwargs)

    def __str__(self):
        # FIXME: re-write this
        return 'C2.St(' + str(self.output_channels) \
               + ',k' + str(self.kernel_size[0]) \
               + (',l' if self.lateral_connections else '') \
               + (',r' if self.recurrent else '') \
               + ')'

    def forward_function(self, x):
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
        mthr = torch.einsum("abcd,b->abcd", self.mem, 1. / (norm + EPSILON)) - b
        self.spk = self.spike_fn(mthr)

        self.spk_rec_hist[:, :, self.history_counter, :, :] = self.spk.detach().cpu()
        self.mem_rec_hist[:, :, self.history_counter, :, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= self.histogram_memory_size:
            self.history_counter = 0

        if self.flatten_output:
            output = self.spk.view(batch_size, self.output_channels * np.prod(self.output_shape))
        else:
            output = self.spk

        return output
