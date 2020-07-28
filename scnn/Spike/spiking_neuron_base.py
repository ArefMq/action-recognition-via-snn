import torch
import torch.nn as nn
import numpy as np

from scnn.default_configs import *


class SpikingNeuronBase(torch.nn.Module):
    IS_CONV = False
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, *args, **kwargs):
        super(SpikingNeuronBase, self).__init__()
        self.name = kwargs.get('name', None)
        dropout_prop = kwargs.get('dropout_prop', None)
        self.dropout = None if dropout_prop is None else nn.Dropout(dropout_prop)

        self.kernel_size = kwargs.get('kernel_size', None)
        self.dilation = kwargs.get('dilation', None)
        self.stride = kwargs.get('stride', None)
        self.input_channels = kwargs.get('input_channels', None)
        self.input_shape = kwargs.get('input_shape', None)

        self.output_channels = kwargs.get('output_channels', None)
        self.output_shape = kwargs.get('output_shape', None)
        self.spike_fn = kwargs.get('spike_fn', None)
        self.recurrent = kwargs.get('recurrent', False)
        self.lateral_connections = kwargs.get('lateral_connections', False)

        self.flatten_output = kwargs.get('flatten_output', False)
        self.w_init_mean = kwargs.get('w_init_mean', W_INIT_MEAN)
        self.w_init_std = kwargs.get('w_init_std', W_INIT_STD)

        if self.kernel_size is not None and not isinstance(self.kernel_size, np.ndarray):
            self.kernel_size = np.array(self.kernel_size)
        if self.dilation is not None and not isinstance(self.dilation, np.ndarray):
            self.dilation = np.array(self.dilation)
        if self.stride is not None and not isinstance(self.stride, np.ndarray):
            self.stride = np.array(self.stride)

        self.spk_rec_hist = None
        self.mem_rec_hist = None
        self.serializer_content = kwargs

    def forward_function(self, x):
        pass

    def trainable(self):
        return []

    def serialize(self):
        return {
            'type': type(self).__name__,
            'params': self.serializer_content
        }

    def __str__(self):
        pass

    def reset_parameters(self):
        pass

    def clamp(self):
        pass

    def draw(self, batch_id=0):
        pass

    # Functions
    def forward(self, x):
        res = self.forward_function(x)

        if self.dropout:
            return self.dropout(res)
        else:
            return res

    def reset_layer(self):
        self.reset_parameters()
        self.clamp()

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        res = [{'params': p} for p in self.trainable()]
        if not res:
            return res

        if lr is not None:
            for r in res:
                r['lr'] = lr
        if weight_decay is not None:
            res[0]['weight_decay'] = weight_decay
        return res
