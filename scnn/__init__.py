import torch

from .conv1d import SpikingConv1DLayer
from .conv2d import SpikingConv2DLayer
from .conv3d import SpikingConv3DLayer
from .pool2d import SpikingPool2DLayer
from .dense import SpikingDenseLayer
from .readout import ReadoutLayer
from .heaviside import SurrogateHeaviside


class SNN(torch.nn.Module):
    def __init__(self, spike_fn=None):
        super(SNN, self).__init__()
        self.layers = []
        self.default_spike_fn = spike_fn if spike_fn is not None else SurrogateHeaviside.apply
        self.last_layer_shape = None
        self.last_layer_is_conv = None

    def add_conv1d(self, **kwargs):
        self.add_layer(SpikingConv1DLayer, **kwargs)

    def add_conv2d(self, **kwargs):
        self.add_layer(SpikingConv2DLayer, **kwargs)

    def add_conv3d(self, **kwargs):
        self.add_layer(SpikingConv3DLayer, **kwargs)

    def add_pool2d(self, **kwargs):
        self.add_layer(SpikingPool2DLayer, **kwargs)

    def add_dense(self, **kwargs):
        self.add_layer(SpikingDenseLayer, **kwargs)

    def add_readout(self, **kwargs):
        self.add_layer(ReadoutLayer, **kwargs)

    def add_layer(self, layer, **kwargs):
        if layer.IS_SPIKING and self.default_spike_fn is not None and 'spike_fn' not in kwargs:
            kwargs['spike_fn'] = self.default_spike_fn

        if not layer.IS_CONV and self.last_layer_is_conv:
            self.layers[-1].flatten_output = True

        # calculating this layer inputs based on last layer
        if self.last_layer_shape is not None:
            llsh = self.last_layer_shape
            if layer.IS_CONV and 'input_channels' not in kwargs and 'input_shape' not in kwargs:
                if self.last_layer_is_conv:
                    kwargs['input_channels'] = llsh['channels']
                    kwargs['input_shape'] = llsh['shape']
                else:
                    raise NotImplementedError()  # FIXME : handle if last layer was not conv
            elif not layer.IS_CONV and 'input_shape' not in kwargs:
                if self.last_layer_is_conv:
                    input_shape = 1
                    for i in range(len(llsh['shape'])):
                        input_shape *= llsh['shape'][i]
                    input_shape *= llsh['channels']
                    kwargs['input_shape'] = input_shape
                else:
                    kwargs['input_shape'] = llsh['shape']

        if layer.IS_CONV:
            self.last_layer_shape = {'channels': kwargs['output_channels'], 'shape': kwargs['output_shape']}
        else:
            self.last_layer_shape = {'shape': kwargs['output_shape']}

        self.layers.append(layer(**kwargs))
        self.last_layer_is_conv = layer.IS_CONV

    def compile(self):
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        loss_seq = []
        for l in self.layers:
            x, loss = l(x)
            loss_seq.append(loss)
        return x, loss_seq

    def clamp(self):
        for l in self.layers:
            l.clamp()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
