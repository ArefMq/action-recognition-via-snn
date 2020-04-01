import torch


class SNN(torch.nn.Module):
    def __init__(self, spike_fn=None):
        super(SNN, self).__init__()
        self.layers = []
        self.default_spike_fn = spike_fn
        self.last_layer_shape = None
        self.last_layer_is_conv = None

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
