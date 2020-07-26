import torch
import numpy as np
import matplotlib.pyplot as plt


class ReadInLayer(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, input_shape, input_channels=1, output_shape=None, output_channels=None, flatten_output=False):
        super(ReadInLayer, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels

        self.output_shape = output_shape if output_shape is not None else input_shape
        self.output_channels = output_channels if output_channels is not None else input_channels

        self.flatten_output = flatten_output
        self.spk_rec_hist = Nones

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        return []

    def serialize(self):
        return {
            'type': 'readin',
            'params': {
                'input_shape': self.input_shape,
                'input_channels': self.input_channels
            }
        }

    def serialize_to_text(self):
        return 'I(' + str(self.input_channels) + 'x' + 'x'.join([str(i) for i in self.input_shape]) + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        if self.flatten_output:
            x = x.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
        else:
            x = x.view(batch_size, self.output_channels, nb_steps, *self.output_shape)

        self.spk_rec_hist = x.detach().cpu().numpy()
        return x

    def reset_parameters(self):
        pass

    def clamp(self):
        pass

    def draw(self, *kwargs):
        pass
