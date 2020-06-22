import torch
import numpy as np

from scnn.default_configs import *


class ReadoutStream(torch.nn.Module):
    IS_CONV = False
    IS_SPIKING = False
    HAS_PARAM = True

    def __init__(self, input_shape, output_shape, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD, eps=EPSILON):
        super(ReadoutStream, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.eps = eps

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(0.7 * np.ones((1))), requires_grad=True)  # FIXME : REMOVE THIS
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        # variables
        self.mem = None
        self.mem_rec_hist = None
        self.history_counter = 0

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        res = [{'params': self.w}, {'params': self.b}, {'params': self.beta}]

        if lr is not None:
            for r in res:
                r['lr'] = lr
        if weight_decay is not None:
            res[0]['weight_decay'] = weight_decay
        return res

    def serialize(self):
        return {
            'type': 'readout_stream',
            'params': {
                'output_shape': self.output_shape,
                'w_init_mean': self.w_init_mean,
                'w_init_std': self.w_init_std,
            }
        }

    def serialize_to_text(self):
        return 'stream(' + str(self.output_shape) + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        if self.mem is None:
            self.reset_mem(batch_size, x.device, x.dtype)
        h = torch.einsum("ac,cd->ad", x, self.w)
        norm = (self.w ** 2).sum(0)

        output = None

        # membrane potential update
        self.mem = self.mem * self.beta + (1 - self.beta) * h[:, :]
        self.mem_rec_hist[:, self.history_counter, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= HISTOGRAM_MEMORY_SIZE:
            self.history_counter = 0

        output = self.mem / (norm + 1e-8) - self.b
        return output

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / np.prod(self.input_shape)))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.mem = torch.zeros((batch_size, self.output_shape), dtype=x_dtype, device=x_device)
        self.mem_rec_hist = torch.zeros((batch_size, HISTOGRAM_MEMORY_SIZE, self.output_shape), dtype=x_dtype)
        self.history_counter = 0

    def clamp(self):
        self.beta.data.clamp_(0., 1.)

