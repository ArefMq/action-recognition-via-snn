import torch
import numpy as np

from scnn.default_configs import *


class SpikingDenseStream(torch.nn.Module):
    IS_CONV = False
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD,
                 recurrent=False, lateral_connections=True, eps=EPSILON):
        super(SpikingDenseStream, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_shape, output_shape)), requires_grad=True)

        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

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
            'type': 'dense_stream',
            'params': {
                'output_shape': self.output_shape,
                'recurrent': self.recurrent,
                'lateral_connections': self.lateral_connections,

                'w_init_mean': self.w_init_mean,
                'w_init_std': self.w_init_std,
            }
        }

    def serialize_to_text(self):
        return 'D.St(' + str(self.output_shape) + ('r' if self.recurrent else '') + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        if self.mem is None:
            self.reset_mem(batch_size, x.device, x.dtype)

        h = torch.einsum("ac,cd->ad", x, self.w)

        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)

        # reset term
        if self.lateral_connections:
            rst = torch.einsum("ab,bc ->ac", self.spk, d)
        else:
            rst = self.spk * self.b * norm

        if self.recurrent:
            h = h + torch.einsum("ab,bc->ac", self.spk, self.v)

        self.mem = (self.mem - rst) * self.beta + h * (1. - self.beta)
        mthr = torch.einsum("ab,b->ab", self.mem, 1. / (norm + self.eps)) - self.b
        self.spk = self.spike_fn(mthr)

        self.spk_rec_hist[:, self.history_counter, :] = self.spk.detach().cpu()
        self.mem_rec_hist[:, self.history_counter, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= HISTOGRAM_MEMORY_SIZE:
            self.history_counter = 0

        return self.spk

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean,
                                  std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.mem = torch.zeros((batch_size, self.output_shape), dtype=x_dtype, device=x_device)
        self.spk = torch.zeros((batch_size, self.output_shape), dtype=x_dtype, device=x_device)
        self.spk_rec_hist = torch.zeros((batch_size, HISTOGRAM_MEMORY_SIZE, self.output_shape), dtype=x_dtype)
        self.mem_rec_hist = torch.zeros((batch_size, HISTOGRAM_MEMORY_SIZE, self.output_shape), dtype=x_dtype)
        self.history_counter = 0

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


