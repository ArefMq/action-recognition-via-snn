import torch
import numpy as np

from .default_configs import *


class SpikingDenseLayer(torch.nn.Module):
    IS_CONV = False
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD,
                 recurrent=False, lateral_connections=True, eps=EPSILON):
        super(SpikingDenseLayer, self).__init__()

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
        self.spk_rec_hist = None
        self.mem_rec_hist = None
        self.training = True

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
            'type': 'dense',
            'params': {
                'output_shape': self.output_shape,
                'recurrent': self.recurrent,
                'lateral_connections': self.lateral_connections,

                'w_init_mean': self.w_init_mean,
                'w_init_std': self.w_init_std,
            }
        }

    def serialize_to_text(self):
        return 'D(' + str(self.output_shape) + ('r' if self.recurrent else '') + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        h = torch.einsum("abc,cd->abd", x, self.w)

        # membrane potential
        mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)
        self.mem_rec_hist = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype)

        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)

        for t in range(nb_steps):
            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bc ->ac", spk, d)
            else:
                rst = spk * self.b * norm

            input_ = h[:, t, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b
            spk = self.spike_fn(mthr)

            spk_rec[:, t, :] = spk
            self.mem_rec_hist[:, t, :] = mem.detach().cpu()

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = self.mem_rec_hist.numpy()
        return spk_rec

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)

