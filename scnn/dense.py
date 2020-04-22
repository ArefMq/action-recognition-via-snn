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

        # TODO : check this
        tau_mem = 10e-3
        tau_syn = 5e-3
        time_step = 1e-3
        self._alpha = float(np.exp(-time_step / tau_syn))
        self._beta = float(np.exp(-time_step / tau_mem))

    def get_trainable_parameters(self, lr):
        res = [
            {'params': self.w, 'lr': lr, "weight_decay": DEFAULT_WEIGHT_DECAY},
            {'params': self.b},
            {'params': self.beta},
        ]

        if self.recurrent:
            res.append({'params': self.v})
        return res

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        h = torch.einsum("abc,cd->abd", x, self.w)

        # membrane potential 
        mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        syn = torch.zeros((batch_size, self.output_shape), device=x.device, dtype=x.dtype)  # FIXME
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)
        self.mem_rec_hist = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)

        for t in range(nb_steps):
            # spike term
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b
            spk = self.spike_fn(mthr)

            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bc ->ac", spk, d)
            else:
                rst = spk * self.b * norm

            input_ = h[:, t, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            # membrane potential update
            # FIXME : check to see if it's better to train alpha and beta or not
            new_syn = self._alpha * syn + input_
            new_mem = self._beta * mem + syn - rst
            mem = new_mem
            syn = new_syn

            # mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            # TODO : check if it's better to do this in the next fram (like legacy) or not
            # mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b
            # spk = self.spike_fn(mthr)

            spk_rec[:, t, :] = spk
            self.mem_rec_hist[:, t, :] = mem

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = self.mem_rec_hist.detach().cpu().numpy()
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

