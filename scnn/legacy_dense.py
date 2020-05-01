import torch
import numpy as np
from .default_configs import *


class LegacyDense(torch.nn.Module):
    IS_CONV = False
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD,
                 recurrent=False, lateral_connections=True, eps=EPSILON):
        super(LegacyDense, self).__init__()

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

        tau_mem = 10e-3
        tau_syn = 5e-3
        time_step = 1e-3

        self._alpha = float(np.exp(-time_step / tau_syn))
        self._beta = float(np.exp(-time_step / tau_mem))

        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()
        self.spk_rec_hist = None
        self.mem_rec_hist = None
        self.training = True

    def get_trainable_parameters(self, lr):
        res = [
            {'params': self.w, 'lr': lr, "weight_decay": DEFAULT_WEIGHT_DECAY},
            {'params': self.b, 'lr': lr},
        ]

        if self.recurrent:
            res.append({'params': self.v, 'lr': lr})
        return res

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        h1 = torch.einsum("abc,cd->abd", (x, self.w))
        syn = torch.zeros((batch_size, self.output_shape), device=x.device, dtype=x.dtype)
        mem = torch.zeros((batch_size, self.output_shape), device=x.device, dtype=x.dtype)

        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)
        mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

        # Compute hidden layer activity
        for t in range(nb_steps):
            mthr = mem - 1.0
            out = self.spike_fn(mthr)
            rst = torch.zeros_like(mem)
            c = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]

            new_syn = self._alpha * syn + h1[:, t]
            new_mem = self._beta * mem + syn - rst

            mem = new_mem
            syn = new_syn

            mem_rec[:, t, :] = mem
            spk_rec[:, t, :] = out

        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()
        return spk_rec

    #         for t in range(nb_steps):
    #             # reset term
    #             if self.lateral_connections:
    #                 rst = torch.einsum("ab,bc ->ac", spk, d)
    #             else:
    #                 rst = spk * self.b * norm

    #             input_ = h[:, t, :]
    #             if self.recurrent:
    #                 input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

    #             # membrane potential update
    #             mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
    #             mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b

    #             print('mthr', mthr)
    #             spk = self.spike_fn(mthr)
    #             print('spk', spk)

    #             spk_rec[:, t, :] = spk
    #             self.mem_rec_hist[:, t, :] = mem

    #             # save spk_rec for plotting
    #         self.spk_rec_hist = spk_rec.detach().cpu().numpy()
    #         self.mem_rec_hist = self.mem_rec_hist.detach().cpu().numpy()
    #         loss = 0.5 * (spk_rec ** 2).mean()
    #         return spk_rec, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))

    def clamp(self):
        pass

