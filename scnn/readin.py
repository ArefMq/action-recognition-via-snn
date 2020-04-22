import torch
import numpy as np

from .default_configs import *


class ReadoutLayer(torch.nn.Module):
    IS_CONV = False
    IS_SPIKING = False
    HAS_PARAM = True

    def __init__(self, input_shape, output_shape, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD, eps=EPSILON, time_reduction="mean"):

        assert time_reduction in ["mean", "max"], 'time_reduction should be "mean" or "max"'

        super(ReadoutLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.eps = eps
        self.time_reduction = time_reduction

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
#         self.w = torch.nn.Parameter(torch.empty((10, 20)), requires_grad=True)
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.tensor(0.7 * np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.mem_rec_hist = None

    def get_trainable_parameters(self):
        res = [
            {'params': self.w},  #, 'lr': lr, "weight_decay": DEFAULT_WEIGHT_DECAY}
            {'params': self.b},
        ]

        if self.time_reduction == "max":
            res.append({'params': self.beta})
        return res

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.einsum("abc,cd->abd", x, self.w)
        norm = (self.w ** 2).sum(0)

        output = None
        if self.time_reduction == "max":
            nb_steps = x.shape[1]

            mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
            mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

            for t in range(nb_steps):
                # membrane potential update
                mem = mem * self.beta + (1 - self.beta) * h[:, t, :]
                mem_rec[:, t, :] = mem

            output = torch.max(mem_rec, 1)[0] / (norm + 1e-8) - self.b

        elif self.time_reduction == "mean":
            mem_rec = h
            output = torch.mean(mem_rec, 1) / (norm + 1e-8) - self.b

        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()

        loss = None
        return output#, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / np.prod(self.input_shape)))
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        if self.time_reduction == "max":
            self.beta.data.clamp_(0., 1.)

