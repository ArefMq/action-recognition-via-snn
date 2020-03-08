import torch
import numpy as np


class SpikingDenseLayer(torch.nn.Module):
    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std, recurrent=False, lateral_connections=True, eps=1e-8):
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
        self.training = True

    def forward(self, x):
        batch_size = x.shape[0]

        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]

        # membrane potential 
        mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

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

            # membrane potential update
            mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b

            spk = self.spike_fn(mthr)

            spk_rec[:, t, :] = spk

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        loss = 0.5 * (spk_rec ** 2).mean()
        return spk_rec, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)

