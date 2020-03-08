import torch
import numpy as np

class SNN(torch.nn.Module):
    def __init__(self, layers):
        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

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
