import torch
import numpy as np

from scnn.network import SpikingNeuralNetworkBase


class SpikingNeuralNetwork(SpikingNeuralNetworkBase):
    def __init__(self, *args, **kwargs):
        super(SpikingNeuralNetwork, self).__init__(*args, **kwargs)

    def batch_step(self, loss_func, xb, yb, opt=None):
        log_softmax_fn = torch.nn.LogSoftmax(dim=1)  # TODO: investigate this
        yb = torch.from_numpy(yb.astype(np.long)).to(self.device)

        y_pred = self.predict(xb)
        log_y_pred = log_softmax_fn(y_pred)
        loss = loss_func(log_y_pred, yb)

        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), 5)  # TODO: investigate this
            opt.step()
            self.clamp()  # TODO: investigate this
            opt.zero_grad()

        for l in self.layers:
            if 'mem' in l.__dict__:
                l.mem.detach_()
            if 'spk' in l.__dict__:
                l.spk.detach_()

        return loss.item(), len(xb)

    def compute_classification_accuracy(self, data_dl, calc_map=True):
        accs = []
        nb_outputs = self.layers[-1].output_shape
        heatmap = np.zeros((nb_outputs, nb_outputs))
        with torch.no_grad():
            for x_batch, y_batch in data_dl:
                output = self.predict(x_batch)
                _, am = torch.max(output, 1)  # argmax over output units
                y_batch = torch.from_numpy(y_batch.astype(np.long)).to(self.device)
                tmp = np.mean((y_batch == am).detach().cpu().numpy())  # compare to labels
                accs.append(tmp)

                if calc_map:
                    for i in range(y_batch.shape[0]):
                        heatmap[y_batch[i], am[i]] += 1
        if calc_map:
            return np.mean(accs), heatmap
        else:
            return np.mean(accs)

    def reset_mem(self, batch_size=None):
        if batch_size is None:
            batch_size = self.layers[1].mem.shape[0]
        for l in self.layers:
            l.reset_mem(batch_size, self.device, self.dtype)
