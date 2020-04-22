import torch
import numpy as np

from time import time
from os import path

from .conv1d import SpikingConv1DLayer
from .conv2d import SpikingConv2DLayer
from .conv3d import SpikingConv3DLayer
from .pool2d import SpikingPool2DLayer
from .dense import SpikingDenseLayer
from .readout import ReadoutLayer
from .heaviside import SurrogateHeaviside
from .readin import ReadInLayer


class SNN(torch.nn.Module):
    def __init__(self, spike_fn=None, device=None, dtype=None, time_expector=None, notifier=None, input_layer=None):
        super(SNN, self).__init__()
        self.layers = [] if input_layer is None else [input_layer]
        self.default_spike_fn = spike_fn if spike_fn is not None else SurrogateHeaviside.apply
        self.time_expector = time_expector
        self.notifier = notifier
        self.dtype = torch.float if dtype is None else dtype
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.to(device, dtype)

    def get_trainable_parameters(self, lr):
        res = []
        for l in self.layers:
            res.extend(l.get_trainable_parameters(lr))
        return res

    def add_conv1d(self, **kwargs):
        self.add_layer(SpikingConv1DLayer, **kwargs)

    def add_conv2d(self, **kwargs):
        self.add_layer(SpikingConv2DLayer, **kwargs)

    def add_conv3d(self, **kwargs):
        self.add_layer(SpikingConv3DLayer, **kwargs)

    def add_pool2d(self, **kwargs):
        self.add_layer(SpikingPool2DLayer, **kwargs)

    def add_dense(self, **kwargs):
        self.add_layer(SpikingDenseLayer, **kwargs)

    def add_readout(self, **kwargs):
        self.add_layer(ReadoutLayer, **kwargs)

    def add_layer(self, layer, **kwargs):
        if not self.layers:
            input_shape = kwargs.pop('input_shape')
            self.layers.append(ReadInLayer(input_shape))

        if layer.IS_SPIKING and self.default_spike_fn is not None and 'spike_fn' not in kwargs:
            kwargs['spike_fn'] = self.default_spike_fn

        if not layer.IS_CONV and self.layers[-1].IS_CONV:
            self.layers[-1].flatten_output = True

        # calculating this layer inputs based on last layer
        if layer.IS_CONV:
            kwargs = self.modify_param_for_conv(kwargs)
        else:
            kwargs = self.modify_param_for_flat(kwargs)

        self.layers.append(layer(**kwargs))

    def modify_param_for_conv(self, param):
        if 'input_shape' in param and 'input_channels' in param:
            return param

        if self.layers[-1].IS_CONV:
            param['input_shape'] = self.layers[-1].output_shape
            param['input_channels'] = self.layers[-1].output_channels
        else:
            raise NotImplementedError()  # FIXME : handle if last layer was not conv
        return param

    def modify_param_for_flat(self, param):
        if 'input_shape' in param:
            return param

        if self.layers[-1].IS_CONV:
            input_shape = 1
            last_out_shape = self.layers[-1].output_shape
            for sh in last_out_shape:
                input_shape *= sh
            input_shape *= self.layers[-1].output_channels
        else:
            input_shape = self.layers[-1].output_shape
        param['input_shape'] = input_shape
        return param

    def compile(self):
        self.layers = torch.nn.ModuleList(self.layers)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device, self.dtype)
        return self.forward(x)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def clamp(self):
        for l in self.layers:
            l.clamp()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    @staticmethod
    def write_result_log(file, loss, val_los, acc, val_acc):
        file.write('loss= % f' % loss)
        file.write('val_los = %f' % val_los)
        file.write('acc = %f' % acc)
        file.write('val_acc = %f' % val_acc)

    def save_network_summery(self, file):
        # for l in self.layers:
        pass

    def fit(self, data_loader, epochs=5, loss_func=None, optimizer=None, dataset_size=None, result_file=None, save_checkpoints=True):
        if result_file is not None:
            self.save_network_summery(result_file)

        # fix params before proceeding
        if loss_func is None:
            loss_func = torch.nn.NLLLoss()
        if dataset_size is None:
            dataset_size = 0.
            for _, _ in data_loader('train'):
                dataset_size += 1.
                if dataset_size % 64 == 1:
                    print('\rpre-processing dataset: %d' % dataset_size, end='')
            print('\rpre-processing dataset: %d' % dataset_size)
        if optimizer is None:
            lr = 0.1
            optimizer = torch.optim.SGD(self.get_trainable_parameters(lr), lr=lr, momentum=0.9)

        # train code
        res_metrics = {'train_loss_mean': [], 'test_loss_mean': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        for epoch in range(epochs):
            if self.time_expector is not None:
                self.time_expector.tick(epochs - epoch)

            # train
            dataset_counter = 0
            self.train()
            losses = []
            nums = []
            for x_batch, y_batch in data_loader('train'):
                dataset_counter += 1
                self.print_progress('Epoch: %d' % epoch, dataset_counter / dataset_size[0], a='-', c='.')
                l, n = self.batch_step(loss_func, x_batch, y_batch, optimizer)
                losses.append(l)
                nums.append(n)
                res_metrics['train_loss'].append(l)
            train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # evaluate
            self.eval()
            with torch.no_grad():
                losses = []
                nums = []
                dataset_counter = 0
                for x_batch, y_batch in data_loader('test'):
                    dataset_counter += 1
                    self.print_progress('Epoch: %d' % epoch, dataset_counter / dataset_size[1], a='=', c='-')
                    l, n = self.batch_step(loss_func, x_batch, y_batch)
                    losses.append(l)
                    nums.append(n)
                    res_metrics['test_loss'].append(l)
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # finishing up
            res_metrics['train_loss_mean'].append(train_loss)
            res_metrics['test_loss_mean'].append(val_loss)
            print("  | loss=%.3f val_loss=%.3f" % (train_loss, val_loss))

            train_accuracy = self.compute_classification_accuracy(data_loader('acc_train'), False)
            valid_accuracy = self.compute_classification_accuracy(data_loader('acc_test'), False)
            res_metrics['train_acc'].append(train_accuracy)
            res_metrics['test_acc'].append(valid_accuracy)
            print('train_accuracy=%.2f%%  |  valid_accuracy=%.2f%%' % (train_accuracy * 100., valid_accuracy * 100.))

            if result_file is not None:
                self.write_result_log(result_file, train_loss, val_loss, train_accuracy, valid_accuracy)

            if save_checkpoints:
                self.save_checkpoint()

            if self.time_expector is not None:
                self.time_expector.tock()
        return res_metrics

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

    @staticmethod
    def print_progress(msg, value, width=60, a='=', b='>', c='.'):
        print('\r%s [%s%s%s] %d%%    ' % (msg, a*int((value-0.001)*width), b, c*int((1.-value)*width), value*100), end='')

    def save_checkpoint(self):
        self.save(path.join('checkpoints', 'result_checkpoint_%d.net' % time()))

    def save(self, file):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, file)

    def load(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
