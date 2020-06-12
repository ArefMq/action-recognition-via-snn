import torch
import numpy as np

import json
from time import time
from datetime import datetime
from os import path
from os import walk

from .conv1d import SpikingConv1DLayer
from .conv2d import SpikingConv2DLayer
from .conv3d import SpikingConv3DLayer
from .pool2d import SpikingPool2DLayer
from .dense import SpikingDenseLayer
from .readout import ReadoutLayer
from .heaviside import SurrogateHeaviside
from .readin import ReadInLayer


def default_notifier(*msg, **kwargs):
    if kwargs.get('print_in_console', True):
        print(*msg)


class SNN(torch.nn.Module):
    def __init__(self, spike_fn=None, device=None, dtype=None, time_expector=None, notifier=None, input_layer=None):
        super(SNN, self).__init__()
        self.layers = [] if input_layer is None else [input_layer]
        self.default_spike_fn = spike_fn if spike_fn is not None else SurrogateHeaviside.apply
        self.time_expector = time_expector
        self.notifier = notifier if notifier is not None else default_notifier
        self.dtype = torch.float if dtype is None else dtype
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.to(device, dtype)

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        res = []
        for l in self.layers:
            res.extend(l.get_trainable_parameters(lr=lr, weight_decay=weight_decay))
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
        file.write('loss: % f\n' % loss)
        file.write('val_los: %f\n' % val_los)
        file.write('acc: %f\n' % acc)
        file.write('val_acc: %f\n' % val_acc)
        file.write('--------------------------------------------\n')
        file.flush()

    def save_network_summery(self, file):
        file.write('\n============================================\n')
        file.write('New Run: ' + str(datetime.now()) + "\n")
        file.write("network_design: " + self.serialize() + "\n")
        file.flush()

    def fit(self, data_loader, epochs=5, loss_func=None, optimizer=None, dataset_size=None, result_file=None, save_checkpoints=True):
        if self.time_expector is not None:
            self.time_expector.reset()

        if result_file is not None:
            self.save_network_summery(result_file)

        # fix params before proceeding
        if loss_func is None:
            loss_func = torch.nn.NLLLoss()
        if dataset_size is None:
            dataset_size = [0., 0.]
            for _, _ in data_loader('train'):
                dataset_size[0] += 1.
                if dataset_size[0] % 64 == 1:
                    print('\rpre-processing dataset: %d' % dataset_size[0], end='')
            print('\rpre-processing dataset: %d' % dataset_size[0])
            for _, _ in data_loader('test'):
                dataset_size[1] += 1.
                if dataset_size[1] % 64 == 1:
                    print('\rpre-processing dataset: %d' % dataset_size[1], end='')
            print('\rpre-processing dataset: %d' % dataset_size[1])
        if optimizer is None:
            lr = 0.1
            optimizer = torch.optim.SGD(self.get_trainable_parameters(lr), lr=lr, momentum=0.9)

        # train code
        res_metrics = {
            'train_loss_mean': [],
            'test_loss_mean': [],

            'train_loss_max': [],
            'train_loss_min': [],
            'test_loss_max': [],
            'test_loss_min': [],

            'train_acc': [],
            'test_acc': []
        }

        if result_file is not None:
            result_file.write('New Run\n------------------------------\n')

        for epoch in range(epochs):
            if self.time_expector is not None:
                self.time_expector.macro_tick()

            # train
            dataset_counter = 0
            self.train()
            losses = []
            nums = []
            for x_batch, y_batch in data_loader('train'):
                dataset_counter += 1
                self.print_progress(epoch, epochs, dataset_counter, dataset_size)
                l, n = self.batch_step(loss_func, x_batch, y_batch, optimizer)
                self.reset_timer()
                losses.append(l)
                nums.append(n)
            res_metrics['train_loss_max'].append(np.max(losses))
            res_metrics['train_loss_min'].append(np.min(losses))
            train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # evaluate
            self.eval()
            with torch.no_grad():
                losses = []
                nums = []
                dataset_counter = 0
                for x_batch, y_batch in data_loader('test'):
                    dataset_counter += 1
                    self.print_progress(epoch, epochs, dataset_counter, dataset_size, test=True)
                    l, n = self.batch_step(loss_func, x_batch, y_batch)
                    self.reset_timer()
                    losses.append(l)
                    nums.append(n)
            res_metrics['test_loss_max'].append(np.max(losses))
            res_metrics['test_loss_min'].append(np.min(losses))
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # finishing up
            self.print_progress(epoch, epochs, dataset_counter, dataset_size, finished=True)
            res_metrics['train_loss_mean'].append(train_loss)
            res_metrics['test_loss_mean'].append(val_loss)

            train_accuracy = self.compute_classification_accuracy(data_loader('acc_train'), False)
            valid_accuracy = self.compute_classification_accuracy(data_loader('acc_test'), False)
            res_metrics['train_acc'].append(train_accuracy)
            res_metrics['test_acc'].append(valid_accuracy)

            print('')
            print('| Lss.Trn | Lss.Tst | Acc.Trn | Acc.Tst |')
            print('|---------|---------|---------|---------|')
            print('|  %6.4f |  %6.4f | %6.2f%% | %6.2f%% |' % (train_loss, val_loss, train_accuracy * 100., valid_accuracy * 100.))
            print('')

            if result_file is not None:
                self.write_result_log(result_file, train_loss, val_loss, train_accuracy, valid_accuracy)

            if save_checkpoints:
                self.save_checkpoint()

            self.notifier('epoch %d ended (acc=%.2f ~ %.2f)' % (epoch, train_accuracy, valid_accuracy), print_in_console=False)

        self.notifier('Done', mark='ok', print_in_console=False)
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

    def reset_timer(self):
        if self.time_expector is not None:
            self.time_expector.tock()

    def print_progress(self, epoch, epoch_count, dataset_counter, dataset_size, test=False, finished=False):
        d_size = dataset_size[1] if test else dataset_size[0]
        iter_per_epoch = dataset_size[0] + dataset_size[1]
        d_left = iter_per_epoch - dataset_counter
        epoch_left = epoch_count - epoch - 1

        if self.time_expector is not None:
            if finished:
                expectation = self.time_expector.macro_tock()
            else:
                self.time_expector.tick()
                expectation = self.time_expector.expectation(epoch_left, d_left, iter_per_epoch)
        else:
            expectation = ''

        self._print_progress('Epoch: %d' % (epoch+1),
                             dataset_counter / d_size,
                             a='=' if test or finished else '-',
                             c='-' if test or finished else '.',
                             expectation=expectation)

        if finished:
            print('')

    @staticmethod
    def _print_progress(msg, value, width=60, a='=', b='>', c='.', expectation=''):
        print('\r%s [%s%s%s] %3d%%  %30s   ' % (msg, a*int((value-0.001)*width), b, c*int((1.-value)*width), value*100, expectation), end='')

    @staticmethod
    def load_from_file(file, weight_file=None, device=None, dtype=None):
        res = SNN(device=device, dtype=dtype)
        res.load(file, weight_file)
        return res.to(device, dtype)

    def load(self, file, weight_file=None, load_structure=True):
        if load_structure:
            with open(file, 'r') as f:
                network_ser = json.load(f)
            self.deserialize(network_ser)

        if weight_file is None:
            weight_file = file + '.weight'
        checkpoint = torch.load(weight_file, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save_checkpoint(self):
        self.save(path.join('checkpoints', 'result_checkpoint_%d.net' % time()), False)

    def save(self, file, weight_file=None, save_structure=True):
        if weight_file is None:
            weight_file = file + '.weight'
        torch.save({
            'model_state_dict': self.state_dict(),
        }, weight_file)

        if save_structure:
            with open(file, 'w') as f:
                json.dump(self.json_normalizer(self.serialize()), f, indent=2)

    def _json_normalizer_list(self, j):
        return [self.json_normalizer(i) for i in j]

    def _json_normalizer_dict(self, j):
        return {k: self.json_normalizer(v) for k, v in j.items()}

    def json_normalizer(self, j):
        if isinstance(j, dict):
            return self._json_normalizer_dict(j)
        elif isinstance(j, list):
            return self._json_normalizer_list(j)
        elif isinstance(j, np.ndarray):
            return self._json_normalizer_list(list(j))
        elif isinstance(j, np.int64):
            return int(j)
        else:
            return j

    def serialize_to_text(self):
        res = ''
        for l in self.layers:
            if res != '':
                res += ' => '
            res += l.serialize_to_text()
        return res

    def deserialize(self, network_ser):
        for p in network_ser['network']:

            if p['type'] == 'readin':
                self.layers.append(ReadInLayer(**p['params']))
            elif p['type'] == 'conv1d':
                self.add_conv1d(**p['params'])
            elif p['type'] == 'conv2d':
                self.add_conv2d(**p['params'])
            elif p['type'] == 'conv3d':
                self.add_conv3d(**p['params'])
            elif p['type'] == 'pool2d':
                self.add_pool2d(**p['params'])
            elif p['type'] == 'dense':
                self.add_dense(**p['params'])
            elif p['type'] == 'readout':
                self.add_readout(**p['params'])
        self.compile()

    def serialize(self):
        res = []
        for l in self.layers:
            res.append(l.serialize())
        return {'time': time(), 'network': res}

    def load_last_checkpoint(self, checkpoint_path='checkpoints'):
        files = []
        for (dirpath, dirnames, filenames) in walk(checkpoint_path):
            files.extend(filenames)
            break
        files = [f.replace('result_checkpoint_', '').replace('.net', '') for f in files if f.endswith('.net') and f.startswith('result_checkpoint_')]

        max_id = None
        for f in files:
            try:
                i = int(f)
                if max_id is None or i > max_id:
                    max_id = i
            except:
                continue

        if max_id is not None:
            try:
                self.load(path.join(checkpoint_path, 'result_checkpoint_%d.net' % max_id), load_structure=False)
                return True
            except:
                return False
        return False
