import torch
import numpy as np

import json
from time import time
from datetime import datetime
from os import path
from os import walk

from .heaviside import SurrogateHeaviside

from .Spike.network import SpikingNeuralNetwork as BatchNetwork
from .Spike.conv1d import SpikingConv1DLayer
from .Spike.conv2d import SpikingConv2DLayer
from .Spike.conv3d import SpikingConv3DLayer
from .Spike.pool2d import SpikingPool2DLayer
from .Spike.dense import SpikingDenseLayer
from .Spike.readout import ReadoutLayer
from .Spike.readin import ReadInLayer

from .StreamSpike.network import SpikingNeuralNetwork as StreamNetwork
from .StreamSpike.conv2d_stream import SpikingConv2DStream
from .StreamSpike.pool2d_stream import SpikingPool2DStream
from .StreamSpike.dense_stream import SpikingDenseStream
from .StreamSpike.readout_stream import ReadoutStream
from .StreamSpike.readin_stream import ReadInStream


def default_notifier(*msg, **kwargs):
    if kwargs.get('print_in_console', True):
        print(*msg)


def tabler(table, column_config=None):
    # Find number of columns
    if column_config is None:
        num_of_columns = max([len(tr) for tr in table])
    else:
        num_of_columns = len(column_config)
        for i, cc in enumerate(column_config):
            if 'dir' not in cc:
                column_config[i]['dir'] = '<'

    # Fill N/A with empty string
    for t in range(len(table)):
        if num_of_columns > len(table[t]):
            for _ in range(num_of_columns - len(table[t])):
                table[t].append('')

    # Calculate each column length
    col_lens = []
    for c in range(num_of_columns):
        if column_config is not None:
            col_lens.append(max(len(column_config[c]['title']), max([len(tr[c]) for tr in table])))
        else:
            col_lens.append(max([len(tr[c]) for tr in table]))

    bar = '=' * (sum(col_lens) + num_of_columns*3) + '\n'

    # Create table
    str_formatter = ' | '.join(['{%d: %s%d}' % (i, column_config[i]['dir'] if column_config is not None else '<', cl) for i, cl in enumerate(col_lens)])
    result = ''
    if column_config is not None:
        result += bar
        result += str_formatter.format(*[cc['title'] for cc in column_config]) + '\n'
        result += bar
    for tr in table:
        result += str_formatter.format(*tr) + '\n'
    if column_config is not None:
        result += bar
    return result


class SNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNN, self).__init__()
        custom_network = kwargs.pop('custom_network', None)
        self.stream_network = kwargs.pop('stream_network', False)
        spike_fn = kwargs.pop('spike_fn', None)

        self.default_spike_fn = spike_fn if spike_fn is not None else SurrogateHeaviside.apply
        kwargs['save_network_summery_function'] = self.save_network_summery
        kwargs['write_result_log_function'] = self.write_result_log
        kwargs['save_checkpoint_function'] = self.save_checkpoint

        if custom_network and self.stream_network:
            raise Exception('can not select two type of network at the same time.')
        elif custom_network:
            self.network = custom_network(*args, **kwargs)
        elif self.stream_network:
            self.network = StreamNetwork(*args, **kwargs)
        else:
            self.network = BatchNetwork(*args, **kwargs)

    def add_conv1d(self, **kwargs):
        if self.stream_network:
            raise NotImplementedError('you probably want to add conv2d')
        self.add_layer(SpikingConv1DLayer, **kwargs)

    def add_conv2d(self, **kwargs):
        self.add_layer(SpikingConv2DStream if self.stream_network else SpikingConv2DLayer, **kwargs)

    def add_conv3d(self, **kwargs):
        if self.stream_network:
            raise NotImplementedError('you probably want to add conv2d')
        self.add_layer(SpikingConv3DLayer, **kwargs)

    def add_pool2d(self, **kwargs):
        self.add_layer(SpikingPool2DStream if self.stream_network else SpikingPool2DLayer, **kwargs)

    def add_dense(self, **kwargs):
        self.add_layer(SpikingDenseStream if self.stream_network else SpikingDenseLayer, **kwargs)

    def add_readout(self, **kwargs):
        self.add_layer(ReadoutStream if self.stream_network else ReadoutLayer, **kwargs)

    def add_layer(self, layer, **kwargs):
        if not self.network.layers:
            input_shape = kwargs.pop('input_shape')
            self.network.layers.append(ReadInStream(input_shape=input_shape) if self.stream_network else ReadInLayer(input_shape=input_shape))

        if layer.IS_SPIKING and self.default_spike_fn is not None and 'spike_fn' not in kwargs:
            kwargs['spike_fn'] = self.default_spike_fn

        if not layer.IS_CONV and self.network.layers[-1].IS_CONV:
            self.network.layers[-1].flatten_output = True

        # calculating this layer inputs based on last layer
        if layer.IS_CONV:
            kwargs = self.modify_param_for_conv(kwargs)
        else:
            kwargs = self.modify_param_for_flat(kwargs)

        self.network.layers.append(layer(**kwargs))

    def modify_param_for_conv(self, param):
        if 'input_shape' in param and 'input_channels' in param:
            return param

        if self.network.layers[-1].IS_CONV:
            param['input_shape'] = self.network.layers[-1].output_shape
            param['input_channels'] = self.network.layers[-1].output_channels
        else:
            raise NotImplementedError()  # FIXME : handle if last layer was not conv
        return param

    def modify_param_for_flat(self, param):
        if 'input_shape' in param:
            return param

        if self.network.layers[-1].IS_CONV:
            input_shape = 1
            last_out_shape = self.network.layers[-1].output_shape
            for sh in last_out_shape:
                input_shape *= sh
            input_shape *= self.network.layers[-1].output_channels
        else:
            input_shape = self.network.layers[-1].output_shape
        param['input_shape'] = input_shape
        return param

    def compile(self):
        self.network.compile()
        self.network = self.network.to(self.network.device, self.network.dtype)

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
        file.write("network_design: " + self.__str__() + "\n")
        file.flush()

    @staticmethod
    def from_file(file, weight_file=None, device=None, dtype=None):
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
        self.save(path.join('checkpoints', 'result_checkpoint_%d.net' % time()), save_structure=True)

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

    def __str__(self):
        res = ''
        for l in self.network.layers:
            if res != '':
                res += ' => '
            res += l.__str__()
        return res

    def print_summery(self):
        print('Network Summery:')
        print(self.summery())

    def summery(self):
        def multiply(l):
            res = 1
            for i in l:
                res *= i
            return res

        table_config = [{'title': t} for t in ['#', 'name', 'shape', 'parameters']]
        table_values = []
        for i, l in enumerate(self.network.layers):
            if 'readin' in l.name.lower() or 'pool' in l.name.lower():
                num_of_params = 'N/A'
            elif l.IS_CONV:
                num_of_params = str(multiply([l.output_channels, *l.kernel_size, l.input_channels]))
            else:
                num_of_params = str(l.output_shape * l.input_shape)

            row = [
                str(i),
                l.name,
                '{0}{1}'.format(l.output_shape, '' if l.output_channels is None else 'x %d' % l.output_channels),
                num_of_params,
            ]

            table_values.append(row)
            if 'pool' in l.name.lower():
                table_values.append([''] * len(row))

        return tabler(table_values, table_config)

    def deserialize(self, network_ser):
        for p in network_ser['network']:

            if p['type'] == 'readin':
                self.network.layers.append(ReadInLayer(**p['params']))
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
        for l in self.network.layers:
            res.append(l.serialize())
        return {'time': time(), 'network': res}

    @staticmethod
    def from_last_checkpoint(checkpoint_path='checkpoints', device=None, dtype=None):
        res = SNN(device=device, dtype=dtype)
        if not res.load_last_checkpoint(checkpoint_path, True):
            print('Can not find/load any checkpoint')
        return res.to(device, dtype)

    def load_last_checkpoint(self, checkpoint_path='checkpoints', load_structure=False):
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
                self.load(path.join(checkpoint_path, 'result_checkpoint_%d.net' % max_id), load_structure=load_structure)
                return True
            except:
                return False
        return False

    # FIXME: refactor these functions
    def get_trainable_parameters(self, *args, **kwargs):
        return self.network.get_trainable_parameters(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.network.fit(*args, **kwargs)

    def set_time_expector(self, te):
        self.network.time_expector = te

    def predict(self, *args, **kwargs):
        return self.network.predict(*args, **kwargs)

    def plot_one_batch(self, *args, **kwargs):
        return self.network.plot_one_batch(*args, **kwargs)

    def compute_classification_accuracy(self, *args, **kwargs):
        return self.network.compute_classification_accuracy(*args, **kwargs)

    def _parse_dense(self, l_str):
        l = [l.strip() for l in l_str.split(',')]
        neurons = l[0]
        recurrent = len(l) > 1 and 'r' in l[1]
        lateral = len(l) > 1 and 'l' in l[1]

        self.add_dense(
            output_shape=int(neurons),
            w_init_mean=0.0,
            w_init_std=0.8,
            recurrent=recurrent,
            lateral_connections=lateral,
            dropout_prob=0.3,
        )

    def _parse_conv(self, l_str):
        l = [l.strip() for l in l_str.split(',')]
        neurons = l[0]
        recurrent = (len(l) > 1 and 'r' in l[1]) or (len(l) > 2 and 'r' in l[2])
        lateral = (len(l) > 1 and 'l' in l[1]) or (len(l) > 2 and 'l' in l[2])
        kernel = [int(l[-1].replace('k', ''))]*3

        self.add_conv3d(
            output_channels=int(neurons),
            kernel_size=kernel,
            dilation=(1, 1, 1),
            recurrent=recurrent,
            lateral_connections=lateral,

            w_init_mean=0.00,
            w_init_std=0.05
        )
        self.add_pool2d(kernel_size=(2, 2), reduction='max')

    def parse_str(self, net_str):
        net = [n.strip() for n in net_str.split('->')]
        for l in net:
            if l.startswith('d'):
                self._parse_dense(l.replace('d(', '').replace(')', ''))
            if l.startswith('c'):
                self._parse_conv(l.replace('c(', '').replace(')', ''))
