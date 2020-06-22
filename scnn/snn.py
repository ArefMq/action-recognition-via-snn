import torch
import numpy as np

import json
from time import time
from datetime import datetime
from os import path
from os import walk

from .heaviside import SurrogateHeaviside

from .BatchSpike.network import SpikingNeuralNetwork as BatchNetwork
from .BatchSpike.conv1d import SpikingConv1DLayer
from .BatchSpike.conv2d import SpikingConv2DLayer
from .BatchSpike.conv3d import SpikingConv3DLayer
from .BatchSpike.pool2d import SpikingPool2DLayer
from .BatchSpike.dense import SpikingDenseLayer
from .BatchSpike.readout import ReadoutLayer
from .BatchSpike.readin import ReadInLayer

from .StreamSpike.network import SpikingNeuralNetwork as StreamNetwork
from .StreamSpike.conv2d_stream import SpikingConv2DStream
from .StreamSpike.pool2d_stream import SpikingPool2DStream
from .StreamSpike.dense_stream import SpikingDenseStream
from .StreamSpike.readout_stream import ReadoutStream
from .StreamSpike.readin_stream import ReadInStream


def default_notifier(*msg, **kwargs):
    if kwargs.get('print_in_console', True):
        print(*msg)


class SNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNN, self).__init__()
        self.stream_network = kwargs.pop('stream_network', False)
        spike_fn = kwargs.pop('stream_network', None)
        self.default_spike_fn = spike_fn if spike_fn is not None else SurrogateHeaviside.apply

        if self.stream_network:
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
            self.network.layers.append(ReadInStream(input_shape) if self.stream_network else ReadInLayer(input_shape))

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
        file.write("network_design: " + self.serialize_to_text() + "\n")
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

    def serialize_to_text(self):
        res = ''
        for l in self.network.layers:
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
