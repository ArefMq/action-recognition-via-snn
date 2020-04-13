import torch
import numpy as np

from .conv1d import SpikingConv1DLayer
from .conv2d import SpikingConv2DLayer
from .conv3d import SpikingConv3DLayer
from .pool2d import SpikingPool2DLayer
from .dense import SpikingDenseLayer
from .readout import ReadoutLayer
from .heaviside import SurrogateHeaviside


class SNN(torch.nn.Module):
    def __init__(self, spike_fn=None, device=None, dtype=None, time_expector=None, notifier=None):
        super(SNN, self).__init__()
        self.layers = []
        self.default_spike_fn = spike_fn if spike_fn is not None else SurrogateHeaviside.apply
        self.last_layer_shape = None
        self.last_layer_is_conv = None
        self.time_expector = time_expector
        self.notifier = notifier
        self.dtype = torch.float if dtype is None else dtype
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.to(device, dtype)

    def get_trainable_parameters(self):
        res = []
        for l in self.layers:
            res.extend(l.get_trainable_parameters())
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
        if layer.IS_SPIKING and self.default_spike_fn is not None and 'spike_fn' not in kwargs:
            kwargs['spike_fn'] = self.default_spike_fn

        if not layer.IS_CONV and self.last_layer_is_conv:
            self.layers[-1].flatten_output = True

        # calculating this layer inputs based on last layer
        if self.last_layer_shape is not None:
            llsh = self.last_layer_shape
            if layer.IS_CONV and 'input_channels' not in kwargs and 'input_shape' not in kwargs:
                if self.last_layer_is_conv:
                    kwargs['input_channels'] = llsh['channels']
                    kwargs['input_shape'] = llsh['shape']
                else:
                    raise NotImplementedError()  # FIXME : handle if last layer was not conv
            elif not layer.IS_CONV and 'input_shape' not in kwargs:
                if self.last_layer_is_conv:
                    input_shape = 1
                    for i in range(len(llsh['shape'])):
                        input_shape *= llsh['shape'][i]
                    input_shape *= llsh['channels']
                    kwargs['input_shape'] = input_shape
                else:
                    kwargs['input_shape'] = llsh['shape']

        new_layer = layer(**kwargs)
        if new_layer.IS_CONV:
            self.last_layer_shape = {'channels': new_layer.output_channels, 'shape': new_layer.output_shape}
        else:
            self.last_layer_shape = {'shape': new_layer.output_shape}

        self.layers.append(new_layer)
        self.last_layer_is_conv = layer.IS_CONV

    def compile(self):
        self.layers = torch.nn.ModuleList(self.layers)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        shp = x.shape
        if self.layers[0].IS_CONV:
            x = x.view(shp[0], 1, shp[1], shp[2], shp[3])
        else:
            x = x.view(shp[0], shp[1], shp[2] * shp[3])
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

    def fit(self, data_loader, epochs=5, loss_func=None, optimizer=None, dataset_size=None):
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
            optimizer = torch.optim.SGD(self.get_trainable_parameters(), lr=0.1, momentum=0.9)

        # train code
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
                self.print_progress('Epoch: %d' % epoch, dataset_counter / dataset_size, width=60)
                l, n = self.batch_step(loss_func, x_batch, y_batch, optimizer)
                losses.append(l)
                nums.append(n)
            train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # evaluate
            self.eval()
            with torch.no_grad():
                losses = []
                nums = []
                for x_batch, y_batch in data_loader('test'):
                    l, n = self.batch_step(loss_func, x_batch, y_batch)
                    losses.append(l)
                    nums.append(n)
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # finishing up
            print("  | loss=%.3f val_loss=%.3f" % (train_loss, val_loss))

            train_accuracy = self.compute_classification_accuracy(data_loader('acc_train'))
            valid_accuracy = self.compute_classification_accuracy(data_loader('acc_test'))
            print('train_accuracy=%.2f%%  |  valid_accuracy=%.2f%%' % (train_accuracy * 100., valid_accuracy * 100.))

            if self.time_expector is not None:
                self.time_expector.tock()

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

    def compute_classification_accuracy(self, data_dl):
        accs = []
        with torch.no_grad():
            for x_batch, y_batch in data_dl:
                output = self.predict(x_batch)
                y_batch = torch.from_numpy(y_batch.astype(np.long)).to(self.device)
                _, am = torch.max(output, 1)  # argmax over output units
                tmp = np.mean((y_batch == am).detach().cpu().numpy())  # compare to labels
                accs.append(tmp)
        return np.mean(accs)

    @staticmethod
    def print_progress(msg, value, width=80, a='=', b='>', c='.'):
        print('\r%s [%s%s%s] %d%%' % (msg, a*int(value*width), b, c*int((1.-value)*width), value*100), end='')

