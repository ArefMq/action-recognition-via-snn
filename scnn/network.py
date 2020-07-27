import torch
import numpy as np

from scnn.funny_printer import FunPack


def default_notifier(*msg, **kwargs):
    if kwargs.get('print_in_console', True):
        print(*msg)


class SpikingNeuralNetworkBase(torch.nn.Module):
    def __init__(self, save_network_summery_function, write_result_log_function, save_checkpoint_function, device=None, dtype=None, time_expector=None, notifier=None, input_layer=None):
        super(SpikingNeuralNetworkBase, self).__init__()
        self.layers = [] if input_layer is None else [input_layer]
        self.time_expector = time_expector
        self.notifier = notifier if notifier is not None else default_notifier
        self.dtype = torch.float if dtype is None else dtype
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.to(device, dtype)
        self.cute_print = False
        self.fp = FunPack()
        self.save_network_summery = save_network_summery_function
        self.write_result_log = write_result_log_function
        self.save_checkpoint = save_checkpoint_function

        self.res_metrics = {
            'train_loss_mean': [],
            'test_loss_mean': [],

            'train_loss_max': [],
            'train_loss_min': [],
            'test_loss_max': [],
            'test_loss_min': [],

            'train_acc': [],
            'test_acc': []
        }

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        res = []
        for l in self.layers:
            res.extend(l.get_trainable_parameters(lr=lr, weight_decay=weight_decay))
        return res

    def init(self):
        self.reset_parameters()
        self.clamp()

    def compile(self):
        self.layers = torch.nn.ModuleList(self.layers)
        self.init()

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
        for k in self.res_metrics.keys():
            if len(self.res_metrics[k]):
                self.res_metrics[k].append(0)

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
            self.res_metrics['train_loss_max'].append(np.max(losses))
            self.res_metrics['train_loss_min'].append(np.min(losses))
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
            self.res_metrics['test_loss_max'].append(np.max(losses))
            self.res_metrics['test_loss_min'].append(np.min(losses))
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # finishing up
            self.print_progress(epoch, epochs, dataset_counter, dataset_size, finished=True)
            self.res_metrics['train_loss_mean'].append(train_loss)
            self.res_metrics['test_loss_mean'].append(val_loss)

            train_accuracy = self.compute_classification_accuracy(data_loader('acc_train'), False)
            valid_accuracy = self.compute_classification_accuracy(data_loader('acc_test'), False)
            self.res_metrics['train_acc'].append(train_accuracy)
            self.res_metrics['test_acc'].append(valid_accuracy)

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
        return self.res_metrics

    def reset_timer(self):
        if self.time_expector is not None:
            self.time_expector.tock()

    def print_progress(self, epoch, epoch_count, dataset_counter, dataset_size, test=False, finished=False):
        dataset_counter *= 1.
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

        self._print_progress('Epoch: %d' % (epoch + 1),
                             dataset_counter / d_size if not finished else 1.,
                             a='=' if test or finished else '-',
                             c='-' if test or finished else '.',
                             expectation=expectation,
                             funny_func=self.fp.funnify if self.cute_print else None)

        if finished:
            if self.cute_print:
                self.fp.init_fp(42)
            print('')

    @staticmethod
    def _print_progress(msg, value, width=60, a='=', b='>', c='.', expectation='', funny_func=None, reverse=False):
        if funny_func is not None:
            reverse = True
            width *= .7

        progress_text = '%s%s%s' % (
            (a * int((value - 0.001) * width)) if not reverse else (c * int((1. - value) * width)),
            b,
            (c * int((1. - value) * width)) if not reverse else (a * int((value - 0.0000001) * width))
        )

        if funny_func is not None:
            progress_text = ''.join([funny_func(fn_i, fn_c, c) for fn_i, fn_c in enumerate(progress_text)])

        output = '\r%s [%s] %3d%%  %30s   ' % (
            msg,
            progress_text,
            value * 100,
            expectation
        )

        print(output, end='')

    def plot_one_batch(self, x_batch, y_batch=None, batch_id=0):
        self.predict(x_batch)

        for i, l in enumerate(self.layers):
            if 'spk_rec_hist' in l.__dict__:
                print("Layer {}: average number of spikes={:.4f}".format(i, l.spk_rec_hist.mean()))
            l.draw(batch_id)
