import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from random import random, randint, shuffle, choice, choices


def train_test_split(x_data, y_data, test_size):
    def _select(data, indices):
        return [data[i] for i in indices]
        
    idx_train = []
    idx_test = []
    for i in range(len(x_data)):
        if random() > test_size:
            idx_train.append(i)
        else:
            idx_test.append(i)

    return _select(x_data, idx_train), _select(x_data, idx_test), _select(y_data, idx_train), _select(y_data, idx_test)


def data_hist(y_data, num_of_classes):
    hist = {i:0 for i in range(num_of_classes)}
    for i in y_data:
        hist[i] += 1

    max_data = max(hist.values())
    for k, i in hist.items():
        print('%2d)' % k, '#' * int(30 * i / max_data), '(%d)' % i)


def equalizer(data_x, data_y, num_of_classes, augmentation_multiplier=0):
    label_indicies = {i: np.array(data_y) == i for i in range(num_of_classes)}
    label_histogram = {i: sum(label_indicies[i]) for i in range(num_of_classes)}

    target_value = max(label_histogram.values()) * (1+augmentation_multiplier)
    for i in range(num_of_classes):
        to_add = target_value - label_histogram[i]
        if not to_add:
            continue

        ids = np.nonzero(label_indicies[i])[0]
        for i in choices(ids, k=to_add):
            data_x.append(data_x[i])
            data_y.append(data_y[i])

    # print(len(data_x), '\n', len(data_y))
    idx = [i for i in range(len(data_x))]
    shuffle(idx)
    return [data_x[i] for i in idx], [data_y[i] for i in idx]


def plot_spikes_in_time(layer, batch_id=0):
    if not layer.HAS_PARAM:
        return

    if layer.IS_CONV:
        _plot_spikes_conv(layer, batch_id)
    else:
        _plot_spikes_dense(layer, batch_id)


def _plot_spikes_dense(layer, batch_id=0):
    if 'mem_rec_hist' in layer.__dict__:
        mem_rec_hist = layer.mem_rec_hist[batch_id]
        for i in range(mem_rec_hist.shape[1]):
            plt.plot(mem_rec_hist[:, i], label='mem')
            if i > 30:
                break
        plt.xlabel('Time')
        plt.ylabel('Membrace Potential')
        plt.show()

    if 'spk_rec_hist' in layer.__dict__:
        spk_rec_hist = layer.spk_rec_hist[batch_id]
        plt.plot(spk_rec_hist, 'b.')
        plt.xlabel('Time')
        plt.ylabel('Spikes')
        plt.show()

        plt.matshow(spk_rec_hist)
        plt.xlabel('Neuron')
        plt.ylabel('Spike Time')
        plt.axis([-1, spk_rec_hist.shape[1], -1, spk_rec_hist.shape[0]])
        plt.show()


def _plot_spikes_conv(layer, batch_id=0):
    spk_rec_hist = layer.spk_rec_hist[batch_id]
    mem_rec_hist = layer.mem_rec_hist[batch_id]

    time_step = mem_rec_hist.shape[1]
    channels = mem_rec_hist.shape[0]
    rest_shape = mem_rec_hist.shape[2:]

    tmp_spk = np.zeros((time_step, channels, *rest_shape))
    tmp_mem = np.zeros((time_step, channels, *rest_shape))
    for i in range(time_step):
        tmp_spk[i, :, :, :] = spk_rec_hist[:, i, :, :]
        tmp_mem[i, :, :, :] = mem_rec_hist[:, i, :, :]
    spk_rec_hist = tmp_spk
    mem_rec_hist = tmp_mem

    flat_spk = np.reshape(spk_rec_hist, (time_step, channels * np.prod(mem_rec_hist.shape[2:])))
    flat_mem = np.reshape(mem_rec_hist, (time_step, channels * np.prod(mem_rec_hist.shape[2:])))

    # Plot Flats
    max_flats = 25
    if flat_mem.shape[1] > max_flats:
        inx = np.random.randint(flat_mem.shape[1], size=max_flats)
        flat_spk = flat_spk[:, inx]
        flat_mem = flat_mem[:, inx]

    for i in range(flat_mem.shape[1]):
        plt.plot(flat_mem[:, i], label='mem')
    plt.xlabel('Time')
    plt.ylabel('Membrace Potential')
    plt.show()

    plt.plot(flat_spk, '.')
    plt.xlabel('Time')
    plt.ylabel('Spikes')
    plt.show()

    plt.matshow(flat_spk, cmap=plt.cm.gray_r, origin="upper", aspect='auto')
    plt.xlabel('Neuron')
    plt.ylabel('Spike Time')
    plt.axis([-1, flat_spk.shape[1], -1, flat_spk.shape[0]])
    plt.show()

    # Visual Plots
    max_visual = 5

    time_idx = list(range(0, time_step, int(time_step / max_visual)))
    neur_idx = np.random.randint(mem_rec_hist.shape[1], size=max_visual)

    gs = GridSpec(max_visual, max_visual)
    plt.figure(figsize=(30, 20))

    gs = GridSpec(max_visual, max_visual)
    plt.figure(figsize=(30, 20))

    # Draw Time based mems
    counter = 0
    for n in neur_idx:
        for t in time_idx:
            if counter == 0:
                a0 = ax = plt.subplot(gs[counter])
            else:
                ax = plt.subplot(gs[counter], sharey=a0)
            ax.imshow(mem_rec_hist[t, n, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='auto')
            plt.title('t(%d) - n(%d)' % (t, n))
            counter += 1
    plt.show()

    # Draw  Filters
    gs = GridSpec(3, 20)
    plt.figure(figsize=(10, 10))

    counter = 0
    for c_output in range(layer.output_channels):
        for c_input in range(layer.input_channels):
            if counter == 0:
                a0 = ax = plt.subplot(gs[counter])
            else:
                ax = plt.subplot(gs[counter], sharey=a0)
            ax.imshow(layer.w.detach().cpu().numpy()[c_output, c_input, 0, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='equal')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # plt.title('in(%d) - out(%d)' % (t, n))
            counter += 1

            if counter >= 60:
                break
        if counter >= 60:
            break
    plt.show()


def print_and_plot_accuracy_metrics(network, data_dl_train, data_dl_test, save_plot_path=None):
    plt.close()
    print('\n----------------------------------------')
    train_accuracy, heatmap_train = network.compute_classification_accuracy(data_dl_train)
    print("Final Train Accuracy=%.2f%%" % (train_accuracy * 100.))
    test_accuracy, heatmap_test = network.compute_classification_accuracy(data_dl_test)
    print("Final Test Accuracy=%.2f%%" % (test_accuracy * 100.))

    sns.heatmap(heatmap_train)
    plt.title('Train Result Heatmap (%.1f%%)' % (np.mean(np.array(train_accuracy))*100))
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    if save_plot_path is not None:
        plt.savefig(save_plot_path + 'train.png')
    plt.show()
    plt.close()

    sns.heatmap(heatmap_test)
    plt.title('Test Result Heatmap (%.1f%%)' % (np.mean(np.array(test_accuracy))*100))
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    if save_plot_path is not None:
        plt.savefig(save_plot_path + 'test.png')
    plt.show()
    plt.close()


def plot_metrics(res, save_plot_path=None):
    plt.close()
    plt.plot(res['train_loss_mean'], 'b', label='train')
    plt.plot(res['train_loss_max'], 'b--')
    plt.plot(res['train_loss_min'], 'b--')

    plt.plot(res['test_loss_mean'], 'r', label='test')
    plt.plot(res['test_loss_max'], 'r--')
    plt.plot(res['test_loss_min'], 'r--')
    plt.title('Loss Value')
    plt.legend()
    if save_plot_path is not None:
        plt.savefig(save_plot_path + 'loss.png')
    plt.show()
    plt.close()

    plt.plot(res['train_acc'], 'b', label='train')
    plt.plot(res['test_acc'], 'r--', label='test')
    plt.title('Accuracy Metrics')
    plt.legend()
    if save_plot_path is not None:
        plt.savefig(save_plot_path + 'accuracy.png')
    plt.show()
    plt.close()
