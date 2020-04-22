import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
# import scipy.io.wavfile as wav

#
# def txt2list(filename):
#     lines_list = []
#     with open(filename, 'r') as txt:
#         for line in txt:
#             lines_list.append(line.rstrip('\n'))
#     return lines_list

#
# def plot_spk_rec(spk_rec, idx):
#     nb_plt = len(idx)
#     d = int(np.sqrt(nb_plt))
#     gs = GridSpec(d, d)
#     fig = plt.figure(figsize=(30, 20), dpi=150)
#     for i in range(nb_plt):
#         plt.subplot(gs[i])
#         plt.imshow(spk_rec[idx[i]].T, cmap=plt.cm.gray_r, origin="lower", aspect='auto')
#         if i == 0:
#             plt.xlabel("Time")
#             plt.ylabel("Units")
#
#
# def plot_mem_rec(mem, idx):
#     nb_plt = len(idx)
#     d = int(np.sqrt(nb_plt))
#     dim = (d, d)
#
#     gs = GridSpec(*dim)
#     plt.figure(figsize=(30, 20))
#     dat = mem[idx]
#
#     for i in range(nb_plt):
#         if i == 0:
#             a0 = ax = plt.subplot(gs[i])
#         else:
#             ax = plt.subplot(gs[i], sharey=a0)
#         ax.plot(dat[i])
#
#
# def get_random_noise(noise_files, size):
#     noise_idx = np.random.choice(len(noise_files))
#     fs, noise_wav = wav.read(noise_files[noise_idx])
#
#     offset = np.random.randint(len(noise_wav) - size)
#     noise_wav = noise_wav[offset:offset + size].astype(float)
#
#     return noise_wav
#
#
# def generate_random_silence_files(nb_files, noise_files, size, prefix, sr=16000):
#     for i in range(nb_files):
#         silence_wav = get_random_noise(noise_files, size)
#         wav.write(prefix + "_" + str(i) + ".wav", sr, silence_wav)
#
#
# def split_wav(waveform, frame_size, split_hop_length):
#     splitted_wav = []
#     offset = 0
#
#     while offset + frame_size < len(waveform):
#         splitted_wav.append(waveform[offset:offset + frame_size])
#         offset += split_hop_length
#
#     return splitted_wav


# New ones are here:

def plot_spikes_in_time(layer, batch_id=0):
    if layer.IS_CONV:
        _plot_spikes_conv(layer, batch_id)
    else:
        _plot_spikes_dense(layer, batch_id)


def _plot_spikes_dense(layer, batch_id=0):
    spk_rec_hist = layer.spk_rec_hist[batch_id]
    mem_rec_hist = layer.mem_rec_hist[batch_id]

    for i in range(mem_rec_hist.shape[1]):
        plt.plot(mem_rec_hist[:, i], label='mem')
    plt.xlabel('Time')
    plt.ylabel('Membrace Potential')

    plt.show()
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

    plt.matshow(flat_spk, cmap=plt.cm.gray_r, origin="lower", aspect='auto')
    plt.xlabel('Neuron')
    plt.ylabel('Spike Time')
    plt.axis([-1, flat_spk.shape[1], -1, flat_spk.shape[0]])
    plt.show()

    # Visual Plots
    max_visual = 5

    #     debug_print(spk_rec_hist, 'spk', pytorch=False)
    #     debug_print(mem_rec_hist, 'mem', pytorch=False)

    time_idx = list(range(0, time_step, int(time_step / max_visual)))
    neur_idx = np.random.randint(mem_rec_hist.shape[1], size=max_visual)

    gs = GridSpec(max_visual, max_visual)
    plt.figure(figsize=(30, 20))

    #     counter = 0
    #     for n in neur_idx:
    #         for t in time_idx:
    #             if counter == 0:
    #                 a0 = ax = plt.subplot(gs[counter])
    #             else:
    #                 ax = plt.subplot(gs[counter], sharey=a0)
    #             ax.imshow(spk_rec_hist[t, n, :, :], cmap=plt.cm.gray_r, origin="lower", aspect='auto')
    #             plt.title('t(%d) - n(%d)' % (t, n))
    #             counter += 1
    #     plt.show()

    gs = GridSpec(max_visual, max_visual)
    plt.figure(figsize=(30, 20))

    counter = 0
    for n in neur_idx:
        for t in time_idx:
            if counter == 0:
                a0 = ax = plt.subplot(gs[counter])
            else:
                ax = plt.subplot(gs[counter], sharey=a0)
            ax.imshow(spk_rec_hist[t, n, :, :], cmap=plt.cm.gray_r, origin="lower", aspect='auto')
            plt.title('t(%d) - n(%d)' % (t, n))
            counter += 1
    plt.show()


def print_and_plot_accuracy_metrics(network, data_dl_train, data_dl_test):
    print('\n----------------------------------------')
    train_accuracy, heatmap_train = network.compute_classification_accuracy(data_dl_train)
    print("Final Train Accuracy=%.2f%%" % (train_accuracy * 100.))
    test_accuracy, heatmap_test = network.compute_classification_accuracy(data_dl_test)
    print("Final Test Accuracy=%.2f%%" % (test_accuracy * 100.))

    sns.heatmap(heatmap_train)
    plt.title('Train Result Heatmap (%.1f%%)' % (np.mean(np.array(train_accuracy))*100))
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show()

    sns.heatmap(heatmap_test)
    plt.title('Test Result Heatmap (%.1f%%)' % (np.mean(np.array(test_accuracy))*100))
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show()
