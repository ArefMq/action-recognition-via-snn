from spikenet.network import Network
from spikenet.image_to_spike_convertor import SpikePlotter
from spikenet.dataset.spiking_mnist import SpikingMNISTDataLoader

spiking_mnist = SpikingMNISTDataLoader(frame_size=14)
splt = SpikePlotter()


def create_net():
    from spikenet.layers.spiking_base import TimeReduction
    from spikenet.layers.spiking_dense import SpikingDenseLayer
    from spikenet.layers.spiking_conv2d import SpikingConv2D
    from spikenet.layers.spiking_pooling import SpikingPoolingLayer

    return (
        Network()
        .add_layer(SpikingConv2D, 16, in_features=1, in_size=(14, 14))
        .add_layer(SpikingPoolingLayer, in_size=(14, 14), out_size=(7, 7))
        .add_layer(SpikingConv2D, 32, in_features=16, in_size=(7, 7))
        .add_layer(SpikingPoolingLayer, in_size=(7, 7), out_size=(3, 3))
        .add_layer(SpikingDenseLayer, 100)
        .add_layer(SpikingDenseLayer, 10, time_reduction=TimeReduction.MemRecMax)
    )

if __name__ == "__main__":
    # data.describe()
    net = create_net()
    # net.build()
    net.summary()
    # print("\n\n")
    # net.fit_on(spiking_mnist, epochs=1)
    # print("\n")