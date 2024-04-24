import torch
import torchvision
import torchvision.transforms as transforms


from spikenet.network import Network
from spikenet.image_to_spike_convertor import ImageToSpikeConvertor, SpikePlotter

splt = SpikePlotter()


class MyData(ImageToSpikeConvertor):
    def __init__(self):
        super().__init__(
            train_data=torchvision.datasets.MNIST(
                root="./data",
                train=True,
                transform=transforms.ToTensor(),
                download=True,
            ),
            test_data=torchvision.datasets.MNIST(
                root="./data", train=False, transform=transforms.ToTensor()
            ),
        )

    def x_transform(self, x):
        x = super().x_transform(x)
        res = x.reshape(-1, self.time_scale, 28 * 28)
        return res


def create_net(input_shape, output_shape):
    from spikenet.layers.spiking_dense import SpikingDenseLayer

    net = (
        Network()
        .add_layer(
            SpikingDenseLayer, 
            name="l1", 
            input_dim=input_shape[1], 
            output_dim=100
        )
        .add_layer(
            SpikingDenseLayer,
            name="l2",
            input_dim=100,
            output_dim=10,
            time_reduction="max",
        )
    )
    print(net)
    return net


if __name__ == "__main__":
    data = MyData()
    data.describe()
    input_shape, output_shape = data.shape

    net = create_net(input_shape, output_shape)
    net = net.fit(data)
