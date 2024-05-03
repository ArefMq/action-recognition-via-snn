import torch
import torchvision
import torchvision.transforms as transforms


from spikenet.network import Network
from spikenet.image_to_spike_convertor import ImageToSpikeConvertor, SpikePlotter

splt = SpikePlotter()


class MyData(ImageToSpikeConvertor):
    RESIZE_SIZE = (8, 8)
    def __init__(self):
        super().__init__(
            train_data=torchvision.datasets.MNIST(
                root="./data",
                train=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(self.RESIZE_SIZE),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            ),
            test_data=torchvision.datasets.MNIST(
                root="./data",
                train=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(self.RESIZE_SIZE),
                        transforms.ToTensor(),
                    ]
                ),
            ),
        )

    def x_transform(self, x):
        x = super().x_transform(x)
        res = x.reshape(-1, self.time_scale, self.RESIZE_SIZE[0] * self.RESIZE_SIZE[1])
        return res

    # def y_transform(self, y: torch.Any) -> torch.Any:
        # y = super().y_transform(y)
        # to one hot
        # y = torch.nn.functional.one_hot(y, num_classes=10).to(torch.float32)
        # return y


def create_net(input_dim=None):
    from spikenet.layers.spiking_dense import SpikingDenseLayer

    # net = (
    #     Network()
    #     .add_layer(SpikingDenseLayer, 500, name="l1", input_dim=input_dim)
    #     .add_layer(SpikingDenseLayer, 100, name="l1a")
    #     # .add_layer(SpikingDenseLayer, 100, name="l1b")
    #     .add_layer(SpikingDenseLayer, 10, name="l2", time_reduction="SpikeRate")
    # )

    net = (
        Network()
        .add_layer(torch.nn.Linear, 500)
        .add_layer(torch.nn.ReLU, in_features=500)
        .add_layer(torch.nn.Linear, 10)
    )

    return net


if __name__ == "__main__":
    data = MyData()
    data.describe()
    input_shape, _ = data.shape
    net = create_net(input_shape[1])
    net.fit_on(data, epochs=1)
    print("\n")