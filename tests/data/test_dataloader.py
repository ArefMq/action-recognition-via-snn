import torch
import torch.utils.data as data

from spikenet.data.dataloader import DataLoader


def _make_dataset(n=100, x_dim=10, num_classes=3):
    x = torch.randn(n, x_dim)
    y = torch.randint(0, num_classes, (n,))
    return data.TensorDataset(x, y)


def test_len_train():
    ds = _make_dataset(n=50)
    loader = DataLoader(train_data=ds, test_data=ds, batch_size=16)
    assert loader.len("train") == 50


def test_len_test():
    train_ds = _make_dataset(n=50)
    test_ds = _make_dataset(n=30)
    loader = DataLoader(train_data=train_ds, test_data=test_ds, batch_size=16)
    assert loader.len("test") == 30


def test_call_yields_batches():
    ds = _make_dataset(n=32)
    loader = DataLoader(train_data=ds, test_data=ds, batch_size=16)
    batches = list(loader("train"))
    assert len(batches) == 2
    x, y = batches[0]
    assert x.shape == (16, 10)


def test_sample_returns_single_item():
    ds = _make_dataset(n=32)
    loader = DataLoader(train_data=ds, test_data=ds, batch_size=16)
    x, y = loader.sample()
    assert x.shape == (10,)
    assert y.shape == ()


def test_shape_property():
    ds = _make_dataset(n=32, x_dim=20)
    loader = DataLoader(train_data=ds, test_data=ds, batch_size=16)
    x_shape, y_shape = loader.shape
    assert x_shape == torch.Size([20])
    assert y_shape == torch.Size([])
