import torch

from spikenet.network.criterion import Criterion


def test_default_values():
    c = Criterion()
    assert c.epochs == 10
    assert c.learning_rate == 0.0001
    assert c.optimizer is None


def test_get_optim_creates_optimizer():
    c = Criterion()
    net = torch.nn.Linear(10, 5)
    optim = c.get_optim(net)
    assert isinstance(optim, torch.optim.SGD)


def test_get_optim_uses_provided_optimizer():
    net = torch.nn.Linear(10, 5)
    custom_optim = torch.optim.Adam(net.parameters(), lr=0.01)
    c = Criterion(optimizer=custom_optim)
    assert c.get_optim(net) is custom_optim


def test_get_loss_fn():
    c = Criterion()
    net = torch.nn.Linear(10, 5)
    loss_fn = c.get_loss_fn(net)
    assert isinstance(loss_fn, torch.nn.NLLLoss)
