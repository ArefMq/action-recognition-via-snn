import torch

from spikenet.functions.heaviside import SurrogateHeaviside


def test_forward_positive_inputs():
    x = torch.tensor([0.5, 1.0, 2.0])
    result = SurrogateHeaviside.apply(x)
    assert torch.equal(result, torch.ones(3))


def test_forward_negative_inputs():
    x = torch.tensor([-0.5, -1.0, -2.0])
    result = SurrogateHeaviside.apply(x)
    assert torch.equal(result, torch.zeros(3))


def test_forward_zero_input():
    x = torch.tensor([0.0])
    result = SurrogateHeaviside.apply(x)
    assert result.item() == 0.0


def test_forward_is_binary():
    x = torch.randn(100)
    result = SurrogateHeaviside.apply(x)
    assert ((result == 0) | (result == 1)).all()


def test_backward_produces_nonzero_gradient():
    x = torch.randn(10, requires_grad=True)
    result = SurrogateHeaviside.apply(x)
    result.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_backward_gradient_near_zero_is_largest():
    """Surrogate gradient should be largest near zero and decay away from it."""
    x_near = torch.tensor([0.01], requires_grad=True)
    x_far = torch.tensor([5.0], requires_grad=True)

    SurrogateHeaviside.apply(x_near).backward()
    SurrogateHeaviside.apply(x_far).backward()

    assert x_near.grad.abs().item() > x_far.grad.abs().item()
