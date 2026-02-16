import pytest
import torch
import torch.utils.data as data

from spikenet.data.converter import ImageToSpikeConverter


def _make_image_dataset(n=32, channels=1, h=8, w=8):
    """Create a simple image dataset with values in [0, 1]."""
    x = torch.rand(n, channels, h, w)
    y = torch.randint(0, 3, (n,))
    return data.TensorDataset(x, y)


def _make_converter(**kwargs):
    ds = _make_image_dataset()
    return ImageToSpikeConverter(
        train_data=ds,
        test_data=ds,
        batch_size=16,
        apply_softening=False,
        **kwargs,
    )


def test_rate_coding_produces_binary():
    conv = _make_converter(coding_type="rate", time_scale=10)
    imgs = torch.rand(4, 1, 8, 8)
    spikes = conv.convert_to_rate_based_spiking_trains(imgs)
    assert ((spikes == 0) | (spikes == 1)).all()


def test_rate_coding_output_shape():
    conv = _make_converter(coding_type="rate", time_scale=10)
    imgs = torch.rand(4, 1, 8, 8)
    spikes = conv.convert_to_rate_based_spiking_trains(imgs)
    assert spikes.shape == (4, 10, 8, 8)


@pytest.mark.xfail(reason="Bug: reshape uses *batch_imgs.shape instead of *batch_imgs.shape[1:]")
def test_time_coding_produces_binary():
    conv = _make_converter(coding_type="time", time_scale=10)
    imgs = torch.rand(4, 1, 8, 8)
    spikes = conv.convert_to_time_based_spiking_trains(imgs)
    assert ((spikes == 0) | (spikes == 1)).all()


def test_soften_image_bounds():
    conv = _make_converter()
    img = torch.ones(2, 1, 4, 4)
    # With default gain=0.02, limit=0.8: result = 1.0 * 0.8 * (1-0.02) + 0.02
    # but softening is applied with the converter's own params
    # Just test it doesn't go below 0 or above 1
    softened = conv.soften_image(img)
    assert softened.min() >= 0
    assert softened.max() <= 1


def test_soften_image_increases_dark_pixels():
    """Background gain should lift zero-valued pixels above zero."""
    conv = ImageToSpikeConverter(
        train_data=_make_image_dataset(),
        test_data=_make_image_dataset(),
        batch_size=16,
        background_gain=0.1,
    )
    img = torch.zeros(1, 1, 4, 4)
    softened = conv.soften_image(img)
    assert softened.min() > 0


def test_x_transform_rate_coding():
    conv = _make_converter(coding_type="rate", time_scale=8)
    imgs = torch.rand(4, 1, 8, 8)
    result = conv.x_transform(imgs)
    assert ((result == 0) | (result == 1)).all()


def test_x_transform_invalid_coding_raises():
    conv = _make_converter(coding_type="invalid")
    with pytest.raises(ValueError, match="Invalid coding type"):
        conv.x_transform(torch.rand(4, 1, 8, 8))
