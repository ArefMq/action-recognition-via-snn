import pytest
import torch

from spikenet.functions.pooling_reduction import avg_spike_rate, max_spike_rate, spike_time


def test_max_spike_rate_output_shape():
    x = torch.rand(2, 4, 8, 14, 14)
    result = max_spike_rate(x, kernel=(1, 2, 2), stride=(1, 2, 2))
    assert result.shape == (2, 4, 8, 7, 7)


def test_max_spike_rate_selects_max():
    x = torch.zeros(1, 1, 1, 2, 2)
    x[0, 0, 0, 1, 0] = 1.0
    result = max_spike_rate(x, kernel=(1, 2, 2), stride=(1, 2, 2))
    assert result[0, 0, 0, 0, 0].item() == 1.0


def test_avg_spike_rate_output_shape():
    x = torch.rand(2, 4, 8, 14, 14)
    result = avg_spike_rate(x, kernel=(1, 2, 2), stride=(1, 2, 2))
    assert result.shape == (2, 4, 8, 7, 7)


def test_avg_spike_rate_computes_mean():
    x = torch.zeros(1, 1, 1, 2, 2)
    x[0, 0, 0, 0, 0] = 1.0
    x[0, 0, 0, 1, 1] = 1.0
    result = avg_spike_rate(x, kernel=(1, 2, 2), stride=(1, 2, 2))
    assert torch.isclose(result[0, 0, 0, 0, 0], torch.tensor(0.5))


def test_spike_time_not_implemented():
    x = torch.rand(1, 1, 1, 4, 4)
    with pytest.raises(NotImplementedError):
        spike_time(x, kernel=(1, 2, 2), stride=(1, 2, 2))
