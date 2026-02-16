import pytest

from spikenet.tools.window import window_to_and_array


def test_tuple_passthrough():
    result = window_to_and_array((2, 3, 4))
    assert result == (2, 3, 4)


def test_int_flat():
    result = window_to_and_array(5)
    assert result == (1, 5, 5)


def test_int_not_flat():
    result = window_to_and_array(5, flat=False)
    assert result == (5, 5, 5)


def test_invalid_tuple_length():
    with pytest.raises(AssertionError, match="Window must be a tuple of length 3"):
        window_to_and_array((1, 2))
