import numpy
import pytest


@pytest.fixture(scope="module")
def rgen():
    return numpy.random.RandomState(seed=3476583865)
