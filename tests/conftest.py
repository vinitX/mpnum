import numpy
import pytest


@pytest.fixture(scope="module")
def rgen():
    return numpy.random.RandomState(seed=52973992)


@pytest.fixture(scope='function', autouse=True)
def bug_workaround():
    # Workaround for https://github.com/pytest-dev/pytest/issues/1832
    pass
