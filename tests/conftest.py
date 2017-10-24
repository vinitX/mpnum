import numpy as np
import pytest


def pytest_namespace():
    return dict(
        # nr_sites, local_dim, rank
        MP_TEST_PARAMETERS=[(1, 7, np.nan), (2, 3, 3), (3, 2, 4), (6, 2, 4),
                            (4, 3, 5), (5, 2, 1)],
        MP_TEST_DTYPES=[np.float_, np.complex_]
    )


@pytest.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)
