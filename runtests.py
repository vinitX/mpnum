# encoding: utf-8


import os
import os.path
import sys

import pytest


def main():
    # Disable numba JIT optimization, which will take a couple of seconds (i.e.
    # as long as the tests are supposed to take).  This must be done before
    # importing numba.
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    # Run doctests afterwards because they auto-import things. We want to see
    # missing imports.
    pytest.main('tests tests_photonic_tomo')
    pytest.main('--doctest-modules mpnum photonic_tomo')


if __name__ == '__main__':
    main()


