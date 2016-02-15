# encoding: utf-8


import os
import os.path
import sys

import pytest


def main(*args):
    # Disable numba JIT optimization, which will take a couple of seconds (i.e.
    # as long as the tests are supposed to take).  This must be done before
    # importing numba.
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    # Run doctests afterwards because they auto-import things. We want to see
    # missing imports.
    pytest.main(list(args) + ['tests'])
    pytest.main(['--doctest-modules', 'mpnum'])


if __name__ == '__main__':
    main(*sys.argv[1:])
