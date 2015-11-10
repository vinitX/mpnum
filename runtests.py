# encoding: utf-8


import os.path
import sys

import pytest


def main():
    # Run doctests afterwards because they auto-import things. We want to see
    # missing imports.
    pytest.main('tests tests_photonic_tomo')
    pytest.main('--doctest-modules mpnum photonic_tomo')


if __name__ == '__main__':
    main()


