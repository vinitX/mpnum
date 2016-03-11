#!/usr/bin/env python
# encoding: utf-8
# TODO Requirements are not really minimal at the moment...
# TODO Having PyTest in requirements not optimal, fix to make travis use
#      the tests_require

import os
import sys

from setuptools import Command, setup

authors = [u"Daniel Suess", u"Milan Holzäpfel"]
name = "mpnum"
description = "matrix product representation library"
year = "2016"


try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/" + name)
    from mpnum import __version__ as version
except:
    version = "unknown"


class PyTest(Command):
    """Perform tests"""

    description = "Runs test suite"
    user_options = [
        ('selector=', None, "Specifies the tests to run"),
        ('covreport', None, "Run coverage report from tests")
    ]

    def initialize_options(self):
        self.selector = 'not long'
        self.covreport = None

    def finalize_options(self):
        pass

    def run(self):
        import pytest
        # Disable numba JIT optimization, which will take a couple of seconds
        # (i.e. as long as the tests are supposed to take).  This must be done
        # before importing numba.
        os.environ['NUMBA_DISABLE_JIT'] = '1'

        if self.covreport is None:
            # Run doctests afterwards because they auto-import things. We want
            # to see missing imports.
            errno = pytest.main(['-m', self.selector, 'tests'])
            if errno != 0:
                raise SystemExit(errno)

            errno = pytest.main(['--doctest-modules', 'mpnum'])
            if errno != 0:
                raise SystemExit(errno)

        else:
            errno = pytest.main(['-m', self.selector, '--cov-report', 'term',
                                 '--cov-report', 'html', '--cov=mpnum',
                                 '--doctest-modules', 'mpnum', 'tests'])
            if errno != 0:
                raise SystemExit(errno)


if __name__ == '__main__':
    setup(
        name=name,
        author=', '.join(authors),
        url="https://github.com/dseuss/mpnum",
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD",
        description=description,
        install_requires=[
            'NumPy>=1.5.1',
            'SciPy>=0.15',
            'six>=1.0',
            'PyTest>=2.8.7'
        ],
        keywords=[],
        classifiers=[
            "Operating System :: OS Indendent",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Intended Audience :: Science/Research"
        ],
        platforms=['ALL'],
        cmdclass={
            'test': PyTest
        }
    )
