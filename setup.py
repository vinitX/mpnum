#!/usr/bin/env python
# encoding: utf-8
# TODO Requirements are not really minimal at the moment...
# TODO Having PyTest in requirements not optimal, fix to make travis use
#      the tests_require

import os
import sys

from setuptools import find_packages, setup

authors = [u"Daniel Suess", u"Milan Holzäpfel"]
author_emails = ["daniel@dsuess.me", "mail@mjh.name"]
name = "mpnum"
description = "matrix product representation library"
try:
    # can be created via pandoc:
    # pandoc --from=markdown --to=rst --output=README.rst README.md
    long_description = open('README.txt').read()
except IOError:
    long_description = description
year = "2017"


try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/" + name)
    from mpnum import __version__ as version
except:
    version = "unknown"


_install_requires = [
    'SciPy>=0.15',
    'NumPy>=1.5.1',
    'six>=1.0'
]


def _get_install_requires(req, not_on_rtd=['scipy', 'numpy']):
    """Remove packages which cannot be installed on readthedocs.org

    scipy and numpy are available on readthedocs as system packages,
    but they must not appear in install_requires (see
    http://docs.readthedocs.org/en/latest/faq.html).

    """
    on_rtd = os.environ.get('READTHEDOCS') == 'True'
    if on_rtd:
        req = [dep for dep in req
               if all(blocker.lower() not in dep.lower()
                      for blocker in not_on_rtd)]
    return req


if __name__ == '__main__':
    setup(
        name=name,
        author=', '.join(authors),
        author_email=', '.join(author_emails),
        url="https://github.com/dseuss/mpnum",
        version=version,
        packages=find_packages(exclude=['tests']),
        license="BSD",
        description=description,
        long_description=long_description,
        install_requires=_get_install_requires(_install_requires),
        setup_requires=['pytest-runner'],
        package_dir={'mpnum': 'mpnum'},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Science/Research"
        ],
        platforms=['ALL'],
    )
