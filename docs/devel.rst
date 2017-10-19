

.. _mpnum-development:

Contributing
============


This section contains information for anyone who wishes to contribute to
mpnum. Contributions and pull requests for mpnum are very welcome.


.. contents::


Code style
----------

All contributions should be formated according to the `PEP8 standard
<https://www.python.org/dev/peps/pep-0008/>`_.
Slightly more than 80 characters can sometimes be tolerated if
increased line width increases readability.


Unit tests
----------

After any change to mpnum, it should be verified that the test suite
runs without any errors.

A short set of tests takes less than 30 seconds and is invoked with one of

.. code::

   python -m pytest
   python setup.py test

An intermediate set of tests, which takes about 2 minutes to run, is
executed automatically for every commit on GitHub via `Travis
<https://travis-ci.org/dseuss/mpnum>`_ continuous integration.
It can be run locally via

.. code::

   python -m pytest -m "not verylong"
   bash tests/travis.sh

A long set of tests takes about 30 minutes and is invoked with

.. code::

   python -m pytest -m 1

Unit tests are implemented using `pytest
<http://pytest.org/>`_.
Every addition to mpnum should be accompanied by corresponding unit tests.
Make sure to use the right pytest-mark for each test. The intermediate and
long running tests should be marked with the 'long' and 'verylong' pytest
mark, respectively.


Test coverage
-------------

Code not covered by unit tests can be detected with `pytest-cov
<https://pypi.python.org/pypi/pytest-cov>`_. A HTML coverage report
can be generated using

.. code::

   python -m pytest --cov-report term --cov-report html --cov=mpnum

Afterwards, the HTML coverage report is available in
:code:`htmlcov/index.html`.


Building the documentation
--------------------------

The HTML documentation uses `Sphinx <http://www.sphinx-doc.org/>`_. On
Linux/macOS, it can be built with a simple

.. code::

   make -C docs html

or

.. code::

   cd docs; make html

After the build, the HTML documentation is available at
:code:`docs/_build/html/index.html`.

`sphinx-autobuild <https://pypi.python.org/pypi/sphinx-autobuild>`_
can be used to rebuild HTML documentation automatically anytime a
source file is changed::

  pip install sphinx-autobuild
  make -C docs livehtml

On Windows, :code:`docs/make.bat` may be useful. For more information,
see the `Sphinx tutorial
<http://www.sphinx-doc.org/en/stable/tutorial.html>`_.
