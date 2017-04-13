

.. _mpnum-development:

Development
===========


This section contains information for anyone who wishes to modify
mpnum. Contributions and pull requests for mpnum are very welcome.



Code style
----------

All warnings reported by `flake8
<https://pypi.python.org/pypi/flake8>`_ should be fixed::

  python -m flake8 .

Slightly more than 80 characters can sometimes be tolerated if
reformatting would be cumbersome.


Automated unit tests
--------------------

After any change to mpnum, it should be verified that automated tests
succeed.

A short set of tests takes only 15 seconds and is invoked with one of

.. code::

   python -m pytest
   python setup.py test

An intermediate set of tests takes about 2 minutes to run, is executed
automatically on Travis and is invoked with one of

.. code::

   python -m pytest -m "not verylong"
   bash tests/travis.sh

A long set of tests takes about 30 minutes and is invoked with

.. code::

   python -m pytest -m 1

Unit tests are implemented with `pytest
<http://pytest.org/>`_. Additions to mpnum should always be
accompanied by unit tests.


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
Linux, it can be built with a simple

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
