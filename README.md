pyvar
=====
A simple python package for VAR models.

Installation
------------

Install the required dependencies using ``pip``::

    pip install -r requirements.txt

Pinned dependency versions
--------------------------

The test-suite is verified against the versions listed below.  Newer minor
releases may also work but these versions are known to be compatible::

    numpy>=2.3,<2.4
    scipy>=1.16,<1.17
    pandas>=2.3,<2.4
    sympy>=1.14,<1.15
    statsmodels>=0.14,<0.15
    tqdm>=4.67,<4.68
    nose>=1.3,<1.4

You can then install ``pyvar`` in the usual way::

    python setup.py install

Running Tests
-------------

Tests are written using ``nose``.  After installing the requirements you can
run the test-suite with::

    nosetests

``pytest`` will also work since ``nose`` style tests are automatically
discovered.
