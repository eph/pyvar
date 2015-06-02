import setuptools
import numpy.distutils.core

if __name__ == "__main__":
    numpy.distutils.core.setup(
        name='pyvar',
        version='0.0.1',
        platforms='linux',
        packages=['pyvar'], 
        test_suite='nose.collector',
        tests_require=['nose'])
