from __future__ import division

import numpy as np
import pandas as p

from numpy.testing import assert_equal, assert_almost_equal
import pytest

from unittest import TestCase

from pyvar import SimsZhaSVARPrior

import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

class TestSZ(TestCase):

    @pytest.mark.xfail(reason="Sims-Zha prior is numerically sensitive")
    def test_sz_prior(self):
        yy = p.read_csv(os.path.join(dir_path,'sz_2008_joe_data.csv'))

        hyper = np.ones((7, )) 
        hyper[3] = 1.2
        hyper[5] = 0.1
        
        sz = SimsZhaSVARPrior(yy, hyper, p=5)

        cholAF = np.linalg.cholesky(sz.sigma)

        truth = np.loadtxt(os.path.join(dir_path,'AFsigma.txt'))[:54]
        assert_almost_equal(cholAF, truth, decimal=2)
