from __future__ import division

import numpy as np
import pandas as p

from numpy.testing import assert_equal, assert_almost_equal

from unittest import TestCase

from varprior import SimsZhaSVARPrior

class TestSZ(TestCase):

    def test_sz_prior(self):
        yy = p.read_csv('/mq/home/m1eph00/projects/var-smc/var_smc_working/lib_var/sz_2008_joe_data.csv')

        hyper = np.ones((7, )) 
        hyper[3] = 1.2
        hyper[5] = 0.1
        
        sz = SimsZhaSVARPrior(yy, hyper, p=5)

        cholAF = np.linalg.cholesky(sz.sigma)

        truth = np.loadtxt('AFsigma.txt')[:54]
        assert_almost_equal(cholAF, truth)
