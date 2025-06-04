import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from unittest import TestCase

from pyvar.distributions import NormInvWishart
from scipy.stats import invwishart, matrix_normal


class TestNormInvWishart(TestCase):

    def setUp(self):
        self.mu = np.zeros((2, 2))
        self.Omega = np.array([[1.0, 0.1],
                               [0.1, 1.0]])
        self.Psi = np.array([[1.0, 0.2],
                             [0.2, 1.0]])
        self.nu = 5
        self.niw = NormInvWishart(self.mu, self.Omega, self.Psi, self.nu)

    def test_draw_shapes(self):
        beta, sigma = self.niw.draw()
        assert_equal(beta.shape, self.mu.shape)
        assert_equal(sigma.shape, self.Psi.shape)

    def test_rvs_shapes(self):
        betas, sigmas = self.niw.rvs(ndraw=5)
        assert_equal(betas.shape, (5,) + self.mu.shape)
        assert_equal(sigmas.shape, (5,) + self.Psi.shape)

    def test_logpdf_matches_scipy(self):
        beta = np.array([[0.2, -0.1],
                         [0.1, 0.3]])
        sigma = np.array([[1.2, 0.1],
                          [0.1, 1.3]])
        # flatten mean to work with current implementation
        self.niw.μ = self.niw.μ.ravel(order='F')

        got = self.niw.logpdf(beta, sigma)
        expected = (invwishart(df=self.nu, scale=self.Psi).logpdf(sigma) +
                    matrix_normal.logpdf(beta, mean=self.mu,
                                         rowcov=np.linalg.inv(self.Omega),
                                         colcov=sigma))
        assert_almost_equal(got, expected)

