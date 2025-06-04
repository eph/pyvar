import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from unittest import TestCase

from pyvar import VAR, BayesianVAR
from pyvar.varprior import DummyVarPrior


class SimplePrior(DummyVarPrior):
    def __init__(self, ny, p, cons=False):
        super().__init__(ny, p, cons)


class TestVAR(TestCase):
    def setUp(self):
        self.A = np.array([[0.5, 0.1], [0.2, 0.4]])
        self.Sigma = 0.1 * np.eye(2)
        T = 8
        data = np.zeros((T, 2))
        data[0] = np.array([1.0, 0.0])
        for t in range(1, T):
            data[t] = data[t-1].dot(self.A)
        self.data = data
        self.var = VAR(ny=2, p=1, cons=False, data=self.data)
        self.prior = SimplePrior(2, 1, False)
        self.bvar = BayesianVAR(self.prior, self.data)

    def test_companion_form(self):
        CC, TT, RR = self.var.companion_form(self.A)
        assert_almost_equal(TT, self.A.T)
        assert_almost_equal(CC, np.zeros(2))
        assert_almost_equal(RR, np.eye(2))

    def test_is_stationary(self):
        unstable = np.array([[1.1, 0.0], [0.0, 0.9]])
        assert self.var.is_stationary(self.A)
        assert not self.var.is_stationary(unstable)

    def test_mle(self):
        beta, S = self.var.mle()
        assert_almost_equal(beta, self.A)
        assert_almost_equal(S, np.zeros((2, 2)))

    def test_loglik(self):
        ll = self.bvar.loglik(self.A, self.Sigma)
        T = self.data.shape[0] - 1
        manual = -T * 2 / 2 * np.log(2 * np.pi) - T / 2 * np.log(np.linalg.det(self.Sigma))
        assert_almost_equal(ll, manual)

    def test_forecast_shape(self):
        fcst = self.bvar.forecast(self.A, self.Sigma, h=3)
        assert_equal(fcst.shape, (3, 2))

