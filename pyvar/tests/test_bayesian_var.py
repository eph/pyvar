import numpy as np
from numpy.testing import assert_equal
from pyvar import MinnesotaPrior, BayesianVAR


def test_bayesian_var_sample_shapes():
    # small dataset with two variables
    y = np.arange(20, dtype=float).reshape(10, 2)
    prior = MinnesotaPrior(y, np.ones(6), p=1)
    bvar = BayesianVAR(prior, y)

    y_xy, x_xy = bvar.get_xy()
    ydum, xdum = prior.get_pseudo_obs()

    # stacked pseudo observations should appear first
    assert_equal(y_xy[:ydum.shape[0]], ydum)
    assert_equal(x_xy[:xdum.shape[0]], xdum)

    # check shapes after stacking
    T = y.shape[0] - prior.p
    expected_rows = T + ydum.shape[0]
    assert_equal(y_xy.shape, (expected_rows, prior.ny))
    assert_equal(x_xy.shape, (expected_rows, prior.ny * prior.p + prior.cons))

    samples_flat = bvar.sample(nsim=5, flatten_output=True)
    phi_len = (prior.ny * prior.p + prior.cons) * prior.ny
    sigma_len = prior.ny * (prior.ny + 1) // 2
    expected_len = phi_len + sigma_len
    assert_equal(samples_flat.shape, (5, expected_len))

    phis, sigmas = bvar.sample(nsim=5, flatten_output=False)
    assert_equal(phis.shape, (5, prior.ny * prior.p + prior.cons, prior.ny))
    assert_equal(sigmas.shape, (5, prior.ny, prior.ny))
