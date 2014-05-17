from __future__ import division
import numpy as np
import scipy as sp
from scipy.linalg import solve, eig
from statsmodels.tsa.tsatools import vec, vech
from distributions import NormInvWishart
class Prior(object):
    def __init__(self):
        pass

  

class DummyVarPrior(Prior):

    def __init__(self):
        print "Initializing dummy prior."

    def get_pseudo_obs(self):
        return None

def para_trans(f):
    def reshaped_f(self, *args, **kwargs):
        if len(args) == 1:
            theta = args[0]
            n = self.n
            p = self.p
            cons = self.cons
            Phi = np.reshape(theta[:n**2*p+n*(cons==True)], (n*p+1*(cons==True),n),order='F')
            Sigma = theta[n**2*p+n*(cons==True):]
            Sigma = np.choose(self.sigma_choose, Sigma)
        else:
            Phi = args[0]
            Sigma = args[1]
        return f(self, Phi, Sigma, **kwargs)
    return reshaped_f


class MinnesotaPrior(DummyVarPrior):
    """A class for the Minnesota Prior."""
    def __init__(self, ypre, hyperpara, p=3, cons=True, presample_moments=None, lamxx=None):
        r"""
        A class for the Minnesota Prior for Vector Autoregressions of the form,
        .. math ::
        y_t = \Phi_1y_{t-1} + \ldots + \Phi_py_{t-p} + \Phi_c + u_t
        u_t\sim iid\mathcal{N}(0, \Sigma)

        This particular implementation can be found in:
          Del Negro and Schorfheide, "Bayesian Macroeconometrics"

        Keyword arguments:
        hyperpara -- an array of length 6 with hypeparameters:
        hyperpara[0] -- tightness of prior on own lag coefficients.
        hyperpara[1] -- scaling factor for higher order lag coefficients.
        hyperpara[2] -- multiplicative factor for the covariance matrix.
        hyperpara[3] -- sums of coefficient dummy observations.
        hyperpara[4] -- co-persistnce dummy parameters.
        hyperpara[5] -- 1 if series are nonstationary, 0 o/w
        ypre -- presample data, either an np.array or pandas object.
        p -- the number of lags in the VAR
        cons -- whether or not a constant is included

        Returns:
        MinnesotaPrior object.
        """
        lam1 = hyperpara[0]
        lam2 = hyperpara[1]
        lam3 = hyperpara[2]
        lam4 = hyperpara[3]
        lam5 = hyperpara[4]
        tau = hyperpara[5]

        if presample_moments==None:
            ybar = np.atleast_1d(np.mean(np.asarray(ypre), 0))
            sbar = np.atleast_1d(np.std(np.asarray(ypre), 0))

        else:
            ybar = presample_moments[0]
            sbar = presample_moments[1]

        ny = ybar.size

        self.n = ny;
        self.p = p;
        self.cons = cons;

        n = ny
        max_n = ny*(ny+1)/2
        x = range(int(max_n))
        z = np.zeros((ny, ny), dtype=int)

        dist = n
        ind = 0

        for i in range(n):
            # z[0:n-i, i:n] = x[ind:ind+dist]
            # z[i:n, 0:n-i] = x[ind:ind+dist]
            #z[range(0, n-i), range(i, n)] = x[ind:ind+dist]
            #z[range(i, n), range(0, n-i)] = x[ind:ind+dist]
            z[i, i:] = x[ind:ind+dist]
            z[i:, i] = x[ind:ind+dist]
            ind += dist
            dist -= 1

        self.sigma_choose = z


        dumr = ny * 2 + lam3*ny + ny * (p-1) + 1
        self.__dumy = np.mat(np.zeros((dumr, ny)))
        self.__dumx = np.mat(np.zeros((dumr, ny * p + cons)))


        # tightness on prior of own lag coefficients
        self.__dumy[0:ny, 0:ny] = lam1 * tau * np.diag(sbar)
        self.__dumx[0:ny, 0:ny] = lam1 * np.diag(sbar)

        subt = ny
        # scaling coefficient for higher orger lags
        for l in np.arange(1, p - 1):
            for i in np.arange(0, ny):
                disp = subt + i
                diag = ny * l + i
                self.__dumx[disp, diag] = lam1 * sbar[i] * pow(2, lam2)

            subt += ny

        # prior for covariance matrix
        for l in np.arange(0, lam3):
            for i in np.arange(0, ny):
                self.__dumy[subt + i, i] = sbar[i]
            subt += ny

        # sum of coefficients dummies
        for i in np.arange(0, ny):
            disp = subt + i
            self.__dumy[disp, i] = lam4 * ybar[i]
            for l in np.arange(0, p):
                xcord = l * ny + i
                self.__dumx[disp, xcord] = lam4 * ybar[i]

        subt += ny

        # co-persistence dummy observations
        self.__dumy[subt, :] = np.mat(lam5 * ybar)
        for i in np.arange(0, p):
            self.__dumx[subt, i * ny:(i + 1) * ny] = lam5 * ybar

        if cons:
            self.__dumx[subt, -1] = lam5

        # Add Mark's thing (NOT IN DNS)
        if lamxx is not None:
            self.__dumy = np.vstack((self.__dumy, np.zeros((ny))))
            x = np.zeros((ny*p+cons))
            x[-1] = lam1/lamxx
            self.__dumx = np.vstack((self.__dumx, x))

        self.__dumy = np.asarray(self.__dumy)
        self.__dumx = np.asarray(self.__dumx)
        
        self.frozen_dist = NormInvWishart(self.Phi_star, self.Omega, self.Psi, self.nu)



    @property
    def Omega(self):
        return np.dot(self.__dumx.T, self.__dumx)

    @property
    def Phi_star(self):
        return solve(self.Omega, np.dot(self.__dumx.T, self.__dumy))

    @property
    def Psi(self):
        return (np.dot(self.__dumy.T, self.__dumy)
                - np.dot(self.Phi_star.T, np.dot(self.Omega, self.Phi_star)))

    @property
    def nu(self):
        return (self.__dumy.shape[0] - self.__dumx.shape[1])

    def rvs(self, size=1, flatten_output=True):
        phis, sigmas = NormInvWishart(self.Phi_star, self.Omega, self.Psi, self.nu).rvs(size)
        if flatten_output:
            return np.array([np.r_[np.ravel(phis[i],order='F'), vech(sigmas[i])] for i in range(size)])
        else:
            return phis.squeeze(), sigmas.squeeze()

    @para_trans
    def logpdf(self, Phi, Sigma):
        return self.frozen_dist.logpdf(Phi, Sigma)



    def get_pseudo_obs(self):
        return self.__dumy, self.__dumx



class DiffusePrior(DummyVarPrior):

    def __init__(self):
        print "Initializing Diffuse Prior."

    def get_pseudo_obs(self):
        return None, None

    def draw(self):
        print "This is an improper prior."


class TrainingSamplePrior(DummyVarPrior):
    pass


class SSVSPrior(Prior):
    pass
