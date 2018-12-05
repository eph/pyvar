import numpy as np
import scipy.stats as rv
from scipy.stats import chi2
from scipy.special import multigammaln

class InvWishart(object):

    def _check_parameters(self, psi, nu):
        p = psi.shape[0]
        if p is not psi.shape[1]:
            raise ValueError('psi must be a square matrix!')
        if nu < p - 1:
            raise ValueError('nu must be greater than p - 1!')
        return p

    def __init__(self, psi, nu):
        r"""
        This creates a frozen inverse wishart distribution.
        """
        self.p = self._check_parameters(psi, nu)
        self.psi = psi
        self.nu = nu
        self.inv_psi = np.linalg.inv(psi)
        self.chol_inv_psi = np.linalg.cholesky(self.inv_psi)

        # constants
        self.logdetpsi = np.log(np.linalg.det(self.psi))
        self.gamma_const = multigammaln(self.nu/2,self.p)

        self.lik_const = ( -(self.nu*self.p)/2.0 * np.log(2)
                           - self.gamma_const
                           + self.nu/2. * self.logdetpsi )
        
    def rvs(self, ndraw=1):
        """This all could be optimized."""
        return np.array([self.draw() for _ in range(ndraw)])


    def draw(self):
        """A single draw"""
        Z = np.dot(self.chol_inv_psi,
                   rv.norm.rvs(size=(self.p, self.nu)))
        W = np.dot(Z, Z.T)
        IW = np.linalg.inv(W)
        return IW

    def logpdf(self, x):
        
        logdetx = np.log(np.linalg.det(x))

        lpdf = self.lik_const - (self.nu + self.p + 1)/2.0*logdetx \
               - 0.5*np.trace(np.dot(self.psi, np.linalg.inv(x)))

        return lpdf


class NormInvWishart(object):

    def _check_parameters(self, mu, omega):
        pass

    def __init__(self, mu, omega, psi, nu):

        self.iw = InvWishart(psi, nu)
        self.mu = mu
        self.omega = omega
        self.inv_omega = np.linalg.inv(omega)
        self.logdet_inv_omega = np.log(np.linalg.det(self.inv_omega))
        
        self.r, self.c = mu.shape
        self.mu = np.ravel(self.mu,order='F')

        self.n = mu.size


    def draw(self):
        SIGMA = self.iw.draw()
        mvn_covar = np.kron(SIGMA, self.inv_omega)

        BETA = np.random.multivariate_normal(self.mu, mvn_covar)
        BETA = np.reshape(BETA, (self.r, self.c),order='F')
        return BETA, SIGMA

    def rvs(self, ndraw):
        betas = np.zeros((ndraw, self.r, self.c))
        sigmas = np.zeros((ndraw, self.iw.p, self.iw.p))
        from tqdm import tqdm
        for i in tqdm(range(ndraw)):
            b, s = self.draw()
            betas[i,:,:] = b
            sigmas[i,:,:] = s

        return betas, sigmas

    def logpdf(self, BETA, SIGMA):
        BETA = np.ravel(BETA, order='F')

        SIGMA_I = np.linalg.inv(SIGMA)
        #mvn_covar = np.kron(SIGMA, self.inv_omega)

        SIG_I = np.kron(SIGMA_I, self.omega) #np.linalg.inv(mvn_covar)
        
        #_, logdet = np.linalg.slogdet(mvn_covar)
        logdet = (self.omega.shape[0]*np.log(np.linalg.det(SIGMA))
                  + SIGMA.shape[0]*self.logdet_inv_omega)

        z = BETA-self.mu
        lpdf = -self.n/2.0*np.log(2*np.pi) - 0.5*logdet \
               -0.5*(z).T.dot(SIG_I).dot(z)
        return (self.iw.logpdf(SIGMA) + lpdf)

    def logpdf_old(self, BETA, SIGMA):
        BETA = np.ravel(BETA, order='F')

        SIGMA_I = np.linalg.inv(SIGMA)
        mvn_covar = np.kron(SIGMA, self.inv_omega)

        SIG_I = np.linalg.inv(mvn_covar)
        
        _, logdet = np.linalg.slogdet(mvn_covar)
        #logdet = (self.omega.shape[0]*np.log(np.det(SIGMA))
        #          + SIGMA.shape[0]*self.logdet_inv_omega)

        
        lpdf = -self.n/2.0*np.log(2*np.pi) - 0.5*logdet \
               -0.5*(BETA-self.mu).T.dot(SIG_I).dot(BETA-self.mu)
        return (self.iw.logpdf(SIGMA) + lpdf)
