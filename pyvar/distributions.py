import numpy as np
from scipy.stats import invwishart as InvWishart
from scipy.stats import matrix_normal

class NormInvWishart(object):

    def _check_parameters(self, mu, omega):
        pass

    def __init__(self, μ, Ω, Ψ, ν):

        self.iw = InvWishart(scale=Ψ, df=int(ν))
        self.μ = μ
        self.Ω = Ω
        self.inv_omega = np.linalg.inv(Ω)
        self.logdet_inv_omega = np.log(np.linalg.det(self.inv_omega))
        

        self.r, self.c = μ.shape
        self.n = self.r*self.c


    def draw(self):
        SIGMA = self.iw.rvs()
        BETA = matrix_normal.rvs(self.μ, self.inv_omega, SIGMA)
        return BETA, SIGMA

    def rvs(self, ndraw):
        betas = np.zeros((ndraw, self.r, self.c))
        sigmas = np.zeros((ndraw, self.iw.dim, self.iw.dim))
        from tqdm import tqdm
        for i in tqdm(range(ndraw)):
            b, s = self.draw()
            betas[i,:,:] = b
            sigmas[i,:,:] = s

        return betas, sigmas

    def logpdf(self, β, Σ):
        β = np.ravel(β, order='F')

        Σ_I = np.linalg.inv(Σ)
        SIG_I = np.kron(Σ_I, self.Ω)

        logdet = (self.Ω.shape[0]*np.log(np.linalg.det(Σ))
                  + Σ.shape[0]*self.logdet_inv_omega)

        z = β-self.μ
        lpdf = -self.n/2.0*np.log(2*np.pi) - 0.5*logdet \
               -0.5*(z).T.dot(SIG_I).dot(z)
        return (self.iw.logpdf(Σ) + lpdf)

