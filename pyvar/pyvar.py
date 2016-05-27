from __future__ import division
import numpy as np
import scipy as sp
import numpy.matlib as M
import sys

#from var_data import var_data, real_time_dataset
from .mcmc import MCMC
from .varprior import DiffusePrior, MinnesotaPrior, Prior, para_trans_general, to_reduced_form
from statsmodels.tsa.tsatools import vec, vech
from .forecast_evaluation import PredictiveDensity
import matplotlib.pyplot as plt
from .distributions import NormInvWishart
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant
from scipy.special import gammaln

# class MSVAR(object):

#     def __init__(self, ny=3, p=5, cons=True, ns_mu=1, ns_var=2):

#         self.short_names = []
#         self.tex_names = []

#         Astr = "A[s={:d}][{:d},{:d}]"
#         Atex = r"A\left(s_{\mu}={:d})_{{0, {:d}{:d}}}\right)"
#         self.cols = [[] for _ in range(ny)]
        
#         for s in range(ns_mu):
#             for j in range(ny):
#                 lena = j+1
#                 for i in range(lena):
#                     self.short_names.append(Astr.format(s, i, j))
#                     self.cols[]
#                     self.tex_names.append(Atex.format(s, i, j))

#         Fstr = r"Fs_{\mu})[{:d},{:d}]"

class VAR(object):
    """
    A class for Vector Autoregressions of the form.
    .. math ::
    y_t = \Phi_1y_{t-1} + \ldots + \Phi_py_{t-p} + \Theta X_t + \Phi_c + u_t
    u_t\sim iid\mathcal{N}(0, \Sigma)
    """

    def __init__(self, ny=3, p=4, cons=True):
        """
        Initializes the VAR.

        Keyword Arguments:
        ny -- The number of observables in the VAR.
        p -- The number of lags in the VAR.
        cons -- If a constant is included.

        Returns:
        VAR object.
        """
        self._ny, self._p, self._cons = ny, p, cons
        self.n = ny
        self.p = p
        self.cons = cons

    @staticmethod
    def simulate_data(phi, sigma, cons=True, n=100):
        """
        Takes a given VAR parameterization and simulates data, starting at the implied unconditional mean.

        Keyword Arguments:
        phi -- A matrix ..math:: k\times n where ..math::k=np+1 or ..math::k=np depending on whether or not a constant is included.  The matrix is populated as
        .. math ::
        \Phi = \left[\Phi_1, \ldots, \Phi_p, \Phi_c\right]'
        #sigma -- vech(\Sigma), an ..math::n(n+1)/2 array containing the unique elements of \Sigma, the covariance matrix of the reduced form shocks.
        sigma -- \Sigma, a symmetric ..math::n^2 covariance matrix of reducded form shocks.

        >>> y = VAR.simulate_data(np.array((0.95)), np.array((0)))
        """
        phi = np.asarray(phi, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        ny = sigma.shape[0]



        if cons:
            p = (phi.shape[0] - 1) / ny
        else:
            p = phi.shape[0] / ny
            phi = np.vstack((phi, np.mat(np.zeros(1, ny))))

        if p > 0:
            coef_sum = phi[:, -1].T * M.repmat(M.eye(ny), p, 1)
        else:
            coef_sum = 0

        muy = np.linalg.solve(M.eye(ny) - coef_sum, phi[:, -1].T)

        y = var_data.empty(ny, n)
        y.series[0:p, :] = muy.T

        x = y.mlag(p, cons)
        for i in np.arange(p, n):
            y.series[i, :] = (x[i, :] * phi
                              + np.mat(randn(np.zeros(ny), sigma)))

            if i < n - 1:
                x[i+1, cons:(cons+ny*(p-1))] = x[i, (cons+ny):]
                x[i+1, (cons+ny*(p-1)):] = y.series[i, :]

        return y


    def __repr__(self):
        return "A VAR with %d observables, %d lags" % (self._ny, self._p)


class BayesianVAR(VAR):
    """A class for Bayesian VARs."""


    def __init__(self, prior, y):
        """Initialization."""
        self._prior = prior
        self.data = y
        #self.sigma_choose = prior.sigma_choose
        self.para_trans = prior.para_trans
        self.reduced_form = prior.reduced_form
        super(BayesianVAR, self).__init__(prior.ny, prior.p, prior.cons)
        
        self.xest, self.yest = lagmat(y, maxlag=self._p, trim="both", original="sep")
        if self._cons == True:
            self.xest = add_constant(self.xest, prepend=False)
            
    # Temporary fix -- need to inherit from prior 
    # def para_trans(self,*args,**kwargs):
    #     if len(args) == 1:
    #         theta = args[0]
    #         n = self.n
    #         p = self.p
    #         cons = self.cons
    #         Phi = np.reshape(theta[:n**2*p+n*(cons==True)], (n*p+1*(cons==True),n),order='F')
    #         Sigma = theta[n**2*p+n*(cons==True):]
    #         Sigma = np.choose(self.sigma_choose, Sigma)
    #     else:
    #         Phi = args[0]
    #         Sigma = args[1]
    #     return Phi, Sigma

    def sample(self, nsim=1000, y=None,flatten_output=False):

        if y is None:
            y = self.data

        ydum, xdum = self._prior.get_pseudo_obs()
        xest, yest = lagmat(y, maxlag=self._p, trim="both", original="sep")

        if self._cons is True:
            xest = add_constant(xest, prepend=False)

        if not ydum == None:
            yest = np.vstack((ydum, yest))
            xest = np.vstack((xdum, xest))


        # This is just a random initialization point....
        phihatT = np.linalg.solve(xest.T.dot(xest), xest.T.dot(yest))
        S = (yest - xest.dot(phihatT)).T.dot(yest - xest.dot(phihatT))
        nu = yest.shape[0] - self._p * self._ny - self._cons*1
        omega = xest.T.dot(xest)
        muphi = phihatT

        phis, sigmas = NormInvWishart(phihatT, omega, S, nu).rvs(nsim)

        if flatten_output:
            return np.array([np.r_[phis[i].flatten(order='F'), vech(sigmas[i])] for i in range(nsim)])
        else:
            return phis, sigmas


        return phis, sigmas

    def get_xy(self, y=None):
        if y is None:
            y = self.data

        ydum, xdum = self._prior.get_pseudo_obs()
        xest, yest = lagmat(y, maxlag=self._p, trim="both", original="sep")

        if self._cons is True:
            xest = add_constant(xest, prepend=False)

        if not ydum == None:
            yest = np.vstack((ydum, yest))
            xest = np.vstack((xdum, xest))

        return yest, xest

    @para_trans_general     #@to_reduced_form
    def reduced_form_irf(self, phi, sigma, h=8): 
        pass
        
    @para_trans_general     #@to_reduced_form
    def pred_density(self, phi, sigma, y=None, h=8, impact_mat=None):
        """Simulates predictive density..."""

        if y is None:
            #xest, yest = self.xest, self.yest
            xest = lagmat(self.data, maxlag=self._p, trim='backward')
            xest = add_constant(xest, prepend=False)
            xest = xest[-self._p]
        else:
            xest, yest = lagmat(y, maxlag=self._p, trim="both", original="sep")
            if self._cons is True:
                xest = add_constant(xest, prepend=False)

        x = xest

        nx = x.shape[0]
        y = np.zeros((h, self._ny))

        if impact_mat==None:
            impact_mat = np.ones_like(y)

        impact_mat = np.squeeze(impact_mat)

        for i in range(0, h):
            y[i, :] = x.dot(phi) + impact_mat[i] * np.random.multivariate_normal(np.zeros(self._ny), sigma)
            if self._p > 0:
                x[ :(self._ny * (self._p - 1))] = x[(self._ny):(nx-self._cons)]
                x[ (self._ny * (self._p - 1)):-1] = y[i, :]
            else:
                x = y[i, :]
        return y

    def mle(self, y=None):
        """MLE."""
        if y is None:
            y = self.data

        x, y = lagmat(y, maxlag=self._p, trim="both", original="sep")

        if self._cons is True:
            x = add_constant(x, prepend=False)

        beta = np.linalg.solve(x.T.dot(x), x.T.dot(y))
        S = (y - x.dot(beta)).T.dot(y - x.dot(beta))/y.shape[0]

        return (beta, S)


    def logmdd(self, y=None):
        """Computes the log of marginal data density."""
        # if not isinstance(self._prior, "DummyVarPrior"):
        #     print "Can only be computed analytically for DummyVarPriors."
        #     return

        if y is None:
            y = self.data


        ydum, xdum = self._prior.get_pseudo_obs()
        xest, yest = lagmat(y, maxlag=self._p, trim="both", original="sep")

        T     = yest.shape[0]
        Tstar = ydum.shape[0]
        Tbar  = Tstar + T
        self._n = self._ny
        if self._cons is True:
            xest = add_constant(xest, prepend=False)

        yest = np.vstack((ydum, yest))
        xest = np.vstack((xdum, xest))

        logdetx    = np.log(np.linalg.det(xest.T.dot(xest)))
        logdetxdum = np.log(np.linalg.det(xdum.T.dot(xdum)))

        phihatT = np.linalg.solve(xest.T.dot(xest), xest.T.dot(yest))
        S = (yest - xest.dot(phihatT)).T.dot(yest - xest.dot(phihatT))
        logdets = np.log(np.linalg.det(S))

        phidumT = np.linalg.solve(xdum.T.dot(xdum), xdum.T.dot(ydum))
        Sdum = (ydum - xdum.dot(phidumT)).T.dot (ydum - xdum.dot(phidumT))
        logdetsdum = np.log(np.linalg.det(Sdum))
        k = self._n * self._p + 1*(self._cons is True)

        kap = 0
        kapdum = 0
        for i in np.arange(0, self._n):
            kap += gammaln( (Tbar - k - i)/2.0 )
            kapdum += gammaln( (Tstar - k - i)/2.0 )

        lnpy = (-self._n*T/2.0*np.log(np.pi)
                -self._n/2.0*logdetx -(Tbar - k)/2.0*logdets
                #+self._n*(Tbar - k)/2.0*np.log(2.0)
                + kap - kapdum
                + self._n/2.0*logdetxdum + (Tstar - k)/2.0*logdetsdum
                #-self._n*(Tstar - k)/2.0*np.log(2.0) )
                )
        return lnpy

    @to_reduced_form
    def loglik(self, Phi, Sigma, y=None):
        if y is None:
            xest, yest = self.xest, self.yest
        else:
            xest, yest = lagmat(y, maxlag=self._p, trim="both", original="sep")
            if self._cons is True:
                xest = add_constant(xest, prepend=False)

        T,n = yest.shape

        try:
            np.linalg.cholesky(Sigma)
        except:
            return -10000000000000

        ep = yest - xest.dot(Phi)
        Sigma_inv = np.linalg.inv(Sigma)
        lik = (-T*n/2*np.log(2*np.pi)
               -T/2*np.log(np.linalg.det(Sigma))
               -0.5*(Sigma_inv.dot(ep.T).dot(ep)).trace() )
        return lik

    @to_reduced_form
    def lik(self, Phi, Sigma, y=None):
        if y is None:
            xest, yest = self.xest, self.yest
        else:
            xest, yest = lagmat(y, maxlag=self._p, trim="both", original="sep")
            if self._cons is True:
                xest = add_constant(xest, prepend=False)


        T,n = yest.shape

        ep = yest - xest.dot(Phi)
        Sigma_inv = np.linalg.inv(Sigma)


        lik = ( (2*np.pi)**(-T*n/2)
                * np.linalg.det(Sigma)**(-T/2)
                * np.exp(-0.5*(Sigma_inv.dot(ep.T).dot(ep)).trace())
                )
        return lik

    @para_trans_general
    def logpost(self,Phi,Sigma,y=None):
        return self.loglik(Phi,Sigma,y) + self._prior.logpdf(Phi,Sigma)


    @to_reduced_form
    def forecast(self,Phi,Sigma,h=12,y=None,append=True):
        if y is None:
            y = self.data

        y_base = y
        y0 = np.ones((self._ny*self._p+self._cons))
        y0[:self._ny*self._p] = lagmat(y_base,self._p,'none')[-self._p]

        yfcst = np.zeros((h, self._ny))

        for i in range(h):
            yfcst[i] = np.dot(y0, Phi)
            y = np.vstack((y, yfcst[i]))
            if self._cons:
                y0[:-self._cons] = lagmat(y,self._p,'none')[-self._p]
            else:
                y0 = lagmat(y,self._p,'none')[-self._p]
        return yfcst

    @para_trans_general
    def to_state_space(self, Phi, Sigma):

        print("only works for p=1")

        from scipy.linalg import block_diag
        n = self._n
        TT = block_diag(*[Phi[i*n:(i+1)*n, i*n:(i+1)*n].T for i in range(self._p)])
        
        RR = np.zeros(((self._p)*n, n))
        RR[:n, :] = np.eye(n)
        QQ = Sigma

        if self._cons:
            DD = np.linalg.inv(np.eye(n) - TT).dot(Phi[-1, :]).squeeze()
        else:
            DD = np.zeros((self._n))
        ZZ = np.zeros((self._p*n, n)).T
        ZZ[:, :n] = np.eye(n)

        HH = np.zeros((n, n))

        def funcify(x):
            def lam(para):
                return x
            return lam

        TT =funcify(TT)
        RR =funcify(RR)
        QQ =funcify(QQ)

        DD =funcify(DD)
        ZZ =funcify(ZZ)
        HH =funcify(HH)

        
        from dsge.StateSpaceModel import StateSpaceModel

        return StateSpaceModel(self.data, TT, RR, QQ, DD, ZZ, HH, t0=self._p)


        
class ConstrainedVAR(BayesianVAR):

    def __init__(self, prior, y, constraint):

        super(BayesianVAR, self).__init__(prior, y)
        self.contstraint = constraint

    @para_trans_general
    def loglik(self, Phi, Sigma):
        pass
        
        
    
class BayesianSteadyStateVAR(VAR):

    def __init__(self, prior, y):

        # b/c of pickling problems
        self.ntrends = prior.ntrends
        self.cyclical_prior = prior.cyclical_prior

        self.data = y
        self.prior = prior
        D = np.atleast_2d(np.tile(np.arange(y.shape[0]), self.prior.ntrends)).T

        for t in range(self.prior.ntrends):
            D[:, t] = D[:, t]**t

        self.D = D
        self.cyclical_var = BayesianVAR(prior.cyclical_prior, y)

        super(BayesianSteadyStateVAR, self).__init__(prior.n, prior.p, prior.cons)
        
        #self.para_trans = prior.para_trans
        #self.reduced_form = prior.reduced_form

    def para_trans(self, *args, **kwargs):
        if len(args)==1:
            theta = args[0]
            Gamma = np.reshape(theta[-self.ntrends*self.n:], (self.ntrends, self.n),order='F')
            Phi, Sigma = self.cyclical_prior.para_trans(theta[:-self.ntrends*self.n])
        else:
            Phi = args[0]
            Sigma = args[1]
            Gamma = args[2]

        return Phi, Sigma, Gamma


    @para_trans_general
    def forecast(self, Phi, Sigma, Gamma, h=12, y=None, D=None):
        if y is None:
            y = self.data
            D = self.D
        elif D is None:
            D = np.atleast_2d(np.tile(np.arange(y.shape[0]), self.prior.ntrends)).T

            for t in range(self.prior.ntrends):
                D[:, t] = D[:, t]**t

            
        yhat = y - D.dot(Gamma)
        yfcst = self.cyclical_var.forecast(Phi, Sigma, h=h, y=yhat)

        D = np.atleast_2d(np.tile(np.arange(y.shape[0], y.shape[0]+h), self.prior.ntrends)).T

        for t in range(self.prior.ntrends):
            D[:, t] = D[:, t]**t

        
        yfcst = yfcst + D.dot(Gamma)
        return yfcst
    @para_trans_general
    def loglik(self, Phi, Sigma, Gamma, y=None):

        if y is None:
            y = self.data

        yhat = y - self.D.dot(Gamma)
        return self.cyclical_var.loglik(Phi, Sigma, y=yhat)
        
class CompleteModel(BayesianVAR):
    """
    This is a class for Complete Models, i.e., a model and data.
    """

    def __init__(self, bvar, y):
        """
        Initialization full model.
        """
        self.__dict__ = bvar.__dict__   # this might be the best way to do this
        self.y = y


    def loglk(self, para):
        """Returns the log likelihood evaluated at (phi, sigma)."""

        x = self.y.mlag(self._p, self._cons)
        (phi, sigma) = self.get_phi_sigma(para)
        (phi_hat, s_hat) = self.mle()
        T = self._ny - self._p

        llk = (-T / 2 * sigma.det() - 0.5 * (sigma.I * s_hat).trace()
        - 0.5 * (sigma.I * (phi - phi_hat).T * x.T * x * (phi - phi_hat)).trace())

        return llk

    def pred_density(self, parasim, h=8, nthin=1):
        """Simulates from Predictive Density."""
        nsim = parasim.nsim()
        yfcst = np.zeros((nsim, h, self._ny))
        for i in range(0, nsim, nthin):
            parai = parasim.get(i)
            phi = parai["Phi"]
            sigma = parai["Sigma"]
            yfcst[i, ...] = super(CompleteModel, self).pred_density(phi, sigma, self.y, h )

        return yfcst


    def estimate(self, nsim=1000):

        return super(CompleteModel, self).estimate(self.y, nsim)

class ForecastingExercise:

    def __init__(self, rt_data, forecast_model, hmax=8):

        print("Initializing a recursive forecasting exercise with %i sets of estimates." %  rt_data.size())

        self.__nsamp = rt_data.size()
        self.__hmax = hmax
        if not isinstance(forecast_model, list):
            forecast_model = [forecast_model]

        self.__ny = forecast_model[0]._ny
        self.__rt_data = rt_data
        j = 0
        self.__model = []
        for i in range(0, self.__nsamp):
            self.__model.append(CompleteModel(forecast_model[j], rt_data.getVARData(i)))

            if len(forecast_model) > 1:
                j += 1

        print("DONE.")

    def estimate(self, nsim=1000):

        self.__parasim = []
        for i in range(0, self.__nsamp):
            print("Estimating model %i. " % i)
            self.__parasim.append(self.__model[i].estimate(nsim))


    def generate_forecast(self, nthin=1):
        self.__yypred_dens = []

        for i in range(0, self.__nsamp):
            print("Forecasting model %i. " % i)
            yypred = self.__model[i].pred_density(self.__parasim[i], self.__hmax, nthin)
            yypred = PredictiveDensity(yypred, self.__rt_data.getFORData(i).series)
            self.__yypred_dens.append(yypred)


    def evaluate_forecast(self):
        self.__pits = np.zeros((self.__nsamp, self.__hmax, self.__ny))

        for i in range(0, self.__nsamp):
            print("Evaluating model %i. " % i)
            self.__pits[i, ...] = self.__yypred_dens[i].get_unconditional_pits()
            print(self.__pits[i, ...])

    def generate_pit_plot(self, hplot=[1, 4, 8]):
        plt.figure(1)

        i = 0
        for h in range(0, len(hplot)):
            for s in range(0, self.__ny):
                plt.subplot(len(hplot), self.__ny, i + 1)
                N, bins = np.histogram(self.__pits[:, hplot[h] - 1, s], 5, range=(0.0, 1.0))
                N = N / np.sum(N) * 100
                plt.bar(bins[:-1], N,  width=0.2)
                plt.ylim((0, 100))
                plt.xlim((0, 1))
                plt.axhline(y=20)
                i += 1

        plt.show()

    def evaluate_se(self):
        self.__se = np.zeros((self.__nsamp, self.__hmax, self.__ny))
        self.__cse = np.zeros((self.__nsamp, self.__ny, self.__hmax, self.__ny))
        for i in range(0, self.__nsamp):
            print("Evaluating model %i" % i)
            self.__se[i, ...] = self.__yypred_dens[i].get_unconditional_mse()
            self.__cse[i, ...] = self.__yypred_dens[i].get_conditional_mse()

    def get_rmse(self):
        self.__rmse = np.sqrt(np.mean(self.__se, 0))
        return self.__rmse

    def get_crmse(self):
        self.__crmse = np.sqrt(np.mean(self.__cse, 0))
        return self.__crmse

    def generate_rmse_plot(self):
        rmse = self.get_rmse()

        plt.figure(1)
        for i in range(0, self.__ny):
            plt.subplot(1, self.__ny, i + 1)
            plt.plot(np.arange(1, self.__hmax + 1), rmse[:, i])

        plt.show()

    def get_model(self, i):
        """
        Returns the ith model.
        """
        return self.__model[i]

    def get_mcmc(self, i):
        return self.__parasim[i]


class ForecastingPredictiveCheck:

    def __init__(self, estimated_model, estimated_paras):

        self.estimated_model = estimated_model
        self.parasim = estimated_paras


    def generate_trajectories(self, basedir, ntraj):

        parasim = self.parasim[0, ...]

        ind = range(0, parasim["phi"].shape[0], int(parasim.shape[0]/ntraj))

        self.simulated = []
        j = 0
        for i in ind:
            phi = np.mat(mcmcdraws["phi"][i, :, :].T)
            sigma = np.mat(mcmcdraws["sigma"][i, :, :])
            yypred = models.predictive_density(phi, sigma, hmax)

            new_series = self.actuals

from scipy.stats import rv_discrete

class MixtureVAR(object):

    def __init__(self, data_sets, prior, alpha=None):

        if not isinstance(data_sets, list):
            data_sets = [data_sets]

        self.nmix = len(data_sets)

        self.var_models = []
        for dat in data_sets:
            self.var_models.append(BayesianVAR(prior, dat))

        if alpha==None:
            alpha=np.ones((self.nmix))

        alpha = np.asarray(alpha)
        alpha = alpha/np.sum(alpha)

        assert len(self.var_models)==alpha.size
        self.alpha = alpha


        self.prior = prior

        self.mdds = np.asarray(map(lambda x: x.logmdd(), self.var_models))
        self.tot = np.sum(self.mdds)
        self.c = self.tot+100

        self.n = self.var_models[0].n
        self.p = self.var_models[0].p
        self.cons = self.var_models[0].cons
        self.sigma_choose = self.var_models[0]._prior.sigma_choose


    # Temporary fix -- need to inherit from prior 
    def para_trans(self,*args,**kwargs):
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
        return Phi, Sigma




    def log_mdd(self):
        return self.tot

    def rvs(self, size=1, *args, **kwargs):
        modes = np.zeros((size), dtype=int)
        modes[np.random.rand(size)>self.alpha[0]] = 1#self.mix.rvs(size=size)
        return np.asarray([self.var_models[m].sample(nsim=1, *args, **kwargs).flatten() for m in modes])

    @para_trans_general
    def logpost(self, Phi, Sigma):
        post = 0;
        for i in range(self.nmix):
            post += self.alpha[i]*np.exp(self.var_models[i].logpost(Phi, Sigma) + self.tot - self.mdds[i] - self.c)
        return self.c + np.log(post)

    @para_trans_general
    def loglik(self, Phi, Sigma):
        post = 0;
        for i in range(self.nmix):
            post += self.alpha[i]*np.exp(self.var_models[i].loglik(Phi, Sigma) + self.tot - self.mdds[i] - self.c)
        return self.c + np.log(post)

    @para_trans_general
    def logliki(self, Phi, Sigma):
        x = map(lambda x: np.exp(x.logpost(Phi, Sigma) + self.tot - x.logmdd() - self.c), self.var_models)
        x = np.asarray(x).squeeze()
        x = self.c + np.log(self.alpha*x)
        x = np.exp(x - self.logpost(Phi, Sigma))
        return x

    @para_trans_general
    def logprior(self, Phi, Sigma):
        return self.prior.logdens(Phi, Sigma)


    # @para_trans
    # def rwmh(self, proposal, nsim=1000, flatten_output=True):

    #     betas, sigmas = self.var_models[0].sample(nsim)
    #     ms = np.zeros((nsim))

    #     ms[0] = 0               # always start on first mode
    #     for i in range(nsim):

    #         # SIGMA
    #         p0 = proposal.rvs()
    #         SIGMA = np.zeros((n, n))
    #         SIGMA[np.tril_indices((n, -1))] = p0

    #         # BETA
    #         yest, xest = self.var_model[m].get_xy()

    #         phihatT = np.linalg.solve(xest.T.dot(xest), xest.T.dot(yest))
    #         S = (yest - xest * phihatT).T.dot(yest - xest * phihatT)
    #         nu = yest.shape[0] - self._p * self._ny - self._cons*1
    #         omega = xest.T*xest
    #         muphi = phihatT#.flatten()#np.ravel(phihatT)

    #         omega_inv = np.linalg.inv(omega)
    #         mvn_covar = np.kron(SIGMA, self.inv_omega)

    #         BETA = np.random.multivariate_normal(np.ravel(muphi, order='F'), mvn_covar)

    #         lnpy1 = self.var_models[m].logpost(BETA, SIGMA)


if __name__ == "__main__":
    # generate trajectories
    data_set = var_data.read_from_csv(filename="3eqvar.csv", header=True, freq="quarterly", start="1965q2")
    rt_data = real_time_dataset(data_set, estart="1996q1", t0="1983q1")
    unifpr = DiffusePrior()             # initialize uniform prior

    # Minnesota Prior
    lam1 = 5.0
    lam2 = 0.5
    lam3 = 1.0
    lam4 = 1.0
    lam5 = 5.0
    tau = 0
    hyperpara = np.array([lam1, lam2, lam3, lam4, lam5, tau])
    minnpr = MinnesotaPrior(data_set.series[1:16, :], hyperpara, p=4)

    bvar_set = []
    for i in np.arange(0, rt_data.size()):
        presamp = rt_data.getVARData(i).series
        #minnpr = MinnesotaPrior(presamp[1:16, :], hyperpara, p=4)
        bvar_set.append(BayesianVAR(unifpr, ny=3, p=0, cons=True, rw=True))

    MN = ForecastingExercise(rt_data, bvar_set)
    MN.estimate(nsim=1000)
    MN.generate_forecast()
    MN.evaluate_forecast()
    MN.generate_pit_plot()
    MN.evaluate_se()
    rmse = MN.get_rmse()
    crmse = MN.get_crmse()
    print(crmse[0, ...] / rmse) 
    print(crmse[1, ...] / rmse) 
    print(crmse[2, ...] / rmse) 
    print(rmse)
