from __future__ import division
import numpy as np

from scipy.linalg import solve, block_diag
from statsmodels.tsa.tsatools import vech, lagmat
from distributions import NormInvWishart


def para_trans_general(f):
    """
    This is a generic decorator to map a vector theta describing the
    parameters of the VAR to the natural tuple describing it.
    """
    def reshaped_f(self, *args, **kwargs):
        trpara = self.para_trans(*args, **kwargs)
        return f(self, *trpara, **kwargs)
    return reshaped_f


def to_reduced_form(f):
    """
    This is similar to para_trans_general, but it only extracts the
    matrices that matter for likelihood evaluation: (Phi, Sigma).

    """
    def reshaped_f(self, *args, **kwargs):
        Phi, Sigma = self.reduced_form(*args, **kwargs)
        return f(self, Phi, Sigma, **kwargs)
    return reshaped_f


class Prior(object):
    """Container Class for prior objects."""
    def __init__(self):
        pass

    def reduced_form(self, *args, **kwargs):
        return args[0], args[1]

class DummyVarPrior(Prior):
    """"Dummy var prior."""
    def __init__(self):
        print "Initializing dummy prior."

    def get_pseudo_obs(self):
        return None

class DiffusePrior(DummyVarPrior):

    def __init__(self):
        print "Initializing Diffuse Prior."

    def get_pseudo_obs(self):
        return None, None

    def draw(self):
        print "This is an improper prior."

class SimsZhaSVARPrior(Prior):

    def __init__(self, ypre, hyperpara, p=3, cons=True, ar_lp=6, presample_moments=None, restriction='A0:upper_triangle'):
        lam0 = hyperpara[0]
        lam1 = hyperpara[1]
        lam2 = hyperpara[2]
        lam3 = hyperpara[3]
        lam4 = hyperpara[4]
        mu5 = hyperpara[5]
        mu6 = hyperpara[6]
        
        mu0     = 1/lam0
        mu1     = 1/lam1
        mu2     = 1/lam2
        mu3     = -lam3
        mu4     = 1/lam4

        ypre = np.array(ypre)
        T, ny = ypre.shape
        k = ny*p + cons
        
        Uar = np.zeros((T-ar_lp, ny))
        for i in np.arange(ny):
            Xar, Yar = lagmat(ypre[:, i], maxlag=ar_lp, trim='both', original='sep')
            #Xar, Yar = Xar[1:, :], Yar[1:] # not sure why Mark does this
            #betaar = np.linalg.solve(Xar.T.dot(Xar), Xar.T.dot(Yar))
            betaar = np.linalg.inv(Xar.T.dot(Xar)).dot(Xar.T.dot(Yar))
            Uar[:, i] = (Yar - Xar.dot(betaar)).flatten()


        ybar = np.atleast_1d(np.mean(np.asarray(ypre), 0))
        sbar = np.atleast_1d(np.std(np.asarray(Uar), 0, ddof=1)) # np.std = sqrt(sum(x-mu)**2/N)


        #------------------------------------------------------------ 
        # A0 hyperparameters
        #------------------------------------------------------------
        A0 = []
        for j in np.arange(ny):
            A0.append(dict())
            
            A0[j]['mu'] = np.zeros((ny, 1))
            A0[j]['sigma'] = np.zeros((ny, ny))
            np.fill_diagonal(A0[j]['sigma'], (lam0/sbar)**2)


        Aplus = []
        for j in np.arange(ny):
            Yid = np.zeros((k+ny, ny))
            Xid = np.zeros((k+ny, k))
            
            Yid[:ny, :ny] = np.diag( mu0*mu1*mu2/(1**mu3) * sbar )

            for r in np.arange(ny):
                for l in np.arange(p):
                    Xid[ny*l+r, ny*l+r] = mu0*mu1*mu2*sbar[r] / ((l+1)**mu3)

            Yid[-(ny+1):-1, :] = np.diag(mu5*ybar)
            Xid[-(ny+1):-1, :][:, :(k-1)] = np.tile(mu5*np.diag(ybar), p)
            Yid[-1, :] = mu6*ybar
            Xid[-1, :] = np.hstack((np.tile(mu6*ybar, p), mu6));

            sig = np.linalg.inv(Xid.T.dot(Xid))
            
            Aplus.append(dict())

            Aplus[j]['sigma'] = sig.dot(Xid.T.dot(Yid)).dot(A0[j]['sigma']).dot((sig.dot(Xid.T.dot(Yid))).T) + sig#sig
            Aplus[j]['cov'] = sig.dot(Xid.T.dot(Yid)).dot(A0[j]['sigma'])
            Aplus[j]['mu'] = sig.dot(Xid.T.dot(Yid))

        self.A0 = A0
        self.Aplus = Aplus
        self.ny = ny
        self.p = p
        self.cons = cons

        MU = np.zeros(ny**2 + ny**2*p + ny*cons)

        SIGMA = block_diag(*([A0[j]['sigma'] for j in range(ny)] + [Aplus[j]['sigma'] for j in range(ny)]))

        j0 = ny**2
        j1 = 0
        for j in range(ny):
            SIGMA[j0:j0+ny*p+cons, :][:, j1:j1+ny] = Aplus[j]['cov']#Aplus[j]['mu'].dot(np.linalg.inv(A0[j]['sigma']))
            SIGMA[j1:j1+ny, :][:, j0:j0+ny*p+cons] = Aplus[j]['cov'].T#Aplus[j]['mu'].dot(np.linalg.inv(A0[j]['sigma'])).T
            j0 += ny*p+cons                                               
            j1 += ny


        self.restriction = restriction
        
        selmat = np.tril_indices(ny, -1)
        selind = np.reshape(np.arange(ny**2), (ny, ny), order='F')[selmat]
        self.selmat = selmat
        self.selind = selind
        mask = np.ones(MU.size, dtype=bool, order='F')
        print MU.size
        mask[selind] = False

        self.restrictions = dict()
        self.restrictions['A0'] = np.ones((ny, ny), dtype=bool, order='F')
        self.restrictions['A0'][np.tril_indices(ny, -1)] = False

        MU    = MU[mask]        # this needs to be generalized
        cov   = SIGMA[mask, :][:, selind]
        var   = SIGMA[selind, :][:, selind]
        SIGMA = SIGMA[mask, :][:, mask] - cov.dot(np.linalg.inv(var)).dot(cov.T)
        
        self.mu = MU
        self.sigma = SIGMA
        
        from scipy_update import multivariate_normal
        self.n = self.ny
        self.prior = multivariate_normal(self.mu, self.sigma)

    def rvs(self, size=1, flatten=True):
        r = self.prior.rvs(size=size)
        if flatten==False:
            A0s = np.zeros((size, self.n, self.n))
            Apluses = np.zeros((size, self.n*self.p + self.cons, self.n))
            for i in range(size):
                A0s[i][self.restrictions['A0']] = r[i][:self.n*(self.n+1)/2] 
                Apluses[i] = np.reshape(r[i][self.n*(self.n+1)/2:], (self.n*self.p+self.cons, self.n))
            r = A0s, Apluses
        return r

    @para_trans_general
    def logpdf(self, A0, Aplus, *args, **kwargs):
        x = np.r_[A0[0, 0], A0[:2, 1], A0[:3, 2], np.ravel(Aplus, order='F')]
        #x = np.r_[np.ravel(A0[self.restrictions['A0']], order='F'), np.ravel(Aplus,order='F')]
        return self.prior.logpdf(x)
        
    def reduced_form(self, A0, Aplus, *args, **kwargs):
        A0i = np.linalg.inv(A0)
        Phi = Aplus.dot(A0i)
        Sigma = np.linalg.inv(A0.dot(A0.T))
        return Phi, Sigma
        
    @para_trans_general
    def variance_decomposition(self, A0, Aplus, *args, **kwargs):
        ny = self.ny
        QQ = np.zeros((ny, ny))

        
    @para_trans_general
    def structural_irf(self, A0, Aplus, h=10):

        Phi, Sigma = self.reduced_form(A0, Aplus)

        y = np.zeros((self.ny, h, self.ny))
        x = np.zeros((self.ny*self.p + self.cons))
        if self.cons:
            x[-1] = 0
            
        nx = x.shape[0]
        for i in range(self.ny):
            QQ = np.zeros((self.ny))
            QQ[i] = 1.0

            impact = QQ.dot(np.linalg.inv(A0))
            y[i, 0, :] = impact

            for j in range(1, h):

                if self.p > 0:
                    x[ :(self.ny * (self.p - 1))] = x[(self.ny):(nx-self.cons)]
                    x[ (self.ny * (self.p - 1)):-1] = y[i, j-1, :]
                else:
                    x = y[i, j-1, :]

                y[i, j, :] = x.dot(Phi)

        return y
        

    def para_trans(self,*args,**kwargs):
        if len(args) == 1:
            theta = np.asarray(args[0])
            n = self.n
            p = self.p
            cons = self.cons
            A0 = np.zeros((n, n), order='F')
            #A0[self.restrictions['A0']] = theta[:n*(n+1)/2]
            A0[0, 0] = theta[0]
            A0[0:2, 1] = theta[1:3]
            A0[0:3, 2] = theta[3:6]
            Aplus = np.reshape(theta[n*(n+1)/2:], (n*p+cons, n), order='F')
        else:
            A0 = args[0]
            Aplus = args[1]
        return A0, Aplus



import sympy

def taylor_to_var(r_star, rho, alpha_pi, alpha_z, sigma_R, tau_1, tau_2):
    a33 = 1/sigma_R

    a13 = -(1-rho)*alpha_z*a33
    a23 = -(1-rho)*alpha_pi*a33

    A0 = [a13, a23, a33]
    Aplus = [tau_1*a33, tau_2*a33, rho*a33, r_star*a33]

    return A0 + Aplus

r_star, rho, alpha_pi, alpha_z, sigma_R, tau_1, tau_2 = sympy.symbols('r_star, rho, alpha_pi, alpha_z, sigma_R, tau_1, tau_2')
a13, a23, a33, b13, b23, b33, b43 = sympy.symbols('a13, a23, a33, b13, b23, b33, b43')
tlr = taylor_to_var(r_star, rho, alpha_pi, alpha_z, sigma_R, tau_1, tau_2)
vcx = [a13, a23, a33, b13, b23, b33, b43]

eq = []
for i in range(7):
    eq.append(vcx[i] - tlr[i])

inv_taylor = sympy.solve(eq, [r_star, rho, alpha_pi, alpha_z, sigma_R, tau_1, tau_2])[0]
g_inv = sympy.lambdify(vcx, inv_taylor)

from sympy.matrices import Matrix
from scipy_update import multivariate_normal
J = Matrix(7, 7, lambda i, j: sympy.diff(inv_taylor[i], vcx[j]))

Jdet = sympy.lambdify(vcx, J.det())

from collections import OrderedDict
class TaylorRulePrior(SimsZhaSVARPrior):

    def __init__(self, taylor_args, *args, **kwargs):

        # Initialize the baseclass
        super(TaylorRulePrior, self).__init__(*args, **kwargs)

        # r_loc  = ordering[0]
        # pi_loc = ordering[1]
        # y_loc  = ordering[2]

        npara = self.n*(self.n+1)/2 + self.n**2*self.p + self.n*self.cons

        A0_r_start = self.n*(self.n+1)/2 - self.n
        A0_r_end = self.n*(self.n+1)/2
        last_col_A0    = np.arange(A0_r_start, A0_r_end)

        A1_r_start = A0_r_end + (self.n-1)*self.n*(self.p) + (self.n-1)*self.cons
        A1_r_end = A1_r_start + self.n
        last_col_A1 = np.arange(A1_r_start, A1_r_end)
        
        cons_Aplus = npara-1
        r_idx = np.r_[last_col_A0, last_col_A1, cons_Aplus]
        r_idx = np.asarray(r_idx, dtype=int)
        print 'r_idx', r_idx
        #r_idx = np.array([ 3,  4,  5,  14,  15,  16,  17])
        self.r_idx = r_idx
        self.non_r_idx = np.ones((npara,), dtype=bool)
        self.non_r_idx[r_idx] = False

        # drop out r variables
        self.sigma_con = self.sigma[self.non_r_idx, :][:, self.non_r_idx]
        self.mu_con = self.mu[self.non_r_idx]
        self.prior_con = multivariate_normal(self.mu_con, self.sigma_con)        

        from scipy.stats import norm, beta, invgamma

        self.tlr = OrderedDict()
        self.tlr['r_star']=taylor_args.pop('r_star', norm(loc=4/100,scale=0.5/100))
        self.tlr['rho']=taylor_args.pop('rho', norm(loc=0.75, scale=0.1))         
        self.tlr['alpha_pi']=taylor_args.pop('alpha_pi', norm(loc=1.5, scale=0.25))
        self.tlr['alpha_z']=taylor_args.pop('alpha_z',norm(loc=1.0, scale=0.25))
        self.tlr['sigma_R']=taylor_args.pop('sigma_R',invgamma(10))
        self.tlr['tau_1']= taylor_args.pop('tau_1',norm(loc=0, scale=0.25))
        self.tlr['tau_2']= taylor_args.pop('tau_2', norm(loc=0, scale=0.25))
        

        
    def rvs(self, size=1, flatten=True):
        x = super(TaylorRulePrior, self).rvs(size=size, flatten=flatten)
        if len(x.shape) > 1:
            for i in range(x.shape[0]):
                x[i, self.r_idx] = taylor_to_var(*[self.tlr[p].rvs() for p in self.tlr.keys()])
        else:
            x[self.r_idx] = taylor_to_var(*[self.tlr[p].rvs() for p in self.tlr.keys()])
        return x

    @para_trans_general
    def logpdf(self, A0, Aplus):
        x = np.r_[A0[0, 0], A0[:2, 1], A0[:3, 2], np.ravel(Aplus, order='F')]
        retval = self.prior_con.logpdf(x[self.non_r_idx])
        tlr_coeff = g_inv(*x[self.r_idx])
        j = 0
        for p in self.tlr.keys():
            retval += self.tlr[p].logpdf(tlr_coeff[j])
            j+=1
        retval += np.log(np.abs(Jdet(*x[self.r_idx])))
        
        return retval
        
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


        # for picking out sigma
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
            z[i, i:] = x[ind:ind+dist]
            z[i:, i] = x[ind:ind+dist]
            ind += dist
            dist -= 1

        self.sigma_choose = z

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
        phis, sigmas = self.frozen_dist.rvs(size)
        if flatten_output:
            return np.array([np.r_[np.ravel(phis[i],order='F'), vech(sigmas[i])] for i in range(size)])
        else:
            return phis.squeeze(), sigmas.squeeze()

    @para_trans_general
    def logpdf(self, Phi, Sigma):
        return self.frozen_dist.logpdf(Phi, Sigma)

    def get_pseudo_obs(self):
        return self.__dumy, self.__dumx

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


# class RedFormTaylorPrior(MinnesotaPrior):

#     def __init__(self, ypre, hyperpara, p=3, cons=True, presample_moments=None, lamxx=None,
#                  rho=0.85, alpha_pi=1.0, alpha_z=1.5, sigma_R=0.006):

#         super(RedFormTaylorPrior, self).__init__(ypre, hyperpara, p=3, cons=True, presample_moments=None, lamxx=None)


#     @property
#     def Psi_new(self):
#         temp_psi = self.psi
#         temp_psi[0, 2] = (1-self.rho)*self.alpha_z*temp_psi[0, 0]
#         temp_psi[2, 0] = temp_psi[0, 0]

#         temp_psi[1, 2] = (1-self.rho)*self.alpha_pi*temp_psi[1, 1]
#         temp_psi[2, 1] = temp_psi[1, 1]

#         temp_psi[2, 2] = (1-self.rho)*self.alpha_pi*temp_psi[1, 1]
#         temp_psi[2, 2] = temp_psi[1, 1]        

#         return temp_psi

#     @property
#     def Phi_star_new(self):
#         pass
        

        

class TrainingSamplePrior(DummyVarPrior):
    pass


class HierarchicalMinnesotaPrior(MinnesotaPrior):
    def __init__():
        pass
