from __future__ import division
import numpy as np

from scipy.linalg import solve, block_diag
from statsmodels.tsa.tsatools import vech, lagmat
from .distributions import NormInvWishart


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
    def __init__(self, ny=3, p=6, cons=True):
        print("Initializing dummy prior.")

        self.n = ny
        self.ny = ny
        self.p = p
        self.cons = cons

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

    def get_pseudo_obs(self):
        return None, None    
class DiffusePrior(DummyVarPrior):

    def __init__(self):
        print("Initializing Diffuse Prior.")

    def get_pseudo_obs(self):
        return None, None

    def draw(self):
        print("This is an improper prior.")

class SimsZhaSVARPrior(Prior):

    def __init__(self, ypre, hyperpara, p=3, cons=True, ar_lp=6,
                 presample_moments=None, restriction='upper_triangle'):

        
        lam0 = hyperpara[0]
        lam1 = hyperpara[1]
        lam2 = hyperpara[2]
        lam3 = hyperpara[3]
        lam4 = hyperpara[4]
        mu5 = hyperpara[5]
        mu6 = hyperpara[6]
        
        mu0 = 1/lam0
        mu1 = 1/lam1
        mu2 = 1/lam2
        mu3 = -lam3
        mu4 = 1/lam4

        ypre = np.array(ypre)
        T, ny = ypre.shape
        k = ny*p + cons
        
        Uar = np.zeros((T-ar_lp, ny))
        for i in np.arange(ny):
            Xar, Yar = lagmat(ypre[:, i], maxlag=ar_lp, trim='both', original='sep')
            betaar = np.linalg.inv(Xar.T.dot(Xar)).dot(Xar.T.dot(Yar))
            Uar[:, i] = (Yar - Xar.dot(betaar)).flatten()


        ybar = np.atleast_1d(np.mean(np.asarray(ypre)[:p, :], 0))
        sbar = np.atleast_1d(np.std(np.asarray(Uar), 0, ddof=1)) # np.std = sqrt(sum(x-mu)**2/N)

        if presample_moments is not None:
            ybar = presample_moments[0]
            sbar = presample_moments[1]

        #------------------------------------------------------------ 
        # A0 hyperparameters
        #------------------------------------------------------------
        A0 = []
        for j in np.arange(ny):
            A0.append(dict())
            
            A0[j]['mu'] = np.zeros((ny, 1))
            A0[j]['sigma'] = np.zeros((ny, ny))
            np.fill_diagonal(A0[j]['sigma'], (lam0/sbar)**2)


        #------------------------------------------------------------
        # Aplus hyperparameters
        #------------------------------------------------------------
        Aplus = []
        for j in np.arange(ny):
            Yid = np.zeros((k+ny+1, ny))
            Xid = np.zeros((k+ny+1, k))
            
            Yid[:ny, :ny] = np.diag( mu0*mu1*mu2/(1**mu3) * sbar )

            for r in np.arange(ny):
                for l in np.arange(p):
                    Xid[ny*l+r, ny*l+r] = mu0*mu1*mu2*sbar[r] / ((l+1)**mu3)

            Xid[-(ny+2), -1] = mu0*mu4

            Yid[-(ny+1):-1, :] = np.diag(mu5*ybar)
            Xid[-(ny+1):-1, :][:, :(k-1)] = np.tile(mu5*np.diag(ybar), p)

            Yid[-1, :] = mu6*ybar
            Xid[-1, :] = np.hstack((np.tile(mu6*ybar, p), mu6));

            sig = np.linalg.inv(Xid.T.dot(Xid))

            Aplus.append(dict())

            Aplus[j]['sigma'] = sig.dot(Xid.T.dot(Yid)).dot(A0[j]['sigma']).dot((sig.dot(Xid.T.dot(Yid))).T) + sig
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
            SIGMA[j0:j0+ny*p+cons, :][:, j1:j1+ny] = Aplus[j]['cov'] 
            SIGMA[j1:j1+ny, :][:, j0:j0+ny*p+cons] = Aplus[j]['cov'].T 
            j0 += ny*p+cons                                               
            j1 += ny

        
        #------------------------------------------------------------
        # Restriction business
        #------------------------------------------------------------
        self.restrictions = dict()
        self.restrictions['A0'] = np.ones((ny, ny), dtype=bool, order='F')
        if restriction=='upper_triangle':
            self.restrictions['A0'][np.tril_indices(ny, -1)] = False
            restriction = self.restrictions['A0']
        else:
            self.restrictions['A0'] = restriction
            # error check!
            #if not(restriction.shape==(ny, ny)):
                
            #    print "badly shaped restriction matrix"
            # if np.sum(~restriction)!=ny:
            #     print "needs an exact identification scheme!"
            #     fdskf
                
                

        
        selind = np.reshape(np.arange(ny**2), (ny, ny), order='F')[~self.restrictions['A0']]
        mask = np.ones(MU.size, dtype=bool, order='F')
        mask[selind] = False

        MU    = MU[mask]        
        cov   = SIGMA[mask, :][:, selind]
        var   = SIGMA[selind, :][:, selind]
        SIGMA = SIGMA[mask, :][:, mask] - cov.dot(np.linalg.inv(var)).dot(cov.T)
        
        self.mu = MU
        self.sigma = SIGMA

        self.n_a0_free = np.sum(restriction)
        
        from scipy.stats import multivariate_normal
        self.n = self.ny
        self.prior = multivariate_normal(self.mu, self.sigma)

    def rvs(self, size=1, flatten=True):
        r = self.prior.rvs(size=size)
        if flatten==False:
            A0s = np.zeros((size, self.n, self.n))
            Apluses = np.zeros((size, self.n*self.p + self.cons, self.n))
            for i in range(size):
                (A0s[i]).T[self.restrictions['A0'].T] = r[i][:self.n_a0_free] 
                Apluses[i] = np.reshape(r[i][self.n_a0_free:], (self.n*self.p+self.cons, self.n))
            r = A0s, Apluses
        return r

    @para_trans_general
    def logpdf(self, A0, Aplus, *args, **kwargs):
        x = np.r_[A0.T[self.restrictions['A0'].T], np.ravel(Aplus, order='F')]
        return self.prior.logpdf(x)
        
    @para_trans_general
    def reduced_form(self, A0, Aplus, *args, **kwargs):
        A0i = np.linalg.inv(A0)
        Phi = Aplus.dot(A0i)
        Sigma = np.linalg.inv(A0.dot(A0.T))
        return Phi, Sigma
        
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
                    #x[ :(self.ny * (self.p - 1))] = x[(self.ny):(nx-self.cons)]
                    #x[ (self.ny * (self.p - 1)):-1] = y[i, j-1, :]
                    #x[ (self.ny * (self.p - 1)):-(1)] = y[i, j-1, :]
                    x[self.ny:-self.cons] = x[:-(self.ny+self.cons)]
                    x[:self.ny] = y[i, j-1, :]

                else:
                    x = y[i, j-1, :]

                y[i, j, :] = x.dot(Phi)

        return y
        

    @para_trans_general
    def companion_form(self, A0, Aplus):

        Phi, Sigma = self.reduced_form(A0, Aplus)

        ptilde = np.max([self.p, 1])

        F = np.diag( np.ones((ptilde-1)*self.ny), -self.ny)

        for i in range(self.p):
            F[:self.ny, :][:, i*self.ny:(i+1)*self.ny] = Phi[i*self.ny:(i+1)*self.ny, :].T

        R = np.zeros((ptilde*self.ny, self.ny))
        R[:self.ny, :] = np.linalg.inv(A0).T

        C = np.zeros((ptilde*self.ny, ))

        if self.cons:
            C[:self.ny] = Phi[-1, :]
            
        return C, F, R

    @para_trans_general
    def structural_fevd(self, A0, Aplus, h=10, return_fev=False, cumulate=None):

        #Phi, Sigma = self.reduced_form(A0, Aplus)

        C, F, R = self.companion_form(A0, Aplus)

        selind = np.arange(self.ny)
        if cumulate is not None:

            R = np.r_[R,  np.zeros((self.ny, self.ny))]

            Fold = F.copy()
            F = np.eye(Fold.shape[0]+self.ny)
            F[:Fold.shape[0], :Fold.shape[0]] = Fold.copy()
            x = np.eye(F.shape[0])
            x[-self.ny:, :self.ny] = -np.eye(self.ny)
            F = np.linalg.inv(x).dot(F)
            R = np.linalg.inv(x).dot(R)

            for ci in cumulate:
                selind[ci] = F.shape[0] - self.ny + ci
            
        fev = np.zeros((h, self.ny))
        fevd = np.zeros((self.ny, h, self.ny))
        
        Fhat = np.eye(F.shape[0])
        Sigma = R.dot(R.T)
        cov = np.zeros_like(F)
        for i in range(h):
            cov += Fhat.dot(Sigma).dot(Fhat.T)
            fev[i, :] = np.diag(cov)[selind]
            Fhat = Fhat.dot(F)

        for i in range(self.ny):

            Rhat = np.zeros_like(R)
            Rhat[:, i] = R[:, i]
            Sigma = Rhat.dot(Rhat.T)

            Fhat = np.eye(F.shape[0])
            cov = np.zeros_like(F)
            for j in range(h):
                cov += Fhat.dot(Sigma).dot(Fhat.T)
                fevd[i, j, :] = np.diag( cov )[selind]
                fevd[i, j, :] = fevd[i, j, :] / fev[j, :]

                Fhat = Fhat.dot(F)

        if return_fev:
            return fevd, fev
        else:
            return fevd

    def para_trans(self,*args,**kwargs):
        dtype = kwargs.pop('dtype', float)
        if len(args) == 1:
            theta = np.asarray(args[0])
            n = self.n
            p = self.p
            cons = self.cons
            A0 = np.zeros((n, n),order='F', dtype=dtype)
            A0.T[self.restrictions['A0'].T] = theta[:self.n_a0_free]
            Aplus = np.reshape(theta[self.n_a0_free:], (n*p+cons, n), order='F')
        else:
            A0 = args[0]
            Aplus = args[1]
        return A0, Aplus


    @para_trans_general
    def logpdf(self, A0, Aplus, *args, **kwargs):
        x = np.r_[A0.T[self.restrictions['A0'].T], np.ravel(Aplus, order='F')]
        return self.prior.logpdf(x)


    def fortran(self, data, name='svar', output_dir='/mq/scratch/m1eph00/mp-premia/svar/'):

        import os
        try:
            os.mkdir(output_dir)
        except:
            pass
        with open('/mq/home/m1eph00/python-repo/dsge/dsge/templates/smc_driver_mpi.f90') as f:
            smc = f.read()

        smc = smc.replace('call read_in_from_files()', 'call load_data()')
        smc = smc.replace('100f', '200f')
        
        with open(output_dir + 'smc_driver_mpi.f90', 'w') as f:
            f.write(smc.format(model=name))

        with open('/mq/home/m1eph00/python-repo/dsge/dsge/templates/Makefile_dsge') as f:
            makefile = f.read()

        makefile = makefile.replace('-O3 -xHost -ipo', '')
        
        with open(output_dir + 'Makefile', 'w') as f:
            f.write(makefile.format(model=name))

        base = os.path.join(output_dir, 'base')
        try:
            os.symlink('/mq/home/m1eph00/code/fortran/base', base)
        except:
            print("file exists")


        output_dir = output_dir + 'model/'
        try:
            os.mkdir(output_dir)
        except:
            pass


        

        output = {}
        output['name'] = name        
        output['p'] = self.p
        output['cons'] = self.cons
        output['ny'] = self.ny
        output['T'] = data.shape[0]
        output['nF'] = self.n*(self.n*self.p + self.cons)
        output['nA'] = np.sum(self.restrictions['A0'])
        output['datafile'] = output_dir + 'yy.txt'

        np.savetxt(output['datafile'], data)

        #------------------------------------------------------------
        # prior
        #------------------------------------------------------------
        output['AFmufile'] = output_dir + 'AFmu.txt'
        output['AFvarfile'] = output_dir + 'AFvar.txt'

        np.savetxt(output['AFmufile'], self.mu)
        np.savetxt(output['AFvarfile'], np.linalg.cholesky(self.sigma))
        
        #------------------------------------------------------------
        # likelihood
        #------------------------------------------------------------
        x =self.rvs(size=100, flatten=True).mean(0)
        output['npara'] = x.size
        x = sympy.symbols(['para({:d})'.format(i+1) for i in range(x.size)], positive=True)
        a0, aplus = self.para_trans(x, dtype=object)

        mat_str = lambda *x: '    {}({}, {}) = {}'.format(*x)
        A0str = [mat_str('A0', i+1, j+1, sympy.fcode(value, source_format='free'))
                 for (i, j), value in np.ndenumerate(a0) if value > 0]
        Apstr = [mat_str('F', i+1, j+1, sympy.fcode(value, source_format='free'))
                 for (i, j), value in np.ndenumerate(aplus) if value > 0]


        output['assign_para'] = '\n'.join(A0str + Apstr)

        with open('/mq/home/m1eph00/python-repo/var/pyvar/svar.f90', 'r') as f:
            svar = f.read()

        with open(output_dir + name + '.f90', 'w') as f:
            f.write(svar.format(**output))
            


from scipy.stats import norm, gamma
class TwoExternalInstrumentsSVARPrior(SimsZhaSVARPrior):

    def __init__(self, *args, **kwargs):

        self.gamma = kwargs.pop('gamma', None)
        self.rho = kwargs.pop('rho', None)

        self.gamma21 = kwargs.pop('gamma21', None)
        self.gamma22 = kwargs.pop('gamma22', None)
        self.rho2 = kwargs.pop('rho2')
        
        super(TwoExternalInstrumentsSVARPrior, self).__init__(*args, **kwargs)

        self.ny = self.ny+2
        
    def rvs(self, size=1, flatten=True):
        x = super(TwoExternalInstrumentsSVARPrior, self).rvs(size=size, flatten=flatten)
        gamma = self.gamma.rvs(size=size)
        rho = self.rho.rvs(size=size)

        gamma21 = self.gamma21.rvs(size=size)
        gamma22 = self.gamma22.rvs(size=size)
        rho2 = self.rho2.rvs(size=size)

        x0 = np.c_[x, gamma, rho, gamma21, gamma22, rho2]
        
        if flatten == True:
            return x0 

    def para_trans(self, *args, **kwargs):
        dtype=kwargs.pop('dtype', float)
        sqrt = np.sqrt
        if dtype==object:
            import sympy
            sqrt = sympy.sqrt
        if len(args) == 1:
            A0, Aplus = super(TwoExternalInstrumentsSVARPrior, self).para_trans(args[0][:-5], dtype=dtype) 

            gam = args[0][-5]
            rho = args[0][-4]
            gam21 = args[0][-3]
            gam22 = args[0][-2]
            rho2 = args[0][-1]


            A0t = np.zeros((self.ny, self.ny), dtype=dtype)
            A0t[:-2, :][:, :-2] = A0
            A0t[-1, -1] = 1
            A0t[-2, -2] = 1
            if dtype==float:
                I = np.eye(self.ny)
            else:
                I = sympy.Matrix(self.ny, self.ny, np.zeros((self.ny**2)))
                for j in range(self.ny):
                    I[j, j] = 1
                
            I[-1, -1] = gam*sqrt( (1 - rho)/rho)
            I[0, -1] = gam
            I[0, -2] = gam21
            I[1, -2] = gam22
            I[-2, -2] = sqrt(gam21**2 + gam22**2)*sqrt( (1-rho2)/rho2)
            
            if dtype==object:
                A0t = sympy.Matrix(A0t) * sympy.Matrix(I).inv()
            else:
                A0t = A0t.dot(np.linalg.inv(I))
            

            Aplust = np.zeros((self.ny*self.p + self.cons, self.ny), dtype=dtype)
            for i in range(self.p):
                ind0 = i*self.n 
                ind0t = i*self.ny
                Aplust[ind0t:ind0t+self.n, :][:, :-2] = Aplus[ind0:ind0+self.n, :]

            if self.cons == True:
                Aplust[-1, :-2] = Aplus[-1, :]

            if dtype==object:
                Aplust = sympy.Matrix(Aplust) * sympy.Matrix(I).inv()
            else:
                Aplust = Aplust.dot(np.linalg.inv(I))

            return A0t, Aplust

        elif len(args) == 2:
            return args[0], args[1]
    
            

    def logpdf(self, x):
        A0, Aplus = super(TwoExternalInstrumentsSVARPrior, self).para_trans(x[:-5]) 
        sz = super(TwoExternalInstrumentsSVARPrior, self).logpdf(A0, Aplus)

        if self.parameterization == 'rho':
            pdfx = self.rho.logpdf(x[-1])
        else:
            pdfx = self.signu.logpdf(x[-1])
            
        return sz + self.gamma.logpdf(x[-2]) + pdfx


class ExternalInstrumentsSVARPrior(SimsZhaSVARPrior):

    def __init__(self, *args, **kwargs):

        self.gamma = kwargs.pop('gamma', None)
        self.rho = kwargs.pop('rho', None)
        self.signu = kwargs.pop('signu', None)

        self.parameterization = 'rho'
        if self.signu is not None:
            self.parameterization = 'signu'
            
        super(ExternalInstrumentsSVARPrior, self).__init__(*args, **kwargs)

        self.ny = self.ny+1
        
    def rvs(self, size=1, flatten=True):
        x = super(ExternalInstrumentsSVARPrior, self).rvs(size=size, flatten=flatten)
        gamma = self.gamma.rvs(size=size)
        if self.parameterization == 'rho':
            rho = self.rho.rvs(size=size)
        else:
            rho = self.signu.rvs(size=size)
        x0 = np.c_[x, gamma, rho]
        
        if flatten == True:
            return x0 

    def para_trans(self, *args, **kwargs):
        dtype=kwargs.pop('dtype', float)
        sqrt = np.sqrt
        if dtype==object:
            import sympy
            sqrt = sympy.sqrt
        if len(args) == 1:
            A0, Aplus = super(ExternalInstrumentsSVARPrior, self).para_trans(args[0][:-2], dtype=dtype) 

            gam = args[0][-2]
            if self.parameterization == 'signu':
                signu = args[0][-1]
                rho = gam**2 / (gam**2 + signu**2)
            else:
                rho = args[0][-1]

            A0t = np.zeros((self.ny, self.ny), dtype=dtype)
            A0t[:-1, :][:, :-1] = A0
            A0t[:-1, -1] = -A0[:, 0] / sqrt( (1-rho)/rho)
            A0t[-1, -1] = 1. / (sqrt( (1-rho)/rho)*gam)

            Aplust = np.zeros((self.ny*self.p + self.cons, self.ny), dtype=dtype)
            for i in range(self.p):
                ind0 = i*self.n 
                ind0t = i*self.ny
                Aplust[ind0t:ind0t+self.n, :][:, :-1] = Aplus[ind0:ind0+self.n, :]

            if self.cons == True:
                Aplust[-1, :-1] = Aplus[-1, :]

            Aplust[:, -1] = -Aplust[:, 0] / sqrt( (1-rho)/rho)
            return A0t, Aplust

        elif len(args) == 2:
            return args[0], args[1]
    
            

    def logpdf(self, x):
        A0, Aplus = super(ExternalInstrumentsSVARPrior, self).para_trans(x[:-2]) 
        sz = super(ExternalInstrumentsSVARPrior, self).logpdf(A0, Aplus)

        if self.parameterization == 'rho':
            pdfx = self.rho.logpdf(x[-1])
        else:
            pdfx = self.signu.logpdf(x[-1])
            
        return sz + self.gamma.logpdf(x[-2]) + pdfx


class FixedEndogeneitySVARPrior(SimsZhaSVARPrior):
    def __init__(self, *args, **kwargs):

        self.gamma = kwargs.pop('gamma', None)
        self.rho = kwargs.pop('rho', None)
        self.zeta = kwargs.pop('zeta', 0.000)

        self.signu = kwargs.pop('signu', None)

        self.parameterization = 'rho'
        if self.signu is not None:
            self.parameterization = 'signu'


        super(FixedEndogeneitySVARPrior, self).__init__(*args, **kwargs)

        self.ny = self.ny+1

    def rvs(self, size=1, flatten=True):
        x = super(FixedEndogeneitySVARPrior, self).rvs(size=size, flatten=flatten)
        gamma = self.gamma.rvs(size=size)
        if self.parameterization == 'rho':
            rho = self.rho.rvs(size=size)
        else:
            rho = self.signu.rvs(size=size)
        x0 = np.c_[x, gamma, rho]

        if flatten == True:
            return x0 

    def para_trans(self, *args, **kwargs):
        dtype=kwargs.pop('dtype', float)
        sqrt = np.sqrt
        if dtype==object:
            import sympy
            sqrt = sympy.sqrt
        if len(args) == 1:
            A0, Aplus = super(FixedEndogeneitySVARPrior, self).para_trans(args[0][:-2], dtype=dtype) 
            rho = args[0][-1]
            gam = args[0][-2]

            gam = args[0][-2]
            if self.parameterization == 'signu':
                signu = args[0][-1]
                rho = gam**2 / (gam**2 + signu**2)
            else:
                rho = args[0][-1]


            A0t = np.zeros((self.ny, self.ny), dtype=dtype)
            A0t[:-1, :][:, :-1] = A0
            A0t[-1, -1] = 1
            if dtype==float:
                I = np.eye(self.ny)
            else:
                I = sympy.Matrix(self.ny, self.ny, np.zeros((self.ny**2)))
                for j in range(self.ny):
                    I[j, j] = 1
            I[0, -1] = 0
            I[-2, -1] = gam
            I[-1, -1] = signu ##sqrt(gam**2 + self.zeta**2)*sqrt((1-rho)/rho)
            
            if dtype==object:
                A0t = sympy.Matrix(A0t) * sympy.Matrix(I).inv()
            else:
                A0t = A0t.dot(np.linalg.inv(I))
            
            # A0t[:-1, -1] = -A0[:, 0] / sqrt( (1-rho)/rho)
            # A0t[-1, -1] = 1. / (sqrt( (1-rho)/rho)*gam)

            Aplust = np.zeros((self.ny*self.p + self.cons, self.ny), dtype=dtype)
            for i in range(self.p):
                ind0 = i*self.n 
                ind0t = i*self.ny
                Aplust[ind0t:ind0t+self.n, :][:, :-1] = Aplus[ind0:ind0+self.n, :]

            if self.cons == True:
                Aplust[-1, :-1] = Aplus[-1, :]
                #Aplust[-2, :] = 0
            #Aplust[:, -1] = -Aplust[:, 0] / sqrt( (1-rho)/rho)
            if dtype==object:
                Aplust = sympy.Matrix(Aplust) * sympy.Matrix(I).inv()
            else:
                Aplust = Aplust.dot(np.linalg.inv(I))

            return A0t, Aplust

        elif len(args) == 2:
            return args[0], args[1]


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
from scipy.stats import multivariate_normal
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
        print('r_idx', r_idx)
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


        dumr = ny * 2 + lam3*ny + ny * (p-1) + cons
        self.__dumy = np.mat(np.zeros((dumr, ny)))
        self.__dumx = np.mat(np.zeros((dumr, ny * p + cons)))


        # tightness on prior of own lag coefficients
        self.__dumy[0:ny, 0:ny] = lam1 * tau * np.diag(sbar)
        self.__dumx[0:ny, 0:ny] = lam1 * np.diag(sbar)

        subt = ny
        # scaling coefficient for higher orger lags
        for l in np.arange(1, p):
            for i in np.arange(0, ny):
                disp = subt + i
                diag = ny * l + i
                self.__dumx[disp, diag] = lam1 * sbar[i] * pow(l+1, lam2)

            subt += ny

        # prior for covariance matrix
        for l in np.arange(0, lam3):
            for i in np.arange(0, ny):
                self.__dumy[subt + i, i] = sbar[i]
            subt += ny


        # co-persistence dummy observations
        self.__dumy[subt, :] = np.mat(lam5 * ybar)
        for i in np.arange(0, p):
            self.__dumx[subt, i * ny:(i + 1) * ny] = lam5 * ybar

        if cons:
            self.__dumx[subt, -1] = lam5
        subt += 1

        # sum of coefficients dummies
        for i in np.arange(0, ny):
            disp = subt + i
            self.__dumy[disp, i] = lam4 * ybar[i]
            for l in np.arange(0, p):
                xcord = l * ny + i
                self.__dumx[disp, xcord] = lam4 * ybar[i]

        subt += ny

        # Add Mark's thing (NOT IN DNS)
        if lamxx is not None:
            self.__dumy = np.vstack((self.__dumy, np.zeros((ny))))
            x = np.zeros((ny*p+cons))
            x[-1] = lam1/lamxx
            self.__dumx = np.vstack((self.__dumx, x))

        self.__dumy = np.asarray(self.__dumy)
        self.__dumx = np.asarray(self.__dumx)


        self.frozen_dist = NormInvWishart(self.Phi_star, self.Omega, self.Psi, self.nu)
        #self.frozen_dist = NormInvWishart(self.Phi_star, self.Omega, self.Psi, 4)

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


class SteadyStateVARPrior(Prior):
    """A class for Steady State VAR prior."""

    def __init__(self, *args, **kwargs):

        self.ntrends = kwargs.pop('ntrends')
        self.gamma_mu = kwargs.pop('gamma_mu')
        self.gamma_var = kwargs.pop('gamma_var')
        self.gamma_pdf = multivariate_normal(self.gamma_mu, self.gamma_var)

        kwargs['cons']=False

        self.cyclical_prior = MinnesotaPrior(*args, **kwargs)
        self.n = self.cyclical_prior.n
        self.p = self.cyclical_prior.p
        self.cons = self.cyclical_prior.cons

    @para_trans_general
    def logpdf(self, Phi, Sigma, Gamma):
        lpdf = self.gamma_pdf.logpdf(Gamma.flatten(order='F'))
        lpdf += self.cyclical_prior.logpdf(Phi, Sigma)
        return lpdf

    def rvs(self, size=1, flatten_output=True):
        gammas = self.gamma_pdf.rvs(size=size)
        var_para = self.cyclical_prior.rvs(size=size, flatten_output=flatten_output)

        if flatten_output:
            return np.c_[var_para, gammas]
        else:
            return var_para[0], var_para[1], gammas
            
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



        


    
        

