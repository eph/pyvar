import numpy as np
import pandas as p

from numpy.testing import assert_equal, assert_almost_equal

from unittest import TestCase

from pyvar import MinnesotaPrior

import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


class TestMinnesota(TestCase):

    def test_minnesota_prior_frank(self):
        
        datfile = os.path.join(dir_path,'sz_2008_joe_data.csv')

        yy = p.read_csv(datfile, names=['XGAP', 'INF', 'FFR'])
        lam = np.ones((6,))
        premom = np.c_[yy.mean().values, yy.std().values].T

        minnpr = MinnesotaPrior(yy, lam, p=1, presample_moments=premom)


        # From MDN/FS code
        YYdum = np.array([[ 0.024863787734143,                  0,                  0], 
                          [                 0,  0.025530860432406,                  0],  
                          [                 0,                  0,  0.033133670737398],  
                          [ 0.024863787734143,                  0,                  0],  
                          [                 0,  0.025530860432406,                  0],  
                          [                 0,                  0,  0.033133670737398],  
                          [-0.004767798723404,  0.037277177127660,  0.060582978723404],  
                          [-0.004767798723404,                  0,                  0],  
                          [                 0,  0.037277177127660,                  0],  
                          [                 0,                  0,  0.060582978723404]]) 


        XXdum = np.array([[   0.024863787734143,                  0,                  0,        0],
                          [                   0,  0.025530860432406,                  0,        0],  
                          [                   0,                  0,  0.033133670737398,        0], 
                          [                   0,                  0,                  0,        0],  
                          [                   0,                  0,                  0,        0],  
                          [                   0,                  0,                  0,        0],  
                          [  -0.004767798723404,  0.037277177127660,  0.060582978723404,   1.0000],  
                          [  -0.004767798723404,                  0,                  0,        0],  
                          [                   0,  0.037277177127660,                  0,        0],  
                          [                   0,                  0,  0.060582978723404,        0]])


        yydum_minn, xxdum_minn = minnpr.get_pseudo_obs()
        assert_almost_equal(yydum_minn, YYdum)
        assert_almost_equal(xxdum_minn, XXdum)


        YYdum = np.array([[0.024863787734143,                   0,                   0], 
                          [                  0,  0.025530860432406,                    0],
                          [                  0,                  0,    0.033133670737398],  
                          [  0.024863787734143,                  0,                    0], 
                          [                  0,  0.025530860432406,                    0],  
                          [                  0,                  0,    0.033133670737398],  
                          [ -0.019071194893617,  0.149108708510638,    0.242331914893617],  
                          [ -0.004767798723404,                  0,                    0],  
                          [                  0,  0.037277177127660,                    0],  
                          [                  0,                  0,    0.060582978723404]])

        XXdum = np.array([[ 0.024863787734143 ,                   0  ,                 0   ,                  0] , 
                          [                 0 ,    0.025530860432406 ,                 0   ,                  0] , 
                          [                 0 ,                  0   , 0.033133670737398   ,                  0] , 
                          [                 0 ,                  0   ,                 0   ,                  0] , 
                          [                 0 ,                 0    ,                0    ,                  0] , 
                          [                 0 ,                 0    ,                0    ,                  0] , 
                          [-0.019071194893617 , 0.149108708510638    , .242331914893617    ,  4.000000000000000] , 
                          [-0.004767798723404 ,                 0    ,                0    ,                  0] , 
                          [                 0 , 0.037277177127660    ,                0    ,                  0] , 
                          [                 0 ,                 0    ,   0.060582978723404 ,                  0]])

        lam[4] = 4.0

        minnpr = MinnesotaPrior(yy, lam, p=1, presample_moments=premom)
        yydum_minn, xxdum_minn = minnpr.get_pseudo_obs()

        assert_almost_equal(yydum_minn, YYdum)
        assert_almost_equal(xxdum_minn, XXdum)

        # matlab call
        # [ydu, xdu, breaks] = varprior_h(3, 5, 1, [1 1 1 1 4]', [ybar' sbar'])
        YYdum = np.loadtxt(os.path.join(dir_path,'ydu_p5.txt'))
        XXdum = np.loadtxt(os.path.join(dir_path,'xdu_p5.txt'))
        lam = np.ones((6,))
        lam[3] = 4.0
        minnpr = MinnesotaPrior(yy, lam, p=5, presample_moments=premom)
        yydum_minn, xxdum_minn = minnpr.get_pseudo_obs()

        assert_almost_equal(yydum_minn, YYdum)
        assert_almost_equal(xxdum_minn, XXdum)

    def test_minnesota_prior_mark(self):
        
        datfile = os.path.join(dir_path,'sz_2008_joe_data.csv')
        yy = p.read_csv(datfile, names=['XGAP', 'INF', 'FFR'])

        lam = [2.5, 1, 1, 1, 3, 1]
        ybar = [-0.0010, 0.0122, 0.0343]
        sbar = [ 0.0076, 0.0111, 0.0092]
        premom = np.array([ybar, sbar])
        m = MinnesotaPrior(yy, lam, p=5, presample_moments=premom, lamxx=True)

        # matlab call
        # prihyp = pri_var_dummy([2.5, 1, 1, 3, 1, 1,  1], premom, 5, 3, 5*3+1)
        from scipy.io import loadmat
        Omega = loadmat(os.path.join(dir_path,'prihyp.mat'))['prihyp'][0][0][0]
        Phi_star = loadmat(os.path.join(dir_path,'prihyp.mat'))['prihyp'][0][0][1]
        Psi = loadmat(os.path.join(dir_path,'prihyp.mat'))['prihyp'][0][0][2]
        nu = loadmat(os.path.join(dir_path,'prihyp.mat'))['prihyp'][0][0][3]

        assert_almost_equal(Omega, m.Omega)
        assert_almost_equal(Phi_star, m.Phi_star)
        assert_almost_equal(Psi, m.Psi)
        

    def test_minnesota_prior_fortran(self):
        datfile = os.path.join(dir_path,'sz_2008_joe_data.csv')
        yy = p.read_csv(datfile, names=['XGAP', 'INF', 'FFR'])

        lam = [2.5, 1, 1, 1, 3, 1]
        ybar = [-0.0010, 0.0122, 0.0343]
        sbar = [ 0.0076, 0.0111, 0.0092]
        premom = np.array([ybar, sbar])
        m = MinnesotaPrior(yy, lam, p=5, presample_moments=premom, lamxx=True)

        # fortran call
        # call write_prior_hyper(2.5_wp,1.0_wp,1.0_wp,1.0_wp,3.0_wp,1.0_wp,1.0_wp, &
        #  (/-0.0010_wp, 0.0122_wp, 0.0343_wp/), (/ 0.0076_wp, 0.0111_wp, 0.0092_wp/))
        Omega_inv = np.loadtxt(os.path.join(dir_path,'hyper_Omega_inv.txt'))
        Phi_star  = np.loadtxt(os.path.join(dir_path,'hyper_phistar.txt'))
        Psi       = np.loadtxt(os.path.join(dir_path,'hyper_iw_Psi.txt'))
        nu        = np.loadtxt(os.path.join(dir_path,'hyper_iw_nu.txt'))

        assert_almost_equal(Omega_inv, np.linalg.inv(m.Omega))
        assert_almost_equal(Phi_star, m.Phi_star.flatten(order='F'))
        assert_almost_equal(Psi, m.Psi)

        lam = [2.5, 2, 1, 1.3, 3, 1]
        m = MinnesotaPrior(yy, lam, p=5, presample_moments=premom, lamxx=True)
        Omega_inv = np.loadtxt(os.path.join(dir_path,'hyper_Omega_inv_two.txt'))
        Phi_star = np.loadtxt(os.path.join(dir_path,'hyper_phistar_two.txt'))
        Psi = np.loadtxt(os.path.join(dir_path,'hyper_iw_Psi_two.txt'))
        nu = np.loadtxt(os.path.join(dir_path,'hyper_iw_nu_two.txt'))

        assert_almost_equal(Omega_inv, np.linalg.inv(m.Omega))
        assert_almost_equal(Phi_star, m.Phi_star.flatten(order='F'))
        assert_almost_equal(Psi, m.Psi)
