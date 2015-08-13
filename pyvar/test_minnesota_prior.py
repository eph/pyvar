from __future__ import division

import numpy as np
import pandas as p

from numpy.testing import assert_equal, assert_almost_equal

from unittest import TestCase

from pyvar import MinnesotaPrior


class TestMinnesota(TestCase):

    def test_minnesota_prior(self):
        
        datfile = '/mq/home/m1eph00/projects/var-smc/var_smc_working/lib_var/sz_2008_joe_data.csv'

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
        print lam
        minnpr = MinnesotaPrior(yy, lam, p=1, presample_moments=premom)
        yydum_minn, xxdum_minn = minnpr.get_pseudo_obs()
        print yydum_minn
        print YYdum
        assert_almost_equal(yydum_minn, YYdum)
        assert_almost_equal(xxdum_minn, XXdum)

