from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import copy
#from var_data import var_data, real_time_dataset
from math import sqrt, pi

class PredictiveDensity:
    """
    A class for the evaluation of empirical predictive densities.
    """
    def __init__(self, yypred, yyact):

        self.__nsim, self.__h, self.__ny = yypred.shape
        print ("Initializing an %i-step predictive density with %i simulations for %i variables." % (yypred.shape[1], yypred.shape[0], yypred.shape[2]))
        self.__yypred = yypred
        self.__yyact = yyact
        self.__hact, self.__nyact = yyact.shape

    def get_unconditional_pits(self):
        """
        Returns the unconditional pits.
        """
        pits = np.ma.masked_array(np.zeros((self.__h, self.__ny)))

        if self.__hact < self.__h:
            pits[self.__hact:, :] = np.ma.masked

        for s in np.arange(0, self.__ny):
            for h in range(0, min(self.__hact, self.__h)):
                fcsts = self.__yypred[:, h, s]
                pits[h, s] = np.mean(fcsts < self.__yyact[h, s])

        return pits

    def get_conditional_pits(self, approx_type="KERNEL"):
        """
        Returns the conditional pits.
        """
        pits = np.ma.masked_array(np.zeros((self.__ny, self.__h, self.__ny)))

        if approx_type == "KERNEL":
            for c in np.arange(0, self.__ny):
                if self.__hact < self.__h:
                    pits[:, self.__hact:, :] = np.ma.masked

                for h in range(0, min(self.__hact, self.__h)):
                    band = 1.06 * np.std(self.__yypred[:, h, c]) * pow(self.__nsim, -0.20)
                    kern = (self.__yypred[:, h, c] - self.__yyact[h, c]) / band
                    weight = 1.0 / sqrt(2 * pi) * np.exp(-0.5 * pow(kern, 2.0))
                    weight = weight / np.sum(weight)
                    for s in np.arange(0, self.__ny):
                        fcsts = self.__yypred[:, h, s]
                        pits[c, h, s] = np.sum((fcsts < self.__yyact[h, s]) * weight)

        elif approx_type == "NORMAL":
            print("Not yet implemented.")

        return pits

    def get_unconditional_mse(self):
        """
        Returns the unconditional MSE (mean(yypred) - yyact)^2.
        """
        mse = np.ma.masked_array(np.zeros((self.__h, self.__ny)))

        if self.__hact < self.__h:
            mse[self.__hact:, :] = np.ma.masked

        for s in np.arange(0, self.__ny):
            for h in range(0, min(self.__hact, self.__h)):
                fcsts = self.__yypred[:, h, s]
                mse[h, s] = pow(np.mean(fcsts) - self.__yyact[h, s], 2)

        return mse

    def get_conditional_mse(self, type="KERNEL"):
        """
        Returns the conditional MSE (mean(yypred|i) - yyact)^2.
        """
        cmse = np.ma.masked_array(np.zeros((self.__ny, self.__h, self.__ny)))

        if self.__hact < self.__h:
            cmse[:, self.__hact:, :] = np.ma.masked

        for c in np.arange(0, self.__ny):
            for h in range(0, min(self.__hact, self.__h)):
                band = 1.06 * np.std(self.__yypred[:, h, c]) * pow(self.__nsim, -0.20)
                kern = (self.__yypred[:, h, c] - self.__yyact[h, c]) / band
                weight = 1.0 / sqrt(2 * pi) * np.exp(-0.5 * pow(kern, 2.0))
                weight = weight / np.sum(weight)
                for s in np.arange(0, self.__ny):
                    fcsts = self.__yypred[:, h, s]
                    cmse[c, h, s] = pow(np.sum(fcsts * weight) - self.__yyact[h, s], 2)

        return cmse

    def __get_kernel_weights():
        pass


# ForecastingExercise(rtdata, BayesianVAR(DiffusePrior()), hmax)

# myBVARSET = []
# unifpr = DiffusePrior()
# pits = np.ma.masked_array(np.zeros((rtdata.size(), hmax, x.ny)))
# rmse = np.ma.masked_array(np.zeros((rtdata.size(), hmax, x.ny)))

# for i in range(0, rtdata.size()):
#     print "VAR %2i" % i
#     myBVARSET.append(BayesianVAR(unifpr))
#     myBVARSET[len(myBVARSET)-1].data = rtdata.getVARData(i)

#     para = myBVARSET[len(myBVARSET)-1].estimate(nsim = 1000)
#     pred = myBVARSET[len(myBVARSET)-1].predictive_density(para, h=hmax)

#     yyfuture = rtdata.getFORData(i)
#     for s in np.arange(0, yyfuture.ny):
#         if yyfuture.n < hmax:
#             pits[i, yyfuture.n:, :] = np.ma.masked
#         for h in range(0, min(yyfuture.n, hmax)):
#             fcsts = pred[:, h, s]
#             pits[i,h,s] = np.mean(fcsts < yyfuture.series[h, s])


# plt.figure(1)
# for i in range(0, 3):
#     plt.subplot(1, 3, i+1)
#     N, bins = np.histogram(pits[:, i], 5, range=(0.0, 1.0))
#     N = N/np.sum(N)*100
#     plt.bar(bins[:-1], N,  width=0.2)
#     plt.ylim((0, 100))
#
#
# plt.show()
# dstart = "1965q1"
# estart = "1996q1"
# hmax = 8;
# x = var_data.read_from_csv(filename="3eqvar.csv", header=True, freq="quarterly", start="1965q1")
# rtdata = real_time_dataset(x, estart="1996q1", t0="1965q1")
