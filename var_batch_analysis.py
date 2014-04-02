from __future__ import division
from var_data import var_data, real_time_dataset
from varprior import DiffusePrior, MinnesotaPrior
from pyvar import BayesianVAR, CompleteModel, ForecastingExercise
import numpy as np
import csv
import os
data_set = var_data.read_from_csv(filename="3eqvar.csv", header=True, freq="quarterly", start="1965q2")
rt_data = real_time_dataset(data_set, estart="1996q1", t0="1965q2")
unifpr = DiffusePrior()             # initialize uniform prior

# Minnesota Prior
lam1 = 5.0
lam2 = 0.5
lam3 = 1.0
lam4 = 1.0
lam5 = 5.0
tau = 0
hyperpara = np.array([lam1, lam2, lam3, lam4, lam5, tau])
bvar_set = []
for i in np.arange(0, rt_data.size()):
    presamp = rt_data.getVARData(i).series
    minnpr = MinnesotaPrior(presamp[1:16, :], hyperpara, p=4)
    bvar_set.append(BayesianVAR(minnpr, ny=3, p=4, cons=True))

MN = ForecastingExercise(rt_data, bvar_set)
MN.estimate()
MN.generate_forecast()
MN.evaluate_forecast()
MN.generate_pit_plot()
MN.evaluate_se()
MN.get_rmse()


ntraj = 100

base_model = MN.get_model(0)
parasim = MN.get_mcmc(i)
nhsim = rt_data.size() + 1

yypred = base_model.pred_density(parasim, h=nhsim)

trajdir = "MN/"
if not os.path.exists(trajdir):
    os.mkdir(trajdir)
for i in np.arange(1, ntraj+1):
    trajfile = "traj%i" % i
    trajfile = trajdir + trajfile + ".csv"
    print trajfile
    trajseries = np.vstack((rt_data.getVARData(0).series, yypred[i, ...]))
    np.savetxt(trajfile, trajseries, delimiter = ", ")

pseudo_forecasts = []
for i in np.arange(1, ntraj+1):
    trajfile = "traj%i" % i
    trajfile = trajdir + trajfile + ".csv"
    data_set = var_data.read_from_csv(filename=trajfile, header=False, freq="quarterly", start="1965q2")
    rt_data = real_time_dataset(data_set, estart="1996q1", t0="1965q2")
    pseudo_forecasts.append(ForecastingExercise(rt_data, bvar_set))

    pseudo_forecasts[i-1].estimate()
    pseudo_forecasts[i-1].generate_forecast()
    pseudo_forecasts[i-1].evaluate_forecast()
