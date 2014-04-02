from __future__ import division
from var_data import var_data, real_time_dataset
from varprior import DiffusePrior, MinnesotaPrior
from pyvar import BayesianVAR, CompleteModel, ForecastingExercise
import numpy as np
import csv
import os

data_set = var_data.read_from_csv(filename="/mq/home/m1eph00/other-peoples-code/noe-var-analysis/US_EURO.TXT", header=False)
