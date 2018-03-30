#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import torch
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats

from nnw import NNW
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import hashlib
import pickle
from sklearn.externals import joblib
import os

from generate_data import generate_data, true_pdf_calc

n_train = 2_000
n_test = 800
x_train, y_train = generate_data(n_train)
x_test, y_test = generate_data(n_test)

print(y_train)
print(min(y_train))
print(max(y_train))

nnw_obj = NNW(
verbose=2,
nn_weights_loss_penal=0.0,
es=True,
hls_multiplier=1,
nhlayers=1,
gpu=False,
)

nnw_obj.fit(x_train, y_train)

nnw_obj.verbose = 0
print("Risk on train:", - nnw_obj.score(x_train, y_train))
for i, estimator in enumerate(nnw_obj.estimators):
    print("Risk on train for estimator", i, "is:",
          ((estimator.predict(x_train) - y_train)**2).mean()
         )

print("Risk on test:", - nnw_obj.score(x_test, y_test))
for i, estimator in enumerate(nnw_obj.estimators):
    print("Risk on test for estimator", i, "is:",
          ((estimator.predict(x_test) - y_test)**2).mean()
         )

"""
#Check using true density information
est_pdf = nnw_obj.predict(x_test)[:, 1:-1]
true_pdf = true_pdf_calc(x_test, nnw_obj.y_grid[1:-1][:,None]).T
sq_errors = (est_pdf - true_pdf)**2
#print("Squared density errors for test:\n", sq_errors)
print("\nAverage squared density errors for test:\n", sq_errors.mean())

import matplotlib.pyplot as plt
plt.plot(true_pdf[1])
plt.plot(est_pdf[1])
plt.show()
"""
