#----------------------------------------------------------------------
# Copyright 2018 Victor Coscrato <vcoscrato@gmail.com>;
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
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from scipy.optimize import nnls

class LinearStack(BaseEstimator):
    '''
    Breiman Stacking
    '''
    def __init__(self, estimators = None, verbose = 1):
        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):
        z = np.zeros(len(y_train)).reshape(len(y_train),1)

        for i in range(len(self.estimators)):
            z = np.hstack((z, cross_val_predict(self.estimators[i], x_train, y_train.reshape(-1)).reshape(len(y_train),1)))

            if self.verbose > 1:
                print('Cross-validated: ', self.estimators[i])

            self.estimators[i].fit(x_train, y_train.reshape(-1))
            if self.verbose > 1:
                print('Estimated: ', self.estimators[i])

        adj = nnls(np.delete(z, 0, 1), y_train.reshape(-1))
        self.parameters = adj[0]
        self.train_mse = adj[1]/len(y_train)

        if self.verbose > 0:
            print('Ensemble fitted MSE (train): ', self.train_mse/len(y_train))

        return self

    def predict(self, x):
        z = np.zeros(x.shape[0]).reshape(x.shape[0], 1)
        for i in range(len(self.estimators)):
            z = np.hstack((z, self.estimators[i].predict(x).reshape(x.shape[0], 1)))

        return np.dot(np.delete(z, 0, 1), self.parameters)

    def score(self, x, y):
        return np.mean(np.square(self.predict(x) - np.array(y)))
