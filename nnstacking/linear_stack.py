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
    """
    Stacks many estimators using Breiman Stacking.

    Parameters
    ----------
    estimators : list
        List of estimators to use. They must be sklearn-compatible.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """
    def __init__(self, estimators = None, verbose = 1):
        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    """
    Fit models and Breiman Stacking coefficients.

    Parameters
    ----------
    x_train : array
        Matrix of features
    y_train : array
        Vector of response variable.
    predictions : array
        Array of dimensions (nobs, est_dim) with the predictions
        to be used by the model. If None, then will generate them
        automatically using cross validation. If not None, then the
        estimators passed to constructor will not be trained, i.e.
        you must train them before!
    """
    def fit(self, x_train, y_train, predictions=None):

        if predictions is not None:
            assert(predictions.shape == (len(x_train), len(self.estimators)))

        else:
            predictions = np.empty((len(train), len(base_est)))
            for i, est in enumerate(self.estimators):
                if self.verbose > 1:
                    print('Cross-validating: ', est)
                predictions[:, i] = cross_val_predict(est, x[train], y[train], cv=2)
                if self.verbose > 1:
                    print('Estimating: ', self.estimators[i])
                est.fit(x[train], y[train])
                cv_predictions[test, i] = est.predict(x[test])

        adj = nnls(predictions, y_train.reshape(-1))
        self.parameters = adj[0]
        self.train_mse = adj[1]/len(y_train)

        if self.verbose > 0:
            print('Ensemble fitted MSE (train): ', self.train_mse/len(y_train))

        return self

    """
    Predict y.

    Parameters
    ----------
    x : array
        Matrix of features
    """
    def predict(self, x):
        predictions = np.empty((len(x), len(self.estimators)))
        for i, est in enumerate(self.estimators):
            predictions[:, i] = est.predict(x)

        return np.dot(predictions, self.parameters)

    """
    Calculate the opposite of mean squared error between prediction and
    x; i.e.: (- (self.pred(x) - y)**2).mean()

    Parameters
    ----------
    x : array
        Matrix of features
    y : array
        Vector of response variable.
    """
    def score(self, x, y):
        return np.mean(np.square(self.predict(x) - np.array(y)))
