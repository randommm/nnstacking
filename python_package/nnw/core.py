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
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------
from __future__ import division

import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ShuffleSplit

from sklearn import svm, linear_model


class NNW(BaseEstimator):
    def __init__(self,
                 estimators=None,
                 weightining_method="cov_to_weights",
                 fit_estimators = True,

                 nn_weights_loss_penal=0,
                 nhlayers=1,
                 hls_multiplier=5,
                 criterion=None,

                 es = True,
                 es_validation_set = 0.1,
                 es_give_up_after_nepochs = 20,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=400,
                 batch_step_multiplier=1.1,
                 batch_step_epoch_expon=1.1,
                 batch_max_size=500,

                 batch_test_size=2000,
                 gpu=True,
                 verbose=1,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def _check_dims(self, x_tc, y_tc):
        if len(x_tc.shape) == 1 or len(y_tc.shape) == 1:
            raise ValueError("x and y must have shape (s, f) "
                             "where s is the sample size and "
                             "f is the number of features")

    def fit(self, x_train, y_train):
        self.gpu = self.gpu and torch.cuda.is_available()

        self._check_dims(x_train, y_train)

        self.x_dim = x_train.shape[1]
        self.y_dim = y_train.shape[1]
        self.epoch_count = 0
        self.nobs = x_train.shape[0]

        if self.criterion is None:
            self.criterion = nn.MSELoss()

        if self.estimators is None:
            self.estimators = [
                linear_model.LinearRegression(),

                linear_model.Lasso(alpha=0.5),
                linear_model.Lasso(alpha=1.0),
                linear_model.Lasso(alpha=2.0),

                linear_model.Ridge(alpha=0.5),
                linear_model.Ridge(alpha=1.0),
                linear_model.Ridge(alpha=2.0),

                linear_model.ElasticNet(alpha=0.5),
                linear_model.ElasticNet(alpha=1.0),
                linear_model.ElasticNet(alpha=2.0),

                linear_model.LassoLars(alpha=0.5),
                linear_model.LassoLars(alpha=1.0),
                linear_model.LassoLars(alpha=2.0),

                #svm.SVC()
                ]

        self.est_dim = len(self.estimators)
        self._construct_neural_net()

        if self.fit_estimators:
            for estimator in self.estimators:
                if self.verbose >= 1:
                    print("Fitting estimator", estimator)
                estimator.fit(x_train, y_train)

        self.predictions = np.empty((self.nobs, self.y_dim,
                                     self.est_dim))
        for eind, estimator in enumerate(self.estimators):
            if self.verbose >= 1:
                print("Calculating prediction for estimator", estimator)
            prediction = estimator.predict(x_train)
            if len(prediction.shape) == 1:
                prediction = prediction[:, None]
            self.predictions[:,:,eind] = torch.from_numpy(prediction)

        if self.gpu:
            self.move_to_gpu()

        return self.improve_fit(x_train, y_train, self.nepoch)

    def move_to_gpu(self):
        self.neural_net.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.neural_net.cpu()
        self.gpu = False

        return self

    def improve_fit(self, x_train, y_train, nepoch):
        self._check_dims(x_train, y_train)
        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        assert(self.nn_weights_loss_penal >= 0)

        assert(self.nhlayers >= 0)
        assert(self.hls_multiplier > 0)

        nnx_train = np.array(x_train, dtype='f4')
        nny_train = np.array(y_train, dtype='f4')
        nnpred_train = np.array(self.predictions, dtype='f4')

        range_epoch = range(nepoch)
        if self.es:
            splitter = ShuffleSplit(n_splits=1,
                test_size=self.es_validation_set,
                random_state=self.es_splitter_random_state)
            index_train, index_val = next(iter(splitter.split(x_train,
                y_train)))

            nnx_val = nnx_train[index_val]
            nny_val = nny_train[index_val]
            nnpred_val = nnpred_train[index_val]
            nnx_val = np.ascontiguousarray(nnx_val)
            nny_val = np.ascontiguousarray(nny_val)
            nnpred_val = np.ascontiguousarray(nnpred_val)

            nnx_train = nnx_train[index_train]
            nny_train = nny_train[index_train]
            nnpred_train = nnpred_train[index_train]
            nnx_train = np.ascontiguousarray(nnx_train)
            nny_train = np.ascontiguousarray(nny_train)
            nnpred_train = np.ascontiguousarray(nnpred_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count() # infty iterator

        batch_max_size = min(self.batch_max_size, x_train.shape[0])
        batch_test_size = min(self.batch_test_size, x_train.shape[0])

        start_time = time.process_time()

        optimizer = optim.Adam(self.neural_net.parameters())
        self.nn_weights_loss_penal = 100
        for _ in range_epoch:
            self.nn_weights_loss_penal /= 2.0
            batch_size = int(min(batch_max_size,
                self.batch_initial +
                self.batch_step_multiplier *
                self.epoch_count ** self.batch_step_epoch_expon))

            permutation = np.random.permutation(nny_train.shape[0])
            nnx_train = nnx_train[permutation]
            nny_train = nny_train[permutation]
            nnpred_train = nnpred_train[permutation]
            nnx_train = np.ascontiguousarray(nnx_train)
            nny_train = np.ascontiguousarray(nny_train)
            nnpred_train = np.ascontiguousarray(nnpred_train)

            self.neural_net.train()
            self._one_epoch("train", batch_size, nnx_train,
                nny_train, nnpred_train, optimizer,
                volatile=False)
            if self.es:
                self.neural_net.eval()
                avloss = self._one_epoch("test", batch_test_size,
                    nnx_val, nny_val, nnpred_val, optimizer,
                    volatile=True)
                if avloss <= self.best_loss_val:
                    self.best_loss_val = avloss
                    best_state_dict = self.neural_net.state_dict()
                    es_tries = 0
                    if self.verbose >= 2:
                        print("This is the lowest validation loss",
                              "so far.")
                else:
                    es_tries += 1
                if es_tries >= self.es_give_up_after_nepochs:
                    self.neural_net.load_state_dict(best_state_dict)
                    if self.verbose >= 1:
                        print("Validation loss did not improve after",
                              self.es_give_up_after_nepochs, "tries.")
                    break

            self.epoch_count += 1

        elapsed_time = time.process_time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def _one_epoch(self, ftype, batch_size, nnx, nny, nnpred,
                   optimizer, volatile):

        nnx = torch.from_numpy(nnx)
        nny = torch.from_numpy(nny)
        nnpred = torch.from_numpy(nnpred)
        if self.gpu:
            nnx = nnx.pin_memory()
            nny = nny.pin_memory()
            nnpred = nnpred.pin_memory()
        nnx = Variable(nnx, volatile=volatile)
        nny = Variable(nny, volatile=volatile)
        nnpred = Variable(nnpred, volatile=volatile)

        loss_vals = []
        batch_sizes = []
        for i in range(0, nny.shape[0] + batch_size, batch_size):
            if i < nny.shape[0]:
                nnx_next = nnx[i:i+batch_size]
                nny_next = nny[i:i+batch_size]
                nnpred_next = nnpred[i:i+batch_size]

                if self.gpu:
                    nnx_next = nnx_next.cuda(async=True)
                    nny_next = nny_next.cuda(async=True)
                    nnpred_next = nnpred_next.cuda(async=True)

            if i != 0:
                batch_actual_size = nnx_this.shape[0]
                if batch_actual_size != batch_size and ftype == "train":
                    continue

                optimizer.zero_grad()
                output = self.neural_net(nnx_this)
                output = nnpred_this * output[:, None, :]
                output = output.sum(2)

                # Main loss
                loss = self.criterion(output, nny_this)

                # Penalize on nn weights
                if self.nn_weights_loss_penal != 0 and ftype == "train":
                    penal = self.neural_net.parameters()
                    penal = map(lambda x: (x**2).sum(), penal)
                    penal = Variable.cat(tuple(penal)).sum()
                    loss += penal * self.nn_weights_loss_penal

                # Correction for last batch as it might be smaller
                if batch_actual_size != batch_size:
                    loss *= batch_actual_size / batch_size

                np_loss = loss.data.cpu().numpy()[0]
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if ftype == "train":
                    loss.backward()
                    optimizer.step()

            nnx_this = nnx_next
            nny_this = nny_next
            nnpred_this = nnpred_next

        avgloss = np.average(loss_vals, weights=batch_sizes)
        if self.verbose >= 2:
            print("Finished epoch", self.epoch_count,
                  "with batch size", batch_size,
                  "and", ftype,
                  ("pseudo-" if ftype == "train" else "") + "loss",
                  avgloss, flush=True)

        return avgloss

    def score(self, x_test, y_test):
        self._check_dims(x_test, y_test)

        predictions = np.empty((x_test.shape[0], y_test.shape[1],
                                     self.est_dim))
        for eind, estimator in enumerate(self.estimators):
            if self.verbose >= 1:
                print("Calculating prediction for estimator", estimator)
            prediction = estimator.predict(x_test)
            if len(prediction.shape) == 1:
                prediction = prediction[:, None]
            predictions[:, :, eind] = torch.from_numpy(prediction)

        self.neural_net.eval()
        nnx = _np_to_var(x_test, volatile=True)
        nny = _np_to_var(y_test, volatile=True)
        nnpred = _np_to_var(predictions, volatile=True)

        if self.gpu:
            nnx = Variable(nnx.data.pin_memory(), volatile=True)
            nny = Variable(nny.data.pin_memory(), volatile=True)
            nnpred = Variable(nnpred.data.pin_memory(), volatile=True)

        batch_size = min(self.batch_test_size, x_test.shape[0])

        loss_vals = []
        batch_sizes = []
        for i in range(0, nny.shape[0] + batch_size, batch_size):
            if i < nny.shape[0]:
                nnx_next = nnx[i:i+batch_size]
                nny_next = nny[i:i+batch_size]
                nnpred_next = nnpred[i:i+batch_size]

                if self.gpu:
                    nnx_next = nnx_next.cuda(async=True)
                    nny_next = nny_next.cuda(async=True)
                    nnpred_next = nnpred_next.cuda(async=True)

            if i != 0:
                output = self.neural_net(nnx_this)
                output = nnpred_this * output[:, None, :]
                output = output.sum(2)

                loss = self.criterion(output, nny_this)

                loss_vals.append(loss.data.cpu().numpy()[0])
                batch_sizes.append(nnx_this.shape[0])

            nnx_this = nnx_next
            nny_this = nny_next
            nnpred_this = nnpred_next

        return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict(self, x_pred):
        self._check_dims(x_pred, np.empty((1,1)))

        for eind, estimator in enumerate(self.estimators):
            if self.verbose >= 1:
                print("Calculating prediction for estimator", estimator)
            prediction = estimator.predict(x_pred)
            if len(prediction.shape) == 1:
                prediction = prediction[:, None]
            if eind == 0:
                predictions = np.empty((x_pred.shape[0],
                                        prediction.shape[1],
                                        self.est_dim))
            predictions[:, :, eind] = torch.from_numpy(prediction)

        self.neural_net.eval()
        nnx = _np_to_var(x_pred, volatile=True)
        nnpred = _np_to_var(predictions, volatile=True)

        if self.gpu:
            nnx = nnx.cuda()
            nnpred = nnpred.cuda()

        output = self.neural_net(nnx)
        output = nnpred * output[:, None, :]
        output = output.sum(2)

        return output.data.cpu().numpy()

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, input_dim, output_dim, nhlayers,
                         hls_multiplier):
                super(NeuralNet, self).__init__()

                next_input_l_size = input_dim
                output_hl_size = int(output_dim * hls_multiplier)
                self.m = nn.AlphaDropout(p=0.5)

                for i in range(nhlayers):
                    lname = "fc_" + str(i)
                    lnname = "fc_n_" + str(i)
                    self.__setattr__(lname,
                        nn.Linear(next_input_l_size, output_hl_size))
                    self.__setattr__(lnname,
                        nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(self.__getattr__(lname))

                self.fc_last = nn.Linear(next_input_l_size, output_dim)
                self._initialize_layer(self.fc_last)

                self.nhlayers = nhlayers
                self.output_dim = output_dim
                self.softmax = torch.nn.Softmax(-1)

            def forward(self, x):
                for i in range(self.nhlayers):
                    fc = self.__getattr__("fc_" + str(i))
                    fcn = self.__getattr__("fc_n_" + str(i))
                    x = fcn(F.relu(fc(x)))
                    self.m(x)
                x = self.fc_last(x)
                x = self.softmax(x)
                return x

            def _initialize_layer(self, layer):
                nn.init.constant(layer.bias, 0)
                gain=nn.init.calculate_gain('relu')
                nn.init.xavier_normal(layer.weight, gain=gain)

        self.neural_net = NeuralNet(self.x_dim, self.est_dim,
                                    self.nhlayers, self.hls_multiplier)

    def __getstate__(self):
        d = self.__dict__.copy()
        if hasattr(self, "neural_net"):
            state_dict = self.neural_net.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].cpu()
            d["neural_net_params"] = state_dict
            del(d["neural_net"])

        #Delete phi_grid (will recreate on load)
        if hasattr(self, "phi_grid"):
            del(d["phi_grid"])
            d["y_grid"] = None

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del(self.neural_net_params)
            if self.gpu:
                self.move_to_gpu()

        #Recreate phi_grid
        if "y_grid" in d.keys():
            del(self.y_grid)
            self._create_phi_grid()

def _np_to_var(arr, volatile=False):
    arr = np.array(arr, dtype='f4')
    arr = Variable(torch.from_numpy(arr), volatile=volatile)
    return arr
