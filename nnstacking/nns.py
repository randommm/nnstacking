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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit, KFold

from sklearn import svm, linear_model

import multiprocessing as mp
from copy import deepcopy

from .radam import RAdam

class NNS(BaseEstimator):
    """
    Stacks many estimators using deep foward neural networks.

    Parameters
    ----------
    estimators : list
        List of estimators to use. They must be sklearn-compatible.
    ensemble_method : str
        Chooses the ensembling method. "CNNS" for features to matrix M and
        "UNNS" for features to directly to weights.
    ensemble_addition : bool
        Additional output from the neural network to the ensembler.
    splitter : object
        Chooses the splitting of data to generate the predictions of the estimators. Must be an instance of a class from sklearn.model_selection (or behave similatly), defaults to "ten-fold".
    nworkers : integer
        Number of worker processes to use for parallel fitting the models. If None, then will use all cpus in the machine.

    num_layers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hidden_size : integer
        Number of nodes (neurons) of each hidden layer.
    criterion : object
        Loss criterion for the neural network, defaults to torch.nn.MSELoss().
    nn_weight_decay : object
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score method nor validation of early stopping).

    es : bool
        If true, then will split the training set into training and validation and calculate the validation internally on each epoch and check if the validation loss increases or not.
    es_validation_set : float
        Size of the validation set if es == True.
    es_give_up_after_nepochs : float
        Amount of epochs to try to decrease the validation loss before giving up and stoping training.
    es_splitter_random_state : float
        Random state to split the dataset into training and validation.

    nepoch : integer
        Number of epochs to run. Ignored if es == True.

    batch_initial : integer
        Initial batch size.
    batch_step_multiplier : float
        See batch_inital.
    batch_step_epoch_expon : float
        See batch_inital.
    batch_max_size : float
        See batch_inital.

    batch_test_size : integer
        Size of the batch for validation and score methods.
        Does not affect training efficiency, usefull when there's
        little GPU memory.
    gpu : bool
        If true, will use gpu for computation, if available.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """
    def __init__(self,
                 estimators=None,
                 ensemble_method="CNNS",
                 ensemble_addition=True,
                 splitter=10,
                 nworkers=2,

                 num_layers=3,
                 hidden_size=100,
                 criterion=None,
                 nn_weight_decay=0,

                 es = True,
                 es_validation_set = 0.1,
                 es_give_up_after_nepochs = 50,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=200,
                 batch_step_multiplier=1.1,
                 batch_step_epoch_expon=1.4,
                 batch_max_size=1000,

                 optim_lr=1e-3,

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


    """
    Fit models and neural network.

    Parameters
    ----------
    x_train : array
        Matrix of features
    y_train : array
        Vector of response variable.
    predictions : array
        Array of dimensions (nobs, y_dim, est_dim) with the predictions
        to be used by the model. If None, then will generate them
        automatically using cross validation. If not None, then the
        estimators passed to constructor will not be trained, i.e.
        you must train them before!
    """
    def fit(self, x_train, y_train, predictions=None):
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
                linear_model.Lasso(),
                linear_model.Ridge(),
                ]

        self.est_dim = len(self.estimators)
        self._construct_neural_net()

        if predictions is not None:
            assert(predictions.shape == (self.nobs, self.y_dim,
                                         self.est_dim))
            self.predictions = predictions
            return self.improve_fit(x_train, y_train, self.nepoch)

        if isinstance(self.splitter, int):
            splitter = KFold(n_splits=self.splitter,
                shuffle=True, random_state=0)
        else:
            splitter = self.splitter

        self.predictions = np.empty((self.nobs, self.y_dim,
                                     self.est_dim))

        if self.nworkers == 1:
            self.predictions = np.empty((self.nobs, self.y_dim,
                                         self.est_dim))
            for eind, estimator in enumerate(self.estimators):
                if self.verbose >= 1:
                    print("Calculating prediction for estimator",
                          estimator)

                prediction = np.empty((self.nobs, self.y_dim))
                for tr_in, val_in in splitter.split(x_train, y_train):
                    estimator.fit(x_train[tr_in], y_train[tr_in])
                    prediction_b = estimator.predict(x_train[val_in])
                    if len(prediction_b.shape) == 1:
                        prediction_b = prediction_b[:, None]
                    prediction[val_in] = prediction_b

                prediction = torch.from_numpy(prediction)
                self.predictions[:, :, eind] = prediction

            for estimator in self.estimators:
                if self.verbose >= 1:
                    print("Fitting full estimator", estimator)
                estimator.fit(x_train, y_train)

        else:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(self.nworkers)
            results = []
            for eind, estimator in enumerate(self.estimators):
                result = pool.apply_async(_pfunc,
                    args = (x_train, y_train, splitter, eind, estimator,
                            self.nobs, self.y_dim, self.verbose),
                    error_callback=_perr)
                results.append(result)

            for result in results:
                prediction, eind, estimator = result.get()
                prediction = torch.from_numpy(prediction)
                self.predictions[:, :, eind] = prediction
                self.estimators[eind] = estimator

            pool.close()
            pool.join()

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
        if self.gpu:
            self.move_to_gpu()

        self._check_dims(x_train, y_train)
        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        assert(self.nn_weight_decay >= 0)

        assert(self.num_layers >= 0)
        assert(self.hidden_size > 0)

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
            batch_test_size = min(self.batch_test_size,
                                  x_train.shape[0])
            self.loss_history_validation = []

        batch_max_size = min(self.batch_max_size, nnx_train.shape[0])
        self.loss_history_train = []

        start_time = time.process_time()

        lr = 0.1
        optimizer = RAdam(
            self.neural_net.parameters(),
            lr=self.optim_lr,
            weight_decay=self.nn_weight_decay
        )
        es_penal_tries = 0
        for _ in range_epoch:
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

            try:
                self.neural_net.train()
                self._one_epoch("train", batch_size, batch_test_size,
                                nnx_train, nny_train, nnpred_train,
                                optimizer, volatile=False)

                self.neural_net.eval()
                avloss = self._one_epoch("train", batch_size,
                                         batch_test_size, nnx_train,
                                         nny_train, nnpred_train,
                                         optimizer,
                                         volatile=True)
                self.loss_history_train.append(avloss)

                if self.es:
                    self.neural_net.eval()
                    avloss = self._one_epoch("val", batch_size,
                        batch_test_size, nnx_train,
                        nny_train, nnpred_train, optimizer,
                        volatile=True)
                    self.loss_history_validation.append(avloss)
                    if avloss <= self.best_loss_val:
                        self.best_loss_val = avloss
                        best_state_dict = self.neural_net.state_dict()
                        best_state_dict = deepcopy(best_state_dict)
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
                                  self.es_give_up_after_nepochs, "tries.",
                                  "Stopping")
                        break

                self.epoch_count += 1
            except RuntimeError as err:
                #if self.epoch_count == 0:
                #    raise err
                if self.verbose >= 2:
                    print("Runtime error problem probably due to",
                           "high learning rate.")
                    print("Decreasing learning rate by half.")

                self._construct_neural_net()
                if self.gpu:
                    self.move_to_gpu()
                lr /= 2
                optimizer = optim.Adamax(self.neural_net.parameters(),
                    lr=lr, weight_decay=self.nn_weight_decay)
                self.epoch_count = 0

                continue
            except KeyboardInterrupt:
                if self.epoch_count > 0 and self.es:
                    print("Keyboard interrupt detected.",
                          "Switching weights to lowest validation loss",
                          "and exiting")
                    self.neural_net.load_state_dict(best_state_dict)
                break

        elapsed_time = time.process_time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def _one_epoch(self, ftype, batch_train_size, batch_test_size,
                   nnx, nny, nnpred, optimizer, volatile):
        with torch.set_grad_enabled(not volatile):
            if volatile:
                batch_size = batch_test_size
            else:
                batch_size = batch_train_size

            if ftype == "train":
                batch_show_size = batch_train_size
            else:
                batch_show_size = batch_test_size

            nnx = torch.from_numpy(nnx)
            nny = torch.from_numpy(nny)
            nnpred = torch.from_numpy(nnpred)
            if self.gpu:
                nnx = nnx.pin_memory()
                nny = nny.pin_memory()
                nnpred = nnpred.pin_memory()

            loss_vals = []
            batch_sizes = []
            for i in range(0, nny.shape[0] + batch_size, batch_size):
                if i < nny.shape[0]:
                    nnx_next = nnx[i:i+batch_size]
                    nny_next = nny[i:i+batch_size]
                    nnpred_next = nnpred[i:i+batch_size]

                    if self.gpu:
                        nnx_next = nnx_next.cuda(non_blocking=True)
                        nny_next = nny_next.cuda(non_blocking=True)
                        nnpred_next =nnpred_next.cuda(non_blocking=True)

                if i != 0:
                    batch_actual_size = nnx_this.shape[0]
                    if batch_actual_size != batch_size and not volatile:
                        continue

                    optimizer.zero_grad()
                    output = self._ensemblize(nnx_this,                       nnpred_this)

                    # Main loss
                    loss = self.criterion(output, nny_this)

                    # Correction for last batch as it might be smaller
                    #if batch_actual_size != batch_size:
                    #    loss *= batch_actual_size / batch_size

                    np_loss = loss.data.cpu().numpy()
                    if np.isnan(np_loss):
                        raise RuntimeError("Loss is NaN")

                    loss_vals.append(np_loss)
                    batch_sizes.append(batch_actual_size)

                    if not volatile:
                        loss.backward()
                        optimizer.step()

                nnx_this = nnx_next
                nny_this = nny_next
                nnpred_this = nnpred_next

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2 and volatile:
                print("Finished epoch", self.epoch_count,
                      "with batch size", batch_show_size,
                      "and", ftype + " loss",
                      avgloss, flush=True)

            return avgloss

    def _calculate_weights(self, nnx):
        output = self.neural_net(nnx)

        if self.ensemble_addition:
            output, extra = output[:, :-1], output[:, [-1]]
        else:
            extra = 0

        if self.ensemble_method == "CNNS":
            output = output.view(-1, self.est_dim,
                self.est_dim)
            output_res = output.new(output.shape[0],
                self.est_dim)
            evec = output.new_ones(self.est_dim)[:, None]
            for i in range(output.shape[0]):
                div_res = output[i]
                #output[i] = output[i].potri()
                div_res = div_res.tril()
                div_res = torch.mm(div_res, div_res.t())
                numerator = torch.mm(div_res, evec)
                denominator = torch.mm(numerator.t(), evec)
                div_res = numerator / denominator
                output_res[i] = div_res[:, 0]
            output = output_res

        return output, extra

    def get_weights(self, x_pred):
        """
        Get the ensembler weights given a x_pred covariate matrix

        Returns
        -------
        Tuple (output, extra), where output is the weights and extra is
        the additional output from the neural network (by definition,
        extra is 0 if self.ensemble_addition == False)
        """
        with torch.no_grad():
            nnx = _np_to_var(x_pred)
            if self.gpu:
                nnx = nnx.cuda()
            output, extra = self._calculate_weights(nnx)
        return output, extra

    def _ensemblize(self, nnx, nnpred):
        output, extra = self._calculate_weights(nnx)

        output = nnpred * output[:, None, :]
        output = output.sum(2)

        if self.ensemble_addition:
            output = output + extra

        return output

    """
    Calculate the opposite of mean squared error between prediction and
    x_test; i.e.: (- (self.pred(x_test) - y_test)**2).mean()

    Parameters
    ----------
    x_test : array
        Matrix of features
    y_test : array
        Vector of response variable.
    """
    def score(self, x_test, y_test):
        with torch.no_grad():
            self._check_dims(x_test, y_test)

            predictions = np.empty((x_test.shape[0], y_test.shape[1],
                                         self.est_dim))
            for eind, estimator in enumerate(self.estimators):
                if self.verbose >= 1:
                    print("Calculating prediction for estimator",
                           estimator)
                prediction = estimator.predict(x_test)
                if len(prediction.shape) == 1:
                    prediction = prediction[:, None]
                predictions[:, :, eind] = torch.from_numpy(prediction)

            self.neural_net.eval()
            nnx = _np_to_var(x_test)
            nny = _np_to_var(y_test)
            nnpred = _np_to_var(predictions)

            if self.gpu:
                nnx = nnx.pin_memory()
                nny = nny.pin_memory()
                nnpred = nnpred.pin_memory()

            batch_size = min(self.batch_test_size, x_test.shape[0])

            loss_vals = []
            batch_sizes = []
            for i in range(0, nny.shape[0] + batch_size, batch_size):
                if i < nny.shape[0]:
                    nnx_next = nnx[i:i+batch_size]
                    nny_next = nny[i:i+batch_size]
                    nnpred_next = nnpred[i:i+batch_size]

                    if self.gpu:
                        nnx_next = nnx_next.cuda(non_blocking=True)
                        nny_next = nny_next.cuda(non_blocking=True)
                        nnpred_next =nnpred_next.cuda(non_blocking=True)

                if i != 0:
                    output = self._ensemblize(nnx_this,
                        nnpred_this)
                    loss = self.criterion(output, nny_this)

                    loss_vals.append(loss.data.cpu().numpy())
                    batch_sizes.append(nnx_this.shape[0])

                nnx_this = nnx_next
                nny_this = nny_next
                nnpred_this = nnpred_next

            return -1 * np.average(loss_vals, weights=batch_sizes)

    """
    Predict y.

    Parameters
    ----------
    x_pred : array
        Matrix of features
    """
    def predict(self, x_pred):
        with torch.no_grad():
            self._check_dims(x_pred, np.empty((1,1)))

            for eind, estimator in enumerate(self.estimators):
                if self.verbose >= 1:
                    print("Calculating prediction for estimator",
                          estimator)
                prediction = estimator.predict(x_pred)
                if len(prediction.shape) == 1:
                    prediction = prediction[:, None]
                if eind == 0:
                    predictions = np.empty((x_pred.shape[0],
                                            prediction.shape[1],
                                            self.est_dim))
                predictions[:, :, eind] = torch.from_numpy(prediction)

            self.neural_net.eval()
            nnx = _np_to_var(x_pred)
            nnpred = _np_to_var(predictions)

            if self.gpu:
                nnx = nnx.cuda()
                nnpred = nnpred.cuda()

            output = self._ensemblize(nnx, nnpred)

            return output.data.cpu().numpy()

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, input_dim, output_dim, num_layers,
                         output_hl_size, softmax):
                super(NeuralNet, self).__init__()

                next_input_l_size = input_dim
                self.m = nn.Dropout(p=0.5)

                if softmax:
                    self.transform = torch.nn.Softmax(-1)
                else:
                    self.transform = _dummy_func

                self.llayers = []
                self.normllayers = []
                for i in range(num_layers):
                    self.llayers.append(
                        nn.Linear(next_input_l_size, output_hl_size))
                    self.normllayers.append(
                        nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(self.llayers[i])
                self.llayers = nn.ModuleList(self.llayers)
                self.normllayers = nn.ModuleList(self.normllayers)

                self.fc_last = nn.Linear(next_input_l_size, output_dim)
                self._initialize_layer(self.fc_last)

                self.num_layers = num_layers
                self.output_dim = output_dim

            def forward(self, x):
                for i in range(self.num_layers):
                    x = F.elu(self.llayers[i](x))
                    x = self.normllayers[i](x)
                    x = self.m(x)
                x = self.fc_last(x)
                x = self.transform(x)
                return x

            def _initialize_layer(self, layer):
                nn.init.constant_(layer.bias, 0)
                gain=nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(layer.weight, gain=gain)

        if self.ensemble_method == "UNNS":
            output_dim = self.est_dim
            softmax = False
        elif self.ensemble_method == "CNNS":
            output_dim = self.est_dim ** 2
            softmax = True
        else:
            raise ValueError("ensemble_method must be UNNS or CNNS")
        if self.ensemble_addition:
            output_dim += 1
        output_hl_size = int(self.hidden_size)
        self.neural_net = NeuralNet(self.x_dim, output_dim,
                                    self.num_layers, output_hl_size,
                                    softmax)

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

def _np_to_var(arr):
    arr = np.array(arr, dtype='f4')
    arr = torch.from_numpy(arr)
    return arr

def _pfunc(x_train, y_train, splitter, eind, estimator, nobs, y_dim,
          verbose):
    if verbose >= 1:
        print("Calculating prediction for estimator",
              estimator)

    prediction = np.empty((nobs, y_dim))
    for tr_in, val_in in splitter.split(x_train, y_train):
        estimator.fit(x_train[tr_in], y_train[tr_in])
        prediction_b = estimator.predict(x_train[val_in])
        if len(prediction_b.shape) == 1:
            prediction_b = prediction_b[:, None]
        prediction[val_in] = prediction_b

    if verbose >= 1:
        print("Fitting full estimator", estimator)
    estimator.fit(x_train, y_train)

    return prediction, eind, estimator

def _perr(err):
    print("Error during multiprocessing:", err)

def _dummy_func(x):
    return x
