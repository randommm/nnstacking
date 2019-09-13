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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import numpy as np
import time
import itertools
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
from .radam import RAdam

def _np_to_tensor(arr):
    arr = np.array(arr, dtype='f4')
    arr = torch.from_numpy(arr)
    return arr

class NNPredict(BaseEstimator):
    """
    Regression estimation using neural networks

    Parameters
    ----------
    nn_weight_decay : object
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score method nor validation of early stopping).
    num_layers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hidden_size : integer
        Number of nodes (neurons) of each hidden layer.
    criterion : object
        Loss criterion for the neural network, defaults to torch.nn.MSELoss().

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

    dataloader_workers : int
        Number of parallel workers for the Pytorch dataloader.

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
                 nn_weight_decay=0,
                 num_layers=10,
                 hidden_size=100,
                 convolutional=False,

                 es = True,
                 es_validation_set_size = None,
                 es_give_up_after_nepochs = 50,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=300,
                 batch_step_multiplier=1.4,
                 batch_step_epoch_expon=2.0,
                 batch_max_size=1000,
                 dataloader_workers=1,

                 optim_lr=1e-3,

                 batch_test_size=2000,
                 gpu=True,
                 verbose=1,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):
        self.gpu = self.gpu and torch.cuda.is_available()

        self.x_dim = x_train.shape[1]
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]
        self.y_dim = y_train.shape[1]

        self._construct_neural_net()
        self.epoch_count = 0

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

    def improve_fit(self, x_train, y_train, nepoch=1):
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]
        criterion = nn.MSELoss()

        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        assert(self.num_layers >= 0)
        assert(self.hidden_size > 0)

        inputv_train = np.array(x_train, dtype='f4')
        target_train = np.array(y_train, dtype='f4')

        range_epoch = range(nepoch)
        if self.es:
            es_validation_set_size = self.es_validation_set_size
            if es_validation_set_size is None:
                es_validation_set_size = round(
                    min(x_train.shape[0] * 0.10, 5000))
            splitter = ShuffleSplit(n_splits=1,
                test_size=es_validation_set_size,
                random_state=self.es_splitter_random_state)
            index_train, index_val = next(iter(splitter.split(x_train,
                y_train)))
            self.index_train = index_train
            self.index_val = index_val

            inputv_val = inputv_train[index_val]
            target_val = target_train[index_val]
            inputv_val = np.ascontiguousarray(inputv_val)
            target_val = np.ascontiguousarray(target_val)

            inputv_train = inputv_train[index_train]
            target_train = target_train[index_train]
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count() # infty iterator

            batch_test_size = min(self.batch_test_size,
                                  inputv_val.shape[0])
            self.loss_history_validation = []

        batch_max_size = min(self.batch_max_size, inputv_train.shape[0])
        self.loss_history_train = []

        start_time = time.time()

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

            permutation = np.random.permutation(target_train.shape[0])
            inputv_train = torch.from_numpy(inputv_train[permutation])
            target_train = torch.from_numpy(target_train[permutation])
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            try:
                self.neural_net.train()
                self._one_epoch(True, batch_size, inputv_train,
                                target_train, optimizer, criterion)

                if self.es:
                    self.neural_net.eval()
                    avloss = self._one_epoch(False, batch_test_size,
                        inputv_val, target_val, optimizer, criterion)
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

        elapsed_time = time.time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def _one_epoch(self, is_train, batch_size, inputv, target,
        optimizer, criterion):
        with torch.set_grad_enabled(is_train):
            inputv = torch.from_numpy(inputv)
            target = torch.from_numpy(target)

            loss_vals = []
            batch_sizes = []

            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=True, drop_last=is_train,
                pin_memory=self.gpu,
                num_workers=self.dataloader_workers)

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)

                batch_actual_size = inputv_this.shape[0]
                optimizer.zero_grad()
                output = self.neural_net(inputv_this)
                loss = criterion(output, target_this)

                np_loss = loss.data.item()
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if is_train:
                    loss.backward()
                    optimizer.step()

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2:
                print("Finished epoch", self.epoch_count,
                      "with batch size", batch_size, "and",
                      ("train" if is_train else "validation"),
                      "loss", avgloss, flush=True)

            return avgloss

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
        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(np.ascontiguousarray(x_test))
            target = _np_to_tensor(y_test)

            batch_size = min(self.batch_test_size, x_test.shape[0])

            loss_vals = []
            batch_sizes = []

            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=True, drop_last=False,
                pin_memory=self.gpu,
                num_workers=self.dataloader_workers)

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)

                batch_actual_size = inputv_this.shape[0]
                output = self.neural_net(inputv_this)
                criterion = nn.MSELoss()
                loss = criterion(output, target_this)

                loss_vals.append(loss.data.item())
                batch_sizes.append(batch_actual_size)

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
            self.neural_net.eval()
            inputv = _np_to_tensor(x_pred)

            if self.gpu:
                inputv = inputv.cuda()

            output_pred = self.neural_net(inputv)

            return output_pred.data.cpu().numpy()

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, x_dim, y_dim, num_layers,
                         hidden_size, convolutional):
                super(NeuralNet, self).__init__()

                output_hl_size = int(hidden_size)
                self.dropl = nn.Dropout(p=0.5)
                self.convolutional = convolutional
                next_input_l_size = x_dim

                if self.convolutional:
                    next_input_l_size = 1
                    self.nclayers = 4
                    clayers = []
                    polayers = []
                    normclayers = []
                    for i in range(self.nclayers):
                        if next_input_l_size == 1:
                            output_hl_size = 16
                        else:
                            output_hl_size = 32
                        clayers.append(nn.Conv1d(next_input_l_size,
                            output_hl_size, kernel_size=5, stride=1,
                            padding=2))
                        polayers.append(nn.MaxPool1d(stride=1,
                            kernel_size=5, padding=2))
                        normclayers.append(nn.BatchNorm1d(output_hl_size))
                        next_input_l_size = output_hl_size
                        self._initialize_layer(clayers[i])
                    self.clayers = nn.ModuleList(clayers)
                    self.polayers = nn.ModuleList(polayers)
                    self.normclayers = nn.ModuleList(normclayers)

                    faked = torch.randn(2, 1, x_dim)
                    for i in range(self.nclayers):
                        faked = polayers[i](clayers[i](faked))
                    faked = faked.view(faked.size(0), -1)
                    next_input_l_size = faked.size(1)
                    del(faked)

                llayers = []
                normllayers = []
                for i in range(num_layers):
                    llayers.append(nn.Linear(next_input_l_size,
                                             output_hl_size))
                    normllayers.append(nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(llayers[i])

                self.llayers = nn.ModuleList(llayers)
                self.normllayers = nn.ModuleList(normllayers)

                self.fc_last = nn.Linear(next_input_l_size, y_dim)
                self._initialize_layer(self.fc_last)
                self.num_layers = num_layers

            def forward(self, x):
                if self.convolutional:
                    x = x[:, None]
                    for i in range(self.nclayers):
                        fc = self.clayers[i]
                        fpo = self.polayers[i]
                        fcn = self.normclayers[i]
                        x = fcn(F.elu(fc(x)))
                        x = fpo(x)
                    x = x.view(x.size(0), -1)

                for i in range(self.num_layers):
                    fc = self.llayers[i]
                    fcn = self.normllayers[i]
                    x = fcn(F.elu(fc(x)))
                    x = self.dropl(x)
                x = self.fc_last(x)

                return x

            def _initialize_layer(self, layer):
                nn.init.constant_(layer.bias, 0)
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(layer.weight, gain=gain)

        self.neural_net = NeuralNet(self.x_dim, self.y_dim,
                                    self.num_layers, self.hidden_size,
                                    self.convolutional)

    def __getstate__(self):
        d = self.__dict__.copy()
        if hasattr(self, "neural_net"):
            state_dict = self.neural_net.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].cpu()
            d["neural_net_params"] = state_dict
            del(d["neural_net"])

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del(self.neural_net_params)
            if self.gpu:
                if torch.cuda.is_available():
                    self.move_to_gpu()
                else:
                    self.gpu = False
                    print("Warning: GPU was used to train this model, "
                          "but is not currently available and will "
                          "be disabled "
                          "(renable with estimator move_to_gpu)")
