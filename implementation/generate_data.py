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

#Comment for non-deterministic results
np.random.seed(10)

x_dim = 50

beta = stats.norm.rvs(size=x_dim, scale=0.4)
beta0 = -.3
sigma = .5

def func(x):
    x_transf = x.copy()
    for i in range(0, x_dim-5, 5):
        x_transf[i] = np.abs(x[i])**1.3
        x_transf[i+1] = np.cos(x[i+1])
        x_transf[i+2] = x[i]*x[i+2]
        x_transf[i+4] = np.sqrt(np.abs(x[i+4]))
        x_transf[i+5] = x[i+5] * np.sin(x[i])
    return np.dot(beta, x_transf)

def true_pdf_calc(x_pred, y_pred):
    logit_y_pred = - np.log(1/y_pred - 1)
    mu = np.apply_along_axis(func, 1, x_pred) + beta0
    density = 0.4 * stats.skewnorm.pdf(logit_y_pred, loc=mu + 1.4, scale=sigma, a=4)
    density += 0.3 * stats.skewnorm.pdf(logit_y_pred, loc=mu - .2, scale=sigma, a=4)
    density += 0.2 * stats.skewnorm.pdf(logit_y_pred, loc=mu, scale=sigma, a=4)
    density += 0.1 * stats.skewnorm.pdf(logit_y_pred, loc=mu - 1.8, scale=sigma, a=4)
    density /= np.abs(y_pred - y_pred**2)
    return density

def generate_data(n_gen):
    x_gen = stats.skewnorm.rvs(scale=0.1, size=n_gen*x_dim, a=2)
    x_gen = x_gen.reshape((n_gen, x_dim))

    mu_gen = np.apply_along_axis(func, 1, x_gen)

    y_gen = stats.skewnorm.rvs(loc=beta0, scale=sigma, size=n_gen, a=4)
    y_gen = mu_gen + y_gen

    rv = stats.multinomial(1, [0.4, 0.3, 0.2, 0.1])
    y_gen += (rv.rvs(n_gen) * [1.4, -.2, 0, -1.8]).sum(1)

    y_gen = np.array(y_gen, dtype='f4')
    y_gen = torch.from_numpy(y_gen)
    y_gen = F.sigmoid(y_gen).numpy()

    return x_gen, y_gen[:, None]
