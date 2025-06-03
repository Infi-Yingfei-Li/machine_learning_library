#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, patsy
import numpy as np
import pandas as pd

import scipy

import statsmodels
import statsmodels.api as sm

class Bspline:
    def __init__(self, knots, degree=3):
        self.knots = knots
        self.degree = degree
        self.df = len(knots) - degree - 1

    def basis(self, x):
        '''
        Evaluate the B-spline basis functions at points x.
        params:
            x: np.ndarray, shape (n_samples, 1), points at which to evaluate the basis functions
        return:
            ans: np.ndarray, shape (n_samples, self.df), evaluated basis functions
        '''
        x = x.flatten()
        ans = np.zeros((len(x), self.df))
        for i in range(self.df):
            c = np.zeros(self.df); c[i] = 1
            spl = scipy.interpolate.BSpline(self.knots, c, self.degree)
            ans[:, i] = spl(x)
        return ans
    
    def smoother(self, x, y, npts=100):
        '''
        Fit a B-spline smoother to the data (x, y).
        params:
            x: np.ndarray, shape (n_samples, 1), input data
            y: np.ndarray, shape (n_samples, 1), target data
        '''
        x_basis = self.basis(x)
        y = y.flatten()
        ols = sm.OLS(y, sm.add_constant(x_basis, prepend=True)).fit()
        fit_x = np.linspace(np.min(x), np.max(x), npts).reshape(-1, 1)
        fit_x_basis = self.basis(fit_x)
        fit_y = ols.predict(sm.add_constant(fit_x_basis, prepend=True)).reshape(-1, 1)
        return {"fit_x": fit_x, "fit_y": fit_y, "R2": ols.rsquared}
