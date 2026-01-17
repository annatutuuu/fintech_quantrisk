#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7.3 Fit T Regression (Student-t errors) via MLE
@author: yichentu
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def fit_regression_t(y, X):
    """
    MLE for: y = Alpha + X*B + e,   e ~ Student-t(df=nu, loc=0, scale=sigma)
    Return: mu(=0), sigma, nu, beta=[Alpha, B1, B2, ...]
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)

    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # add intercept
    X = np.column_stack([np.ones(n), X])
    p = X.shape[1]

    # init with OLS
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid0 = y - X @ beta0
    sigma0 = float(np.sqrt(np.mean(resid0**2)))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 1.0

    nu0 = 5.0  # initial df

    # params
    theta0 = np.concatenate([beta0, [np.log(sigma0), np.log(nu0)]])

    def nll(theta):
        beta = theta[:p]
        sigma = np.exp(theta[p])
        nu = np.exp(theta[p + 1]) 

        e = y - X @ beta
        z = e / sigma

        # log pdf of standardized t
        ll_std = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * (np.log(nu) + np.log(np.pi))
            - ((nu + 1.0) / 2.0) * np.log1p((z**2) / nu)
        )

        ll = np.sum(ll_std - np.log(sigma))  
        return -ll

    opt = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        options={
            "maxiter": 8000,
            "gtol": 1e-10,
            "ftol": 1e-12
        }
    )

    theta = opt.x
    beta_hat = theta[:p]
    sigma_hat = float(np.exp(theta[p]))
    nu_hat = float(np.exp(theta[p + 1]))
    mu_hat = 0.0  

    return mu_hat, sigma_hat, nu_hat, beta_hat, opt


cin = pd.read_csv("/Users/yichentu/Downloads/test7_3.csv")

y = cin["y"].values
X = cin.drop(columns=["y"]).values  # x1 x2 x3

mu_hat, sigma_hat, nu_hat, beta_hat, opt = fit_regression_t(y, X)

out = pd.DataFrame({
    "mu": [mu_hat],
    "sigma": [sigma_hat],
    "nu": [nu_hat],
    "Alpha": [beta_hat[0]],
    "B1": [beta_hat[1]],
    "B2": [beta_hat[2]],
    "B3": [beta_hat[3]],
})

# High precision display (does NOT affect values)
pd.set_option("display.float_format", lambda v: f"{v:.16f}")
#print(out)

out.to_csv("/Users/yichentu/Desktop/duke fintech/testout7_3.csv", index=False)
