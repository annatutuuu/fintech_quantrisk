#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 18:49:14 2026

@author: yichentu
"""
#7.1_normaldist

import pandas as pd
import numpy as np

def fit_normal(x):
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    mu = x.mean()
    sigma = np.sqrt(((x - mu) ** 2).sum() / (n - 1))  
    return mu, sigma

cin = pd.read_csv("/Users/yichentu/Downloads/test7_1.csv")
x = cin.iloc[:, 0].values

mu_hat, sigma_hat = fit_normal(x)

out = pd.DataFrame({"mu": [mu_hat], "sigma": [sigma_hat]})
#print(out)

out.to_csv("/Users/yichentu/Desktop/duke fintech/testout7_1.csv", index=False)

