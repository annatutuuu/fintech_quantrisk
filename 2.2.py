#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:49:13 2026

@author: yichentu
"""

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/yichentu/Downloads/test2.csv')
m, n = df.shape

# lmbda = 0.94
lmbda_corr = 0.94
w_corr = np.array([(1 - lmbda_corr) * (lmbda_corr ** (m - i - 1)) for i in range(m)])
w_corr /= w_corr.sum()

xm_corr = np.sqrt(w_corr[:, np.newaxis]) * (df.values - (w_corr @ df.values))
ew_cov = xm_corr.T @ xm_corr

# Convert Covariance to Correlation
std_devs = np.sqrt(np.diag(ew_cov))
corr_2_2 = pd.DataFrame(ew_cov / np.outer(std_devs, std_devs), index=df.columns, columns=df.columns)
corr_2_2.to_csv('/Users/yichentu/Downloads/testout_2.2.csv')