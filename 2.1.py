#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:30:08 2026

@author: yichentu
"""

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/yichentu/Downloads/test2.csv')
m, n = df.shape
lmbda = 0.97

# Calculate Weights
w = np.array([(1 - lmbda) * (lmbda ** (m - i - 1)) for i in range(m)])
w /= w.sum()

# Weighted Mean and Centering
weighted_mean = w @ df.values
centered_data = df.values - weighted_mean
xm = np.sqrt(w[:, np.newaxis]) * centered_data

# Covariance Matrix
cov_2_1 = pd.DataFrame(xm.T @ xm, index=df.columns, columns=df.columns)
cov_2_1.to_csv('/Users/yichentu/Downloads/testout_2.1.csv')
print(cov_2_1)
