#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:50:01 2026

@author: yichentu
"""

import pandas as pd
import numpy as np


df = pd.read_csv('/Users/yichentu/Downloads/test2.csv')

def get_ew_stats(data, lmbda):
    m, n = data.shape
    # 1. Weights: (1-l)*l^(m-i)
    w = np.array([(1 - lmbda) * (lmbda ** (m - i - 1)) for i in range(m)])
    w /= w.sum()
    
    # 2. Weighted Covariance logic
    weighted_mean = w @ data.values
    centered = data.values - weighted_mean
    xm = np.sqrt(w[:, np.newaxis]) * centered
    cov = xm.T @ xm
    
    # 3. Derive Std Devs and Correlation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return std, corr

# Step 1: Get Volatility (Std Dev) from lambda = 0.97
std_97, _ = get_ew_stats(df, 0.97)

# Step 2: Get Correlation from lambda = 0.94
_, corr_94 = get_ew_stats(df, 0.94)

# Step 3: Combine them to form the final Covariance matrix
# Cov = diag(std_97) * corr_94 * diag(std_97)
final_cov_matrix = np.diag(std_97) @ corr_94 @ np.diag(std_97)

# Convert to DataFrame and save
out_2_3 = pd.DataFrame(final_cov_matrix, index=df.columns, columns=df.columns)
out_2_3.to_csv('/Users/yichentu/Downloads/testout_2.3.csv')

print("2.3 Hybrid Covariance Matrix (Vol 0.97, Corr 0.94):")
print(out_2_3)