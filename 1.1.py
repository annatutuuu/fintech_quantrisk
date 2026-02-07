#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:08:37 2026

@author: yichentu
"""

import pandas as pd

# Load the data
df = pd.read_csv('/Users/yichentu/Downloads/test1.csv')

# 1.1 Covariance: Missing data, skip missing rows (Listwise deletion)
# .dropna() removes any row with at least one NaN
df_clean = df.dropna()
cov_matrix = df_clean.cov()

# Save and print results
cov_matrix.to_csv('/Users/yichentu/Downloads/testout_1.1.csv')
# print("1.1 Covariance (Skip Missing Rows):\n", cov_matrix)