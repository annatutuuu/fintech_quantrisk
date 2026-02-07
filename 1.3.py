#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:18:39 2026

@author: yichentu
"""

import pandas as pd

# Load the data
df = pd.read_csv('/Users/yichentu/Downloads/test1.csv')

# 1.3 Covariance: Missing data, Pairwise
# pandas .cov() uses pairwise deletion by default
cov_matrix = df.cov()

# Save and print results
cov_matrix.to_csv('/Users/yichentu/Downloads/testout_1.3.csv')
# print("1.3 Covariance (Pairwise):\n", cov_matrix)