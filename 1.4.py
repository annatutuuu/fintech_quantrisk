#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:21:24 2026

@author: yichentu
"""

import pandas as pd

df = pd.read_csv('/Users/yichentu/Downloads/test1.csv')

# 1.4 Correlation: Missing data, Pairwise
# pandas .corr() uses pairwise deletion by default
corr_matrix = df.corr()

# Save and print results
corr_matrix.to_csv('/Users/yichentu/Downloads/testout_1.4.csv')
# print("1.4 Correlation (Pairwise):\n", corr_matrix)