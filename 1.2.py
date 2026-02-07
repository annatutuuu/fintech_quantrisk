#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 23:14:27 2026

@author: yichentu
"""

import pandas as pd

# Load the data
df = pd.read_csv('/Users/yichentu/Downloads/test1.csv')

# 1.2 Correlation: Missing data, skip missing rows
df_clean = df.dropna()
corr_matrix = df_clean.corr()

# Save and print results
corr_matrix.to_csv('/Users/yichentu/Downloads/testout_1.2.csv')
# print("1.2 Correlation (Skip Missing Rows):\n", corr_matrix)