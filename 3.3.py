#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 00:32:33 2026

@author: yichentu
"""

import pandas as pd
import numpy as np

def higham_psd_cov(a, iterations=100, epsilon=1e-9):
    std_devs = np.sqrt(np.diag(a))

    std_devs = np.maximum(std_devs, epsilon)
    D = np.diag(std_devs)
    D_inv = np.diag(1.0 / std_devs)
    
    corr = D_inv @ a @ D_inv
    corr = (corr + corr.T) / 2
    
    delta_s = np.zeros_like(corr)
    y = corr.copy()
    
    for _ in range(iterations):
        r = y - delta_s
        vals, vecs = np.linalg.eigh(r)
        
        vals = np.maximum(vals, epsilon)
        
        x = vecs @ np.diag(vals) @ vecs.T
        delta_s = x - r
        y = x.copy()
        
        np.fill_diagonal(y, 1.0)
    
    return D @ y @ D

df = pd.read_csv("/Users/yichentu/Downloads/testout_1.3.csv")

psd_mat = higham_psd_cov(df.values)

result_df = pd.DataFrame(psd_mat, columns=df.columns)
# print(result_df)

result_df.to_csv("/Users/yichentu/Downloads/testout_3.3.csv", index=False)
