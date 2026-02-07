#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 00:14:00 2026

@author: yichentu
"""

import pandas as pd
import numpy as np

def near_psd_covariance(mat, epsilon=0.0):
    n = mat.shape[0]
    out = mat.copy()


    is_covariance = not np.allclose(np.diag(out), 1.0)
    
    invSD = None
    if is_covariance:
        std_devs = np.sqrt(np.diag(out))
        invSD = np.diag(1.0 / std_devs)
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    
    T = 1.0 / (np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    if invSD is not None:
        SD = np.diag(std_devs)
        out = SD @ out @ SD
        
    return out


df = pd.read_csv("/Users/yichentu/Downloads/testout_1.3.csv")
mat = df.values

psd_mat = near_psd_covariance(mat)


df_out = pd.DataFrame(psd_mat, columns=df.columns)
df_out.to_csv('/Users/yichentu/Downloads/testout_3.1.csv', index=False)

print(df_out)