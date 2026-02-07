#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 00:32:25 2026

@author: yichentu
"""
import pandas as pd
import numpy as np


df = pd.read_csv("/Users/yichentu/Downloads/testout_1.4.csv")
mat = df.values


vals, vecs = np.linalg.eigh(mat)


vals = np.maximum(vals, 1e-8) 


T = 1.0 / (np.square(vecs) @ vals)
T_diag = np.diag(np.sqrt(T))
L_diag = np.diag(np.sqrt(vals))


B = T_diag @ vecs @ L_diag
psd_corr = B @ B.T


pd.DataFrame(psd_corr, columns=df.columns).to_csv('/Users/yichentu/Downloads/testout_3.2.csv', index=False)

