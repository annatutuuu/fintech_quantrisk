#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 00:32:42 2026

@author: yichentu
"""

import pandas as pd
import numpy as np

def higham_psd_corr(a, iterations=100):
    delta_s = np.zeros_like(a)
    y = a.copy()
    for _ in range(iterations):
        r = y - delta_s
        vals, vecs = np.linalg.eigh(r)
        vals = np.maximum(vals, 0)
        x = vecs @ np.diag(vals) @ vecs.T
        delta_s = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0) # Ensure correlation diagonal is 1
    return y

df = pd.read_csv("/Users/yichentu/Downloads/testout_1.4.csv")
out = higham_psd_corr(df.values)
# print(out)
pd.DataFrame(out, columns=df.columns).to_csv("/Users/yichentu/Downloads/testout_3.4.csv", index=False)