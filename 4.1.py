#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 02:17:19 2026

@author: yichentu
"""

import pandas as pd
import numpy as np

def chol_psd(a):

    n = a.shape[0]
    root = np.zeros((n, n))
    
    for j in range(n):
        s = 0.0
       
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        
    
        if 0 >= temp >= -1e-8:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp)
        

        if root[j, j] > 0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
                
    return root


df = pd.read_csv("/Users/yichentu/Downloads/testout_3.1.csv")
a_matrix = df.values

root_result = chol_psd(a_matrix)


out_df = pd.DataFrame(root_result, columns=df.columns)
out_df.to_csv("/Users/yichentu/Downloads/testout_4.1.csv", index=False)

print(out_df)