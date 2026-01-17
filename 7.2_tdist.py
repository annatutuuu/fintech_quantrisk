#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:43:04 2026

@author: yichentu
"""
#7.2_tdist

import pandas as pd
import numpy as np
from scipy.stats import t


cin = pd.read_csv("/Users/yichentu/Downloads/test7_2.csv")
x = cin.iloc[:, 0].values


nu_hat, mu_hat, sigma_hat = t.fit(x)


out = pd.DataFrame({
    "mu": [mu_hat],
    "sigma": [sigma_hat],
    "nu": [nu_hat]
})

# print(out)
out.to_csv("/Users/yichentu/Desktop/duke fintech/testout7_2.csv", index=False)
