#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 02:32:13 2026

@author: yichentu
"""
# calculate_log_returns.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 02:27:00 2026

@author: yichentu
Task 6.2: Calculate log returns and save cleaned CSV
"""

import pandas as pd
import numpy as np


input_path = "/Users/yichentu/Downloads/test6.csv"
output_path = "/Users/yichentu/Downloads/testout_6.2.csv"


df = pd.read_csv(input_path)


date_col = df.columns[0]
dates = df[date_col]
prices = df.iloc[:, 1:]



prices_lag = prices.shift(1)

with np.errstate(divide="ignore", invalid="ignore"):
    log_returns = np.log(prices / prices_lag)


log_returns[(prices <= 0) | (prices_lag <= 0)] = np.nan


out = pd.concat([dates, log_returns], axis=1)
out = out.dropna(how="any")
out = out.set_index(date_col)


out.to_csv(output_path, index=True, float_format="%.18f")

print(out)

