#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 02:25:00 2026

@author: yichentu
Task 6.1: Calculate arithmetic returns and save cleaned CSV
"""

import pandas as pd
import numpy as np

input_path = "/Users/yichentu/Downloads/test6.csv"
output_path = "/Users/yichentu/Downloads/testout_6.1.csv"


df = pd.read_csv(input_path)


date_col = df.columns[0]
dates = df[date_col]
prices = df.iloc[:, 1:]

returns = prices.pct_change()


out = pd.concat([dates, returns], axis=1)
out = out.dropna(how="any")          # drop rows that contain any NaN
out = out.set_index(date_col)        # set Date as index (matches screenshot style)


out.to_csv(output_path, index=True, float_format="%.18f")

print(out)


