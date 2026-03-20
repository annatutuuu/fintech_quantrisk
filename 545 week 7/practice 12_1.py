import numpy as np
import pandas as pd
from scipy.stats import norm

def gbsm(call, S, K, ttm, rf, b, sigma):
    if ttm == 0 or sigma == 0:
        # At maturity or zero vol, price is intrinsic
        if call:
            price = max(S - K, 0)
        else:
            price = max(K - S, 0)
        return price, 0, 0, 0, 0, 0  # Greeks are zero
    
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * ttm) / (sigma * np.sqrt(ttm))
    d2 = d1 - sigma * np.sqrt(ttm)
    
    if call:
        price = S * np.exp((b - rf) * ttm) * norm.cdf(d1) - K * np.exp(-rf * ttm) * norm.cdf(d2)
        delta = np.exp((b - rf) * ttm) * norm.cdf(d1)
        theta = -S * sigma * np.exp((b - rf) * ttm) * norm.pdf(d1) / (2 * np.sqrt(ttm)) - rf * K * np.exp(-rf * ttm) * norm.cdf(d2) + (b - rf) * S * np.exp((b - rf) * ttm) * norm.cdf(d1)
        rho = K * ttm * np.exp(-rf * ttm) * norm.cdf(d2)
    else:
        price = K * np.exp(-rf * ttm) * norm.cdf(-d2) - S * np.exp((b - rf) * ttm) * norm.cdf(-d1)
        delta = np.exp((b - rf) * ttm) * (norm.cdf(d1) - 1)
        theta = -S * sigma * np.exp((b - rf) * ttm) * norm.pdf(d1) / (2 * np.sqrt(ttm)) + rf * K * np.exp(-rf * ttm) * norm.cdf(-d2) + (b - rf) * S * np.exp((b - rf) * ttm) * norm.cdf(-d1)
        rho = -K * ttm * np.exp(-rf * ttm) * norm.cdf(-d2)
    
    gamma = np.exp((b - rf) * ttm) * norm.pdf(d1) / (S * sigma * np.sqrt(ttm))
    vega = S * np.sqrt(ttm) * np.exp((b - rf) * ttm) * norm.pdf(d1)
    
    return price, delta, gamma, theta, vega, rho

# Read the CSV
df = pd.read_csv('test12_1.csv')

# Add new columns for results
df['Price'] = 0.0
df['Delta'] = 0.0
df['Gamma'] = 0.0
df['Theta'] = 0.0
df['Vega'] = 0.0
df['Rho'] = 0.0

# Process each row
for index, row in df.iterrows():
    if pd.isna(row['ID']):
        continue
    call = row['Option Type'] == 'Call'
    S = row['Underlying']
    K = row['Strike']
    days = row['DaysToMaturity']
    day_per_year = row['DayPerYear']
    rf = row['RiskFreeRate']
    dividend_rate = row['DividendRate']
    sigma = row['ImpliedVol']
    
    ttm = days / day_per_year
    b = rf - dividend_rate
    
    price, delta, gamma, theta, vega, rho = gbsm(call, S, K, ttm, rf, b, sigma)
    
    df.at[index, 'Price'] = price
    df.at[index, 'Delta'] = delta
    df.at[index, 'Gamma'] = gamma
    df.at[index, 'Theta'] = theta
    df.at[index, 'Vega'] = vega
    df.at[index, 'Rho'] = rho

# Save to CSV
output_df = df[['ID', 'Price', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']].rename(columns={'Price': 'Value'})
output_df = output_df.dropna()  # Remove rows with NaN
output_df['ID'] = output_df['ID'].astype(int)
output_df.to_csv('testout12_1.csv', index=False)
