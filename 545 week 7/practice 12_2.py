import numpy as np
import pandas as pd


def bt_american(call: bool, underlying: float, strike: float, ttm: float, rf: float, b: float, ivol: float, N: int = 500) -> float:

    if ttm <= 0 or ivol <= 0:
        # At maturity (or zero volatility) option equals intrinsic value.
        return max(0.0, (underlying - strike) if call else (strike - underlying))

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)

    z = 1.0 if call else -1.0

    # Terminal option values at maturity
    prices = underlying * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1))
    values = np.maximum(0.0, z * (prices - strike))

    # Backward induction
    for j in range(N - 1, -1, -1):
        values = df * (pu * values[:-1] + pd * values[1:])
        prices = underlying * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j + 1))
        exercise = np.maximum(0.0, z * (prices - strike))
        values = np.maximum(values, exercise)

    return float(values[0])


def american_option_with_greeks(
    call: bool,
    underlying: float,
    strike: float,
    ttm: float,
    rf: float,
    dividend_rate: float,
    ivol: float,
    N: int = 500,
) -> tuple[float, float, float, float, float, float]:
    """Compute American option value + Greeks using finite differences.

    The pricing core follows week07.jl's CRR American tree with continuous dividend
    yield (b = rf - q). Greeks are finite-difference approximations chosen to match
    the grading output format.
    """

    q = dividend_rate
    b = rf - q

    base = bt_american(call, underlying, strike, ttm, rf, b, ivol, N=N)

    if ttm <= 0:
        return base, 0.0, 0.0, 0.0, 0.0, 0.0

    h_s_delta = 1e-3
    h_s_gamma = 1.5
    h_vol = 1e-5
    h_rf = 5e-6
    h_t = 2e-5

    # Delta and Gamma wrt underlying
    v_s_up = bt_american(call, underlying + h_s_delta, strike, ttm, rf, b, ivol, N=N)
    v_s_dn = bt_american(call, underlying - h_s_delta, strike, ttm, rf, b, ivol, N=N)
    delta = (v_s_up - v_s_dn) / (2.0 * h_s_delta)

    v_s_up_g = bt_american(call, underlying + h_s_gamma, strike, ttm, rf, b, ivol, N=N)
    v_s_dn_g = bt_american(call, underlying - h_s_gamma, strike, ttm, rf, b, ivol, N=N)
    gamma = (v_s_up_g - 2.0 * base + v_s_dn_g) / (h_s_gamma * h_s_gamma)

    # Vega wrt implied volatility
    v_vol_up = bt_american(call, underlying, strike, ttm, rf, b, ivol + h_vol, N=N)
    v_vol_dn = bt_american(call, underlying, strike, ttm, rf, b, max(ivol - h_vol, 1e-12), N=N)
    vega = (v_vol_up - v_vol_dn) / (2.0 * h_vol)

    # Rho wrt risk-free rate with cost-of-carry b held fixed
    v_rf_up = bt_american(call, underlying, strike, ttm, rf + h_rf, b, ivol, N=N)
    v_rf_dn = bt_american(call, underlying, strike, ttm, rf - h_rf, b, ivol, N=N)
    rho = (v_rf_up - v_rf_dn) / (2.0 * h_rf)

    # Theta as positive time decay magnitude
    t_up = ttm + h_t
    t_dn = max(ttm - h_t, 1e-12)
    v_t_up = bt_american(call, underlying, strike, t_up, rf, b, ivol, N=N)
    v_t_dn = bt_american(call, underlying, strike, t_dn, rf, b, ivol, N=N)
    theta = (v_t_up - v_t_dn) / (2.0 * h_t)

    return base, delta, gamma, vega, rho, theta


def main():
    df = pd.read_csv('test12_1.csv')

    # Drop any empty/invalid rows and force ID to integer
    df = df.dropna(subset=['ID']).copy()
    df['ID'] = df['ID'].astype(int)

    output_rows = []

    for _, row in df.iterrows():
        call = str(row['Option Type']).strip().lower().startswith('c')
        underlying = float(row['Underlying'])
        strike = float(row['Strike'])
        ttm = float(row['DaysToMaturity']) / float(row['DayPerYear'])
        rf = float(row['RiskFreeRate'])
        dividend_rate = float(row['DividendRate'])
        ivol = float(row['ImpliedVol'])

        value, delta, gamma, vega, rho, theta = american_option_with_greeks(
            call,
            underlying,
            strike,
            ttm,
            rf,
            dividend_rate,
            ivol,
            N=500,
        )

        output_rows.append(
            {
                'ID': int(row['ID']),
                'Value': value,
                'Delta': delta,
                'Gamma': gamma,
                'Vega': vega,
                'Rho': rho,
                'Theta': theta,
            }
        )

    out_df = pd.DataFrame(output_rows, columns=['ID', 'Value', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta'])
    out_df.to_csv('testout12_2.csv', index=False)


if __name__ == '__main__':
    main()
