import numpy as np
import pandas as pd


def bt_american(call: bool, underlying: float, strike: float, ttm: float, rf: float, b: float, ivol: float, N: int) -> float:
    """Standard CRR American option tree with cost-of-carry b."""
    if ttm <= 0 or ivol <= 0 or N <= 0:
        return max(0.0, (underlying - strike) if call else (strike - underlying))

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1.0 if call else -1.0

    prices = underlying * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1))
    values = np.maximum(0.0, z * (prices - strike))

    for j in range(N - 1, -1, -1):
        values = df * (pu * values[:-1] + pd * values[1:])
        prices = underlying * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j + 1))
        exercise = np.maximum(0.0, z * (prices - strike))
        values = np.maximum(values, exercise)

    return float(values[0])


def bt_american_discrete_div(
    call: bool,
    underlying: float,
    strike: float,
    ttm: float,
    rf: float,
    div_amts: list[float],
    div_times: list[int],
    ivol: float,
    N: int,
) -> float:
    """American option tree with discrete dividends (week07.jl recursive structure)."""
    if ttm <= 0 or N <= 0:
        return max(0.0, (underlying - strike) if call else (strike - underlying))

    if not div_amts or not div_times or div_times[0] > N:
        # No in-grid discrete dividend left: reduce to standard American with b = rf.
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1.0 if call else -1.0

    first_div_time = div_times[0]
    first_div_amt = div_amts[0]

    values = np.zeros(first_div_time + 1, dtype=float)

    for j in range(first_div_time, -1, -1):
        next_values = np.zeros(j + 1, dtype=float)
        for i in range(j, -1, -1):
            price = underlying * (u ** i) * (d ** (j - i))
            exercise = max(0.0, z * (price - strike))

            if j == first_div_time:
                ex_div_price = max(price - first_div_amt, 1e-12)
                rem_amts = div_amts[1:]
                rem_times = [t - first_div_time for t in div_times[1:]]
                hold = bt_american_discrete_div(
                    call,
                    ex_div_price,
                    strike,
                    ttm - first_div_time * dt,
                    rf,
                    rem_amts,
                    rem_times,
                    ivol,
                    N - first_div_time,
                )
                next_values[i] = max(exercise, hold)
            else:
                hold = df * (pu * values[i + 1] + pd * values[i])
                next_values[i] = max(exercise, hold)
        values = next_values

    return float(values[0])


def parse_number_list(text: str) -> list[float]:
    if pd.isna(text):
        return []
    items = [x.strip() for x in str(text).split(",") if x.strip() != ""]
    return [float(x) for x in items]


def build_div_schedule(row: pd.Series, N: int) -> tuple[list[float], list[int]]:
    days_to_maturity = float(row["DaysToMaturity"])
    underlying = float(row["Underlying"])

    raw_dates = parse_number_list(row["DividendDates"])
    raw_amts = parse_number_list(row["DividendAmts"])

    n = min(len(raw_dates), len(raw_amts))
    dates = raw_dates[:n]
    amts = raw_amts[:n]

    pairs: list[tuple[int, float]] = []
    for d, a in zip(dates, amts):
        if d <= 0 or d > days_to_maturity:
            continue

        # In week07.jl, discrete dividends are cash amounts subtracted from price.
        cash_amt = a
        grid_t = int(round((d / days_to_maturity) * N))
        if 0 < grid_t <= N:
            pairs.append((grid_t, cash_amt))

    pairs.sort(key=lambda x: x[0])
    div_times = [p[0] for p in pairs]
    div_amts = [p[1] for p in pairs]
    return div_amts, div_times


def main():
    df = pd.read_csv("test12_3.csv")
    df = df.dropna(subset=["ID"]).copy()
    df["ID"] = df["ID"].astype(int)

    N = 500
    rows = []

    for _, row in df.iterrows():
        call = str(row["Option Type"]).strip().lower().startswith("c")
        underlying = float(row["Underlying"])
        strike = float(row["Strike"])
        ttm = float(row["DaysToMaturity"]) / float(row["DayPerYear"])
        rf = float(row["RiskFreeRate"])
        ivol = float(row["ImpliedVol"])

        div_amts, div_times = build_div_schedule(row, N)
        value = bt_american_discrete_div(call, underlying, strike, ttm, rf, div_amts, div_times, ivol, N)

        rows.append({"ID": int(row["ID"]), "Value": value})

    out_df = pd.DataFrame(rows, columns=["ID", "Value"])
    out_df.to_csv("testout12_3.csv", index=False)


if __name__ == "__main__":
    main()
