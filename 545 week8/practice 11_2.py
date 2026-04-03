import csv
import math
from pathlib import Path


def read_matrix_csv(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        cols = next(reader)
        data = [[float(x) for x in row] for row in reader]
    return cols, data


def read_weights(path: str, n_expected: int):
    vals = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            for cell in row:
                t = cell.strip()
                if not t:
                    continue
                vals.append(float(t))
    if len(vals) != n_expected:
        raise ValueError(f"Weight length mismatch: expected {n_expected}, got {len(vals)}")
    return vals


def read_beta(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        factor_names = header[1:]
        stock_names = []
        beta = []
        for row in reader:
            stock_names.append(row[0])
            beta.append([float(x) for x in row[1:]])
    return stock_names, factor_names, beta


def total_compound_return(series):
    return math.exp(sum(math.log1p(x) for x in series)) - 1.0


def sample_std(x):
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    v = sum((xi - m) * (xi - m) for xi in x) / (n - 1)
    return math.sqrt(max(v, 0.0))


def solve_2x2(a11, a12, a22, b1, b2):
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-18:
        raise ValueError("Singular system in volatility attribution regression.")
    x1 = (b1 * a22 - b2 * a12) / det
    x2 = (a11 * b2 - a12 * b1) / det
    return x1, x2


def main():
    factor_path = Path("test11_2_factor_returns.csv")
    stock_path = Path("test11_2_stock_returns.csv")
    beta_path = Path("test11_2_beta.csv")

    # User prompt has test_11_2_weights.csv; folder currently has test11_2_weights.csv.
    weights_path = Path("test_11_2_weights.csv")
    if not weights_path.exists():
        weights_path = Path("test11_2_weights.csv")

    out_path = Path("testout11_2.csv")

    factor_names, factor_returns = read_matrix_csv(str(factor_path))
    stock_names, stock_returns = read_matrix_csv(str(stock_path))
    beta_stocks, beta_factors, beta = read_beta(str(beta_path))

    if len(factor_returns) != len(stock_returns):
        raise ValueError("Factor and stock return files must have the same number of rows.")
    if factor_names != beta_factors:
        raise ValueError("Factor names in beta file do not match factor return columns.")
    if stock_names != beta_stocks:
        raise ValueError("Stock names in beta file do not match stock return columns.")

    n_days = len(stock_returns)
    n_stocks = len(stock_names)
    n_factors = len(factor_names)

    w0 = read_weights(str(weights_path), n_stocks)

    p_return = [0.0] * n_days
    resid_return = [0.0] * n_days
    factor_weights = [[0.0] * n_factors for _ in range(n_days)]
    last_w = w0[:]

    # Week08 problem2 dynamic weighting and factor attribution setup.
    for t in range(n_days):
        sret_t = stock_returns[t]
        fret_t = factor_returns[t]

        # factorWeights[t, j] = sum_i Beta[i,j] * last_w[i]
        for j in range(n_factors):
            factor_weights[t][j] = sum(beta[i][j] * last_w[i] for i in range(n_stocks))

        updated = [last_w[i] * (1.0 + sret_t[i]) for i in range(n_stocks)]
        gross = sum(updated)
        last_w = [x / gross for x in updated]
        p_return[t] = gross - 1.0

        model_part = sum(factor_weights[t][j] * fret_t[j] for j in range(n_factors))
        resid_return[t] = p_return[t] - model_part

    total_ret = total_compound_return(p_return)
    if abs(total_ret) < 1e-18:
        k = 1.0
    else:
        k = math.log1p(total_ret) / total_ret

    carino_k = []
    for pr in p_return:
        if abs(pr) < 1e-18:
            carino_k.append(1.0 / k)
        else:
            carino_k.append((math.log1p(pr) / pr) / k)

    # Return attribution to factors and alpha.
    attrib_factor = [0.0] * n_factors
    attrib_alpha = 0.0
    for t in range(n_days):
        for j in range(n_factors):
            attrib_factor[j] += factor_returns[t][j] * factor_weights[t][j] * carino_k[t]
        attrib_alpha += resid_return[t] * carino_k[t]

    # Total compounded return by factor/alpha/portfolio.
    factor_total = [total_compound_return([factor_returns[t][j] for t in range(n_days)]) for j in range(n_factors)]
    alpha_total = total_compound_return(resid_return)

    # Vol attribution: regress each component series on portfolio return.
    # Y = [factor_returns .* factor_weights, resid_return]
    y_cols = []
    for j in range(n_factors):
        y_cols.append([factor_returns[t][j] * factor_weights[t][j] for t in range(n_days)])
    y_cols.append(resid_return[:])  # alpha column

    s1 = float(n_days)
    sp = sum(p_return)
    spp = sum(pr * pr for pr in p_return)
    p_std = sample_std(p_return)

    csd = []
    for col in y_cols:
        spy = sum(col)
        sppy = sum(col[t] * p_return[t] for t in range(n_days))
        _, slope = solve_2x2(s1, sp, spp, spy, sppy)
        csd.append(slope * p_std)

    # Assemble output same style as week08_problem2 Attribution.
    out_header = ["Value"] + factor_names + ["Alpha", "Portfolio"]
    row_total = ["TotalReturn"] + factor_total + [alpha_total, total_ret]
    row_ret_attr = ["Return Attribution"] + attrib_factor + [attrib_alpha, total_ret]
    row_vol_attr = ["Vol Attribution"] + csd + [p_std]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerow(row_total)
        writer.writerow(row_ret_attr)
        writer.writerow(row_vol_attr)

    print(",".join(out_header))
    print(",".join(str(x) for x in row_total))
    print(",".join(str(x) for x in row_ret_attr))
    print(",".join(str(x) for x in row_vol_attr))


if __name__ == "__main__":
    main()
