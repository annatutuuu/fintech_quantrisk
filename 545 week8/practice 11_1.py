import csv
import math
from pathlib import Path


def read_returns(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        names = next(reader)
        rows = [[float(x) for x in row] for row in reader]

    if not rows:
        raise ValueError("Returns file is empty.")
    n_assets = len(names)
    if any(len(r) != n_assets for r in rows):
        raise ValueError("Returns file has inconsistent row lengths.")
    return names, rows


def read_weights(path: str, n_expected: int):
    values = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # optional header
        for row in reader:
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                try:
                    values.append(float(text))
                except ValueError:
                    continue

    if len(values) != n_expected:
        raise ValueError(f"Weight length mismatch: expected {n_expected}, got {len(values)}.")
    return values


def sample_std(x):
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    var = sum((xi - m) * (xi - m) for xi in x) / (n - 1)
    return math.sqrt(max(var, 0.0))


def total_compound_return(series):
    return math.exp(sum(math.log1p(r) for r in series)) - 1.0


def solve_linear_system(a, b):
    # Gaussian elimination with partial pivoting for small dense systems.
    n = len(a)
    m = [row[:] + [bi] for row, bi in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-18:
            raise ValueError("Singular system in volatility attribution.")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        piv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= piv

        for i in range(n):
            if i == col:
                continue
            fac = m[i][col]
            if fac == 0.0:
                continue
            for j in range(col, n + 1):
                m[i][j] -= fac * m[col][j]

    return [m[i][n] for i in range(n)]


def main():
    returns_path = Path("test11_1_returns.csv")
    weights_path = Path("test11_1_weights.csv")
    out_path = Path("testout11_1.csv")

    names, returns_rows = read_returns(str(returns_path))
    n_days = len(returns_rows)
    n_assets = len(names)

    w0 = read_weights(str(weights_path), n_assets)

    # Week08-style ex-post return stream from dynamic weights.
    p_return = [0.0] * n_days
    weights_hist = [[0.0] * n_assets for _ in range(n_days)]
    last_w = w0[:]

    for t in range(n_days):
        r_t = returns_rows[t]
        weights_hist[t] = last_w[:]

        updated = [wi * (1.0 + ri) for wi, ri in zip(last_w, r_t)]
        p_r_gross = sum(updated)
        last_w = [x / p_r_gross for x in updated]
        p_return[t] = p_r_gross - 1.0

    total_ret = total_compound_return(p_return)
    if abs(total_ret) < 1e-18:
        carino_k = 1.0
    else:
        carino_k = math.log1p(total_ret) / total_ret

    carino_kt = []
    for pr in p_return:
        if abs(pr) < 1e-18:
            carino_kt.append(1.0 / carino_k)
        else:
            carino_kt.append((math.log1p(pr) / pr) / carino_k)

    # Return attribution by asset
    ret_attr = [0.0] * n_assets
    for t in range(n_days):
        for i in range(n_assets):
            ret_attr[i] += returns_rows[t][i] * weights_hist[t][i] * carino_kt[t]

    # Total return for each asset (standalone series)
    asset_total = []
    for i in range(n_assets):
        s = [returns_rows[t][i] for t in range(n_days)]
        asset_total.append(total_compound_return(s))

    # Volatility attribution (same beta-decomposition used in week08)
    y = [[returns_rows[t][i] * weights_hist[t][i] for i in range(n_assets)] for t in range(n_days)]
    p_std = sample_std(p_return)

    # Build X'X once for X=[1, p_return]
    s1 = float(n_days)
    sp = sum(p_return)
    spp = sum(pr * pr for pr in p_return)
    xtx = [[s1, sp], [sp, spp]]

    csd = [0.0] * n_assets
    for i in range(n_assets):
        spy = sum(y[t][i] for t in range(n_days))
        sppy = sum(p_return[t] * y[t][i] for t in range(n_days))
        xty = [spy, sppy]
        beta = solve_linear_system(xtx, xty)[1]
        csd[i] = beta * p_std

    # Assemble output table like week08 attribution.
    header = ["Value"] + names + ["Portfolio"]
    row_total = ["TotalReturn"] + asset_total + [total_ret]
    row_ret_attr = ["Return Attribution"] + ret_attr + [total_ret]
    row_vol_attr = ["Vol Attribution"] + csd + [p_std]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row_total)
        writer.writerow(row_ret_attr)
        writer.writerow(row_vol_attr)

    # Console preview
    print(",".join(header))
    for row in [row_total, row_ret_attr, row_vol_attr]:
        print(",".join(str(x) for x in row))


if __name__ == "__main__":
    main()
