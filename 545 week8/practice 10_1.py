import csv
import math
from pathlib import Path


def read_covariance_matrix(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [[float(x) for x in row] for row in reader]

    n = len(data)
    if n == 0 or any(len(row) != n for row in data):
        raise ValueError("Input must be a non-empty square covariance matrix.")

    return header, data


def mat_vec_mul(a, x):
    return [sum(aij * xj for aij, xj in zip(ai, x)) for ai in a]


def risk_parity_weights(cov, tol=1e-12, max_iter=20000):
    n = len(cov)
    b = [1.0 / n] * n  # equal risk budget

    # Positive initialization
    x = [1.0] * n

    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            c_i = sum(cov[i][j] * x[j] for j in range(n) if j != i)
            a_ii = cov[i][i]
            disc = c_i * c_i + 4.0 * a_ii * b[i]
            x[i] = (-c_i + math.sqrt(disc)) / (2.0 * a_ii)

        max_diff = max(abs(xi - xoi) for xi, xoi in zip(x, x_old))
        if max_diff < tol:
            break

    total_x = sum(x)
    w = [xi / total_x for xi in x]
    return w


def portfolio_stats(cov, w):
    sigma_w = mat_vec_mul(cov, w)
    var_p = sum(wi * swi for wi, swi in zip(w, sigma_w))
    vol_p = math.sqrt(var_p)

    # Component risk contribution to volatility: RC_i = w_i * (Sigma w)_i / sigma_p
    rc = [wi * swi / vol_p for wi, swi in zip(w, sigma_w)]
    rc_pct = [rci / vol_p for rci in rc]

    return vol_p, rc, rc_pct


def main():
    input_path = Path("test5_2.csv")
    output_path = Path("testout10_1.csv")
    names, cov = read_covariance_matrix(str(input_path))

    w = risk_parity_weights(cov)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["W"])
        for wi in w:
            writer.writerow([repr(wi)])

    print("W")
    for wi in w:
        print(repr(wi))


if __name__ == "__main__":
    main()
