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


def risk_parity_weights(cov, budget, tol=1e-12, max_iter=20000):
    n = len(cov)
    if len(budget) != n:
        raise ValueError("Risk budget length must match covariance matrix size.")
    if any(b <= 0.0 for b in budget):
        raise ValueError("Risk budget weights must be positive.")

    # Positive initialization
    x = [1.0] * n

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            c_i = sum(cov[i][j] * x[j] for j in range(n) if j != i)
            a_ii = cov[i][i]
            disc = c_i * c_i + 4.0 * a_ii * budget[i]
            x[i] = (-c_i + math.sqrt(disc)) / (2.0 * a_ii)

        if max(abs(xi - xoi) for xi, xoi in zip(x, x_old)) < tol:
            break

    total_x = sum(x)
    return [xi / total_x for xi in x]


def main():
    input_path = Path("test5_2.csv")
    output_path = Path("testout10_2.csv")
    names, cov = read_covariance_matrix(str(input_path))

    # "1/2 risk weight on X5" means X5 has half the risk-weight of each other asset.
    # Relative risk weights (X1..X5): [1, 1, 1, 1, 0.5]
    budget = [1.0, 1.0, 1.0, 1.0, 0.5]

    w = risk_parity_weights(cov, budget)

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
