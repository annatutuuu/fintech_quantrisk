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


def read_means(path: str, n_expected: int):
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
        raise ValueError(
            f"Mean vector size mismatch: expected {n_expected}, got {len(values)}."
        )
    return values


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def mat_vec_mul(a, x):
    return [sum(aij * xj for aij, xj in zip(ai, x)) for ai in a]


def sharpe_ratio(w, mu, cov, rf):
    excess = dot(w, mu) - rf
    sigma_w = mat_vec_mul(cov, w)
    var_p = dot(w, sigma_w)
    vol_p = math.sqrt(max(var_p, 1e-30))
    return excess / vol_p


def sharpe_gradient(w, mu, cov, rf):
    # grad of SR(w) = (a'w) / sqrt(w'Σw), where a = mu - rf*1
    n = len(w)
    a = [mui - rf for mui in mu]
    u = dot(a, w)
    sigma_w = mat_vec_mul(cov, w)
    v2 = dot(w, sigma_w)
    v = math.sqrt(max(v2, 1e-30))
    v3 = max(v * v * v, 1e-30)

    g = [0.0] * n
    for i in range(n):
        g[i] = a[i] / v - (u / v3) * sigma_w[i]
    return g


def project_to_simplex_with_floor(x, floor):
    # Project x onto {w | sum(w)=1, w_i>=floor}
    n = len(x)
    if floor * n >= 1.0:
        raise ValueError("Floor too large for simplex projection.")

    target = 1.0 - n * floor
    y = [xi - floor for xi in x]

    # Standard simplex projection for y >= 0, sum(y)=target
    u = sorted(y, reverse=True)
    cumsum = 0.0
    rho = -1
    theta = 0.0
    for i, ui in enumerate(u):
        cumsum += ui
        t = (cumsum - target) / (i + 1)
        if ui - t > 0:
            rho = i
            theta = t
    if rho == -1:
        z = [target / n] * n
    else:
        theta = (sum(u[: rho + 1]) - target) / (rho + 1)
        z = [max(yi - theta, 0.0) for yi in y]

    return [zi + floor for zi in z]


def optimize_sharpe_long_only(mu, cov, rf, max_outer=12, max_inner=5000):
    # Week08-style setup:
    # maximize SR(w) with constraints sum(w)=1 and w>=0, starting from 1/n.
    n = len(mu)
    w = [1.0 / n] * n

    # Augmented-Lagrangian + log-barrier (interior-point style) for constraints.
    lam = 0.0
    rho = 10.0
    tau = 1e-2
    eps = 1e-12

    def obj_and_grad(x):
        sr = sharpe_ratio(x, mu, cov, rf)
        gsr = sharpe_gradient(x, mu, cov, rf)
        eq = sum(x) - 1.0

        barrier = -tau * sum(math.log(max(xi, eps)) for xi in x)
        obj = -sr + lam * eq + 0.5 * rho * eq * eq + barrier

        eq_term = lam + rho * eq
        grad = [0.0] * n
        for i, xi in enumerate(x):
            grad[i] = -gsr[i] + eq_term - tau / max(xi, eps)
        return obj, grad, eq

    for _ in range(max_outer):
        for _ in range(max_inner):
            obj, grad, eq = obj_and_grad(w)
            grad_norm2 = dot(grad, grad)
            if grad_norm2 < 1e-22 and abs(eq) < 1e-12:
                break

            step = 1.0
            accepted = False
            for _ in range(40):
                cand = [wi - step * gi for wi, gi in zip(w, grad)]
                if min(cand) <= 0.0:
                    step *= 0.5
                    continue

                cand_obj, _, _ = obj_and_grad(cand)
                if cand_obj <= obj - 1e-4 * step * grad_norm2:
                    w = cand
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                break

        eq = sum(w) - 1.0
        lam += rho * eq
        rho = min(rho * 3.0, 1e8)
        tau = max(tau * 0.2, 1e-12)

        if abs(eq) < 1e-12 and tau <= 1e-10:
            break

    # Final feasible refinement on the constrained set.
    floor = -1e-8
    w = project_to_simplex_with_floor(w, floor)
    cur = sharpe_ratio(w, mu, cov, rf)
    step = 0.2

    for _ in range(6000):
        g = sharpe_gradient(w, mu, cov, rf)
        improved = False
        local_step = step

        for _ in range(35):
            cand = [wi + local_step * gi for wi, gi in zip(w, g)]
            cand = project_to_simplex_with_floor(cand, floor)
            sr_cand = sharpe_ratio(cand, mu, cov, rf)
            if sr_cand > cur:
                w = cand
                cur = sr_cand
                step = min(local_step * 1.2, 1.0)
                improved = True
                break
            local_step *= 0.5

        if not improved:
            break

    return w


def main():
    cov_path = Path("test5_2.csv")
    means_path = Path("test10_3_means.csv")
    output_path = Path("testout10_3.csv")
    rf = 0.04

    names, cov = read_covariance_matrix(str(cov_path))
    mu = read_means(str(means_path), len(cov))

    w = optimize_sharpe_long_only(mu, cov, rf)

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
