import csv
import math
import random
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
        _ = next(reader, None)
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


def project_to_bounded_simplex(v, lower, upper, tol=1e-14, max_iter=200):
    # Project v onto {w | sum(w)=1, lower <= w_i <= upper}.
    n = len(v)
    if n * lower > 1.0 + 1e-15 or n * upper < 1.0 - 1e-15:
        raise ValueError("Infeasible bounds for simplex.")

    lo = min(v) - upper
    hi = max(v) - lower

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        w = [min(max(vi - mid, lower), upper) for vi in v]
        s = sum(w)
        if abs(s - 1.0) < tol:
            return w
        if s > 1.0:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    w = [min(max(vi - mid, lower), upper) for vi in v]
    # Small normalization to eliminate floating drift while keeping bounds.
    scale = 1.0 / sum(w)
    w = [wi * scale for wi in w]
    return [min(max(wi, lower), upper) for wi in w]


def optimize_sharpe_bounded(mu, cov, rf, lower=0.1, upper=0.5, n_starts=60, max_iter=5000):
    n = len(mu)
    rng = random.Random(545)
    eps = 1e-8

    def objective_and_grad(w, lam, rho, tau):
        # Interior-point + augmented Lagrangian form:
        # min -SR(w) s.t. sum(w)=1 and lower < w < upper
        sr = sharpe_ratio(w, mu, cov, rf)
        gsr = sharpe_gradient(w, mu, cov, rf)
        eq = sum(w) - 1.0

        barrier = 0.0
        for wi in w:
            barrier -= tau * (math.log(wi - lower) + math.log(upper - wi))

        obj = -sr + lam * eq + 0.5 * rho * eq * eq + barrier

        g = [0.0] * n
        eq_term = lam + rho * eq
        for i, wi in enumerate(w):
            barrier_grad = -tau * (1.0 / (wi - lower) - 1.0 / (upper - wi))
            g[i] = -gsr[i] + eq_term + barrier_grad
        return obj, g, eq

    def random_start():
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        return project_to_bounded_simplex(v, lower + eps, upper - eps)

    starts = [[1.0 / n] * n] + [random_start() for _ in range(n_starts - 1)]
    starts = [project_to_bounded_simplex(s, lower + eps, upper - eps) for s in starts]

    best_w = None
    best_sr = -1e100

    def projected_refine(w_start):
        w = project_to_bounded_simplex(w_start, lower, upper)
        cur = sharpe_ratio(w, mu, cov, rf)
        step = 0.2

        for _ in range(3000):
            g = sharpe_gradient(w, mu, cov, rf)
            improved = False
            local_step = step

            for _ in range(35):
                cand = [wi + local_step * gi for wi, gi in zip(w, g)]
                cand = project_to_bounded_simplex(cand, lower, upper)
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

    for w0 in starts:
        w = list(w0)
        lam = 0.0
        rho = 10.0
        tau = 1e-2

        # Outer loop updates AL/barrier parameters, similar spirit to NLP solvers.
        for _ in range(12):
            for _ in range(max_iter):
                obj, g, eq = objective_and_grad(w, lam, rho, tau)
                g2 = dot(g, g)
                if g2 < 1e-24:
                    break

                step = 1.0
                accepted = False
                for _ in range(40):
                    cand = [wi - step * gi for wi, gi in zip(w, g)]
                    if any(ci <= lower + eps or ci >= upper - eps for ci in cand):
                        step *= 0.5
                        continue

                    cand_obj, _, _ = objective_and_grad(cand, lam, rho, tau)
                    if cand_obj <= obj - 1e-4 * step * g2:
                        w = cand
                        accepted = True
                        break
                    step *= 0.5

                if not accepted:
                    break

                if abs(eq) < 1e-12 and math.sqrt(g2) < 1e-8:
                    break

            eq = sum(w) - 1.0
            lam += rho * eq
            rho = min(rho * 3.0, 1e8)
            tau = max(tau * 0.2, 1e-12)

            if abs(eq) < 1e-12 and tau <= 1e-10:
                break

        # Final feasibility projection + local ascent refinement on the feasible set.
        w = projected_refine(w)
        cur = sharpe_ratio(w, mu, cov, rf)

        if cur > best_sr:
            best_sr = cur
            best_w = w

    return best_w


def main():
    cov_path = Path("test5_2.csv")
    means_path = Path("test10_3_means.csv")
    output_path = Path("testout10_4.csv")
    rf = 0.04

    names, cov = read_covariance_matrix(str(cov_path))
    mu = read_means(str(means_path), len(cov))

    w = optimize_sharpe_bounded(mu, cov, rf, lower=0.1, upper=0.5)

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
