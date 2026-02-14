import csv
import math
import random
from pathlib import Path
from statistics import NormalDist


def resolve_input_file(name: str) -> Path:
    local = Path(name)
    if local.exists():
        return local
    parent = Path("..") / name
    if parent.exists():
        return parent
    raise FileNotFoundError(f"Cannot find input file: {name}")


def read_single_column_csv(path: Path) -> list[float]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader) 
        return [float(row[0]) for row in reader if row]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sample_std(xs: list[float]) -> float:
    m = mean(xs)
    n = len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def quantile_r7(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 1:
        return ys[0]
    h = 1.0 + (n - 1) * p
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return ys[lo - 1]
    w = h - lo
    return ys[lo - 1] * (1.0 - w) + ys[hi - 1] * w


def var_from_samples(xs: list[float], alpha: float = 0.05) -> float:
    return -quantile_r7(xs, alpha)


def t_pdf(x: float, nu: float) -> float:
    c = math.exp(math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0))
    c /= math.sqrt(nu * math.pi)
    return c * (1.0 + (x * x) / nu) ** (-(nu + 1.0) / 2.0)


def t_cdf(x: float, nu: float) -> float:
    if x == 0.0:
        return 0.5
    sign = 1.0 if x > 0 else -1.0
    a = 0.0
    b = abs(x)
    n = 400  # even
    h = (b - a) / n
    s = t_pdf(a, nu) + t_pdf(b, nu)
    for i in range(1, n):
        xi = a + i * h
        s += (4.0 if i % 2 == 1 else 2.0) * t_pdf(xi, nu)
    integral = s * h / 3.0
    return 0.5 + sign * integral


def t_ppf(p: float, nu: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        c = t_cdf(mid, nu)
        if c < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_general_t_moments(xs: list[float]) -> tuple[float, float, float]:
    """
    Returns (mu, sigma, nu), where:
    X = mu + sigma * T_nu
    """
    mu = mean(xs)
    n = len(xs)
    s2 = sum((x - mu) ** 2 for x in xs) / (n - 1)
    if s2 <= 0:
        return mu, 0.0, 30.0

    m4 = sum((x - mu) ** 4 for x in xs) / n
    kurt_excess = m4 / (s2 * s2) - 3.0

    # Method-of-moments fallback if tails are near normal.
    if kurt_excess <= 0.02:
        nu = 200.0
    else:
        nu = 6.0 / kurt_excess + 4.0
        nu = min(max(nu, 4.1), 200.0)

    sigma = math.sqrt(s2 * (nu - 2.0) / nu)
    return mu, sigma, nu


def simulate_t(mu: float, sigma: float, nu: float, n: int, seed: int = 4) -> list[float]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        z = rng.gauss(0.0, 1.0)
        v = rng.gammavariate(nu / 2.0, 2.0)  
        t = z / math.sqrt(v / nu)
        out.append(mu + sigma * t)
    return out


def write_var_output(path: str, var_abs: float, var_diff_mean: float) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VaR Absolute", "VaR Diff from Mean"])
        writer.writerow([var_abs, var_diff_mean])


def task_8_1() -> None:
    xs = read_single_column_csv(resolve_input_file("test7_1.csv"))
    mu = mean(xs)
    sigma = sample_std(xs)

    q05_abs = NormalDist(mu, sigma).inv_cdf(0.05)
    var_abs = -q05_abs
    var_diff_mean = -NormalDist(0.0, sigma).inv_cdf(0.05)
    write_var_output("testout_8.1.csv", var_abs, var_diff_mean)


def task_8_2() -> tuple[float, float, float]:
    xs = read_single_column_csv(resolve_input_file("test7_2.csv"))
    mu, sigma, nu = fit_general_t_moments(xs)

    q05_t = t_ppf(0.05, nu)
    var_abs = -(mu + sigma * q05_t)
    var_diff_mean = -(sigma * q05_t)
    write_var_output("testout_8.2.csv", var_abs, var_diff_mean)
    return mu, sigma, nu


def task_8_3(mu: float, sigma: float, nu: float) -> None:
    sim = simulate_t(mu, sigma, nu, n=10000, seed=4)
    var_abs = var_from_samples(sim, alpha=0.05)
    sim_mean = mean(sim)
    sim_centered = [x - sim_mean for x in sim]
    var_diff_mean = var_from_samples(sim_centered, alpha=0.05)
    write_var_output("testout_8.3.csv", var_abs, var_diff_mean)


def main() -> None:
    # 8.1 VaR from Normal fit
    task_8_1()
    # 8.2 VaR from T fit
    mu, sigma, nu = task_8_2()
    # 8.3 VaR from simulation using the fitted T model
    task_8_3(mu, sigma, nu)
    print("Generated testout_8.1.csv, testout_8.2.csv, testout_8.3.csv")


if __name__ == "__main__":
    main()
