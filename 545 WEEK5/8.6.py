import csv
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import t

ALPHA = 0.05
N_SIM = 10000
SEED = 545


def read_returns(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # assume first row is header
    return [float(r[0]) for r in rows[1:] if r]


def es_empirical(samples, alpha=ALPHA):
    arr = np.asarray(samples)
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    if tail.size == 0:
        tail = arr[:1]
    es_abs = -float(tail.mean())
    es_diff = es_abs + float(arr.mean())
    return es_abs, es_diff


def es_analytical_t(mu: float, scale: float, nu: float, alpha: float = ALPHA) -> float:
    z = t.ppf(alpha, df=nu)
    fz = t.pdf(z, df=nu)
    tail_term = scale * ((nu + z * z) / (nu - 1.0)) * fz / alpha
    return -(mu - tail_term)


def write_output(path: Path, es_absolute: float, es_diff_from_mean: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ES Absolute", "ES Diff from Mean"])
        writer.writerow([f"{es_absolute:.17f}", f"{es_diff_from_mean:.17f}"])


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test7_2.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("testout_8.6.csv")

    returns = read_returns(input_path)
    # Fit Student-t 
    nu, mu, scale = t.fit(returns)

    # Simulate from fitted t for empirical ES
    rng = np.random.default_rng(SEED)
    sim = t.rvs(df=nu, loc=mu, scale=scale, size=N_SIM, random_state=rng)

    es_absolute, es_diff_from_mean = es_empirical(sim, ALPHA)

    write_output(output_path, es_absolute, es_diff_from_mean)


if __name__ == "__main__":
    main()