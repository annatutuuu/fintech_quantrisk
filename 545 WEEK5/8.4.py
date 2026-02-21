import csv
from pathlib import Path
from statistics import NormalDist, mean, stdev
import sys


def read_returns(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")
        for row in reader:
            if not row:
                continue
            values.append(float(row[0]))
    if len(values) < 2:
        raise ValueError("Need at least 2 return observations.")
    return values


def compute_es_normal(returns: list[float], alpha: float = 0.05) -> tuple[float, float]:
    mu = mean(returns)
    sigma = stdev(returns)

    z_alpha = NormalDist().inv_cdf(alpha)
    phi_z = NormalDist().pdf(z_alpha)

    es_absolute = -(mu - sigma * phi_z / alpha)
    es_diff_from_mean = mu + es_absolute
    return es_absolute, es_diff_from_mean


def write_output(path: Path, es_absolute: float, es_diff_from_mean: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ES Absolute", "ES Diff from Mean"])
        writer.writerow([f"{es_absolute:.17f}", f"{es_diff_from_mean:.17f}"])


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test7_1.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("testout_8.4.csv")

    returns = read_returns(input_path)
    es_absolute, es_diff_from_mean = compute_es_normal(returns, alpha=0.05)
    write_output(output_path, es_absolute, es_diff_from_mean)


if __name__ == "__main__":
    main()