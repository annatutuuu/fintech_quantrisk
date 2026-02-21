import csv
from pathlib import Path
import sys

from scipy.stats import t


def read_returns(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")
        for row in reader:
            if row:
                values.append(float(row[0]))
    if len(values) < 3:
        raise ValueError("Need at least 3 observations.")
    return values


def compute_es_t_mle(returns: list[float], alpha: float = 0.05) -> tuple[float, float]:
    # MLE fit for generalized t: X ~ mu + s * t_nu
    nu, mu, s = t.fit(returns)

    z_alpha = t.ppf(alpha, df=nu)
    t_pdf_z = t.pdf(z_alpha, df=nu)

    # Loss-style ES for X = mu + s*T
    tail_term = s * ((nu + z_alpha * z_alpha) / (nu - 1.0)) * t_pdf_z / alpha
    es_absolute = -(mu - tail_term)
    es_diff_from_mean = mu + es_absolute
    return es_absolute, es_diff_from_mean


def write_output(path: Path, es_absolute: float, es_diff_from_mean: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ES Absolute", "ES Diff from Mean"])
        writer.writerow([f"{es_absolute:.17f}", f"{es_diff_from_mean:.17f}"])


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test7_2.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("testout_8.5.csv")

    returns = read_returns(input_path)
    es_absolute, es_diff_from_mean = compute_es_t_mle(returns, alpha=0.05)
    write_output(output_path, es_absolute, es_diff_from_mean)


if __name__ == "__main__":
    main()
