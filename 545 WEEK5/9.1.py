from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t


ALPHA = 0.05
N_SIM = 100000
SEED = 545


def fit_marginal(x: np.ndarray, distribution: str) -> dict:
    if distribution == "Normal":
        mu, sigma = norm.fit(x)
        u_hist = norm.cdf(x, loc=mu, scale=sigma)
        return {"kind": "normal", "mu": mu, "sigma": sigma, "u": u_hist}

    if distribution == "T":
        nu, loc, scale = t.fit(x)
        u_hist = t.cdf(x, df=nu, loc=loc, scale=scale)
        return {"kind": "t", "nu": nu, "loc": loc, "scale": scale, "u": u_hist}

    raise ValueError(f"Unsupported Distribution '{distribution}'. Use 'Normal' or 'T'.")


def inv_cdf(u: np.ndarray, model: dict) -> np.ndarray:
    if model["kind"] == "normal":
        return norm.ppf(u, loc=model["mu"], scale=model["sigma"])
    return t.ppf(u, df=model["nu"], loc=model["loc"], scale=model["scale"])


def var_es(pnl: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    q = float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= q]
    return -q, -float(np.mean(tail))


def main() -> None:
    returns_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test9_1_returns.csv")
    portfolio_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("test9_1_portfolio.csv")
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("testout_9.1.csv")

    returns_df = pd.read_csv(returns_path)
    portfolio_df = pd.read_csv(portfolio_path)

    # Strict schema: assignment file uses these exact columns.
    required_port_cols = ["Stock", "Holding", "Starting Price", "Distribution"]
    missing = [c for c in required_port_cols if c not in portfolio_df.columns]
    if missing:
        raise ValueError(f"Portfolio missing columns: {missing}")

    stocks = list(returns_df.columns)
    for s in stocks:
        if s not in set(portfolio_df["Stock"]):
            raise ValueError(f"Stock '{s}' in returns not found in portfolio.")

    models: dict[str, dict] = {}
    u_cols: list[np.ndarray] = []

    for s in stocks:
        row = portfolio_df.loc[portfolio_df["Stock"] == s].iloc[0]
        dist = str(row["Distribution"])
        x = returns_df[s].to_numpy(dtype=float)
        model = fit_marginal(x, dist)
        models[s] = model
        u_cols.append(model["u"])

    U = np.column_stack(u_cols)
    corr, _ = spearmanr(U, axis=0)
    k = len(stocks)
    corr = np.array(corr, dtype=float).reshape((k, k))

    rng = np.random.default_rng(SEED)
    z_sim = rng.multivariate_normal(np.zeros(k), corr, size=N_SIM)
    u_sim = norm.cdf(z_sim)

    sim_ret = np.zeros((N_SIM, k), dtype=float)
    for j, s in enumerate(stocks):
        sim_ret[:, j] = inv_cdf(u_sim[:, j], models[s])

    rows = []
    for j, s in enumerate(stocks):
        prow = portfolio_df.loc[portfolio_df["Stock"] == s].iloc[0]
        current_value = float(prow["Holding"]) * float(prow["Starting Price"])
        pnl = current_value * sim_ret[:, j]

        var95, es95 = var_es(pnl, ALPHA)
        rows.append(
            {
                "Stock": s,
                "VaR95": var95,
                "ES95": es95,
                "VaR95_Pct": var95 / current_value,
                "ES95_Pct": es95 / current_value,
            }
        )

    out_df = pd.DataFrame(rows, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    out_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
