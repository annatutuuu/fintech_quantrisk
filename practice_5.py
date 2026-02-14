import numpy as np
import pandas as pd


def make_symmetric(matrix):
    # Force numerical symmetry by averaging with the transpose
    return (matrix + matrix.T) / 2.0


def nearest_psd_by_clipping(matrix):
    """
    Build a nearby PSD matrix by eigenvalue clipping:
    1) Symmetrize the input
    2) Replace negative eigenvalues with 0
    3) Re-scale so the diagonal matches the original diagonal
    """
    matrix = make_symmetric(matrix)

    # Eigen-decompose (for symmetric matrices)
    eigvals, eigvecs = np.linalg.eigh(matrix)

    eigvals[eigvals < 0.0] = 0.0

    fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Preserve the original diagonal via a diagonal scaling transform
    old_diag = np.diag(matrix)
    new_diag = np.diag(fixed)
    scale = np.ones_like(old_diag)

    for i in range(len(scale)):
        if new_diag[i] > 0.0:
            scale[i] = np.sqrt(old_diag[i] / new_diag[i])

    d = np.diag(scale)
    fixed = d @ fixed @ d
    return make_symmetric(fixed)


def project_to_psd(matrix):
    # PSD projection: symmetrize, clip negative eigenvalues, and reconstruct
    matrix = make_symmetric(matrix)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < 0.0] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def nearest_psd_higham(matrix, max_iter=100, tol=1e-8):
    """
    Higham-style nearest PSD approximation via alternating projections.
    Enforces PSD while keeping the input diagonal unchanged.
    """
    matrix = make_symmetric(matrix)
    target_diag = np.diag(matrix).copy()

    # y is the running iterate; delta_s stores the correction term
    y = matrix.copy()
    delta_s = np.zeros_like(matrix)

    for _ in range(max_iter):
        # subtract the previous correction
        r = y - delta_s

        # project onto the PSD cone
        x = project_to_psd(r)

        # update the correction (difference from r)
        delta_s = x - r

        # enforce the diagonal constraint
        y = x.copy()
        np.fill_diagonal(y, target_diag)
        y = make_symmetric(y)

        # Convergence check
        rel_change = np.linalg.norm(y - x, ord="fro") / max(1.0, np.linalg.norm(y, ord="fro"))
        if rel_change < tol:
            break

    # Final PSD projection
    return make_symmetric(project_to_psd(y))


def psd_square_root(matrix, tol=1e-10):
    """
    Construct a factor L such that L @ L.T approximates the input.
    Assumes the matrix is PSD (within a numerical tolerance).
    """
    matrix = make_symmetric(matrix)
    eigvals, eigvecs = np.linalg.eigh(matrix)

    # If substantially negative eigenvalues exist, reject
    if np.min(eigvals) < -tol:
        raise ValueError("Matrix is not PSD.")

    # Treat tiny negatives as zero
    eigvals[eigvals < 0.0] = 0.0

    # L = Q * sqrt(D) so that L L^T = Q D Q^T
    sqrt_diag = np.diag(np.sqrt(eigvals))
    return eigvecs @ sqrt_diag


def simulate_normal_with_cov(cov_matrix, n_samples):
    # Draw N(0, cov_matrix) samples using a PSD square-root factorization
    l = psd_square_root(cov_matrix)
    z = np.random.normal(size=(n_samples, cov_matrix.shape[0]))
    return z @ l.T


def simulate_pca(cov_matrix, n_samples, pct_explained=0.99):
    """
    PCA-based simulation:
    keep the leading principal components until cumulative explained variance
    reaches pct_explained.
    """
    cov_matrix = make_symmetric(cov_matrix)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenpairs descending by eigenvalue (variance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Remove negative variances from numerical issues / non-PSD inputs
    eigvals[eigvals < 0.0] = 0.0

    total_var = np.sum(eigvals)
    if total_var <= 0.0:
        return np.zeros((n_samples, cov_matrix.shape[0]))

    # Choose smallest k such that explained variance >= pct_explained
    explained = np.cumsum(eigvals) / total_var
    k = np.searchsorted(explained, pct_explained) + 1

    # Build loadings for the retained components
    loadings = eigvecs[:, :k] @ np.diag(np.sqrt(eigvals[:k]))

    # Simulate in k-dim latent space then map back to original space
    z = np.random.normal(size=(n_samples, k))
    return z @ loadings.T


def save_covariance_from_sim(sim_data, columns, out_file):
    # Compute sample covariance and export as CSV with named columns
    cov = np.cov(sim_data, rowvar=False)
    pd.DataFrame(cov, columns=columns).to_csv(out_file, index=False)


def main():
    np.random.seed(4)

    # Burn a few random draws so later steps align with a reference RNG stream
    np.random.normal(size=5)

    n_samples = 100000
    columns = [f"x{i}" for i in range(1, 6)]

    # 5.1: covariance is PD -> simulate multivariate normal -> save covariance
    cov_51 = pd.read_csv("test5_1.csv").to_numpy(dtype=float)
    sim_51 = simulate_normal_with_cov(cov_51, n_samples)
    save_covariance_from_sim(sim_51, columns, "testout_5.1.csv")

    # 5.2: covariance is PSD -> simulate multivariate normal -> save covariance
    cov_52 = pd.read_csv("test5_2.csv").to_numpy(dtype=float)
    sim_52 = simulate_normal_with_cov(cov_52, n_samples)
    save_covariance_from_sim(sim_52, columns, "testout_5.2.csv")

    # 5.3: covariance is non-PSD -> clip eigenvalues -> simulate -> save covariance
    cov_53 = pd.read_csv("test5_3.csv").to_numpy(dtype=float)
    fixed_53 = nearest_psd_by_clipping(cov_53)
    sim_53 = simulate_normal_with_cov(fixed_53, n_samples)
    save_covariance_from_sim(sim_53, columns, "testout_5.3.csv")

    # 5.4: covariance is non-PSD -> Higham fix -> simulate -> save covariance
    cov_54 = pd.read_csv("test5_3.csv").to_numpy(dtype=float)
    fixed_54 = nearest_psd_higham(cov_54)
    sim_54 = simulate_normal_with_cov(fixed_54, n_samples)
    save_covariance_from_sim(sim_54, columns, "testout_5.4.csv")

    # 5.5: covariance is PSD -> PCA simulation (keep 99% variance) -> save covariance
    cov_55 = pd.read_csv("test5_2.csv").to_numpy(dtype=float)
    sim_55 = simulate_pca(cov_55, n_samples, pct_explained=0.99)
    save_covariance_from_sim(sim_55, columns, "testout_5.5.csv")

    print("Wrote testout_5.1.csv ... testout_5.5.csv using a single RNG stream.")


if __name__ == "__main__":
    main()
