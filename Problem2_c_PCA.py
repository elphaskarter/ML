# --Problem 2(b) pca analysis--
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from Problem1_a_HSI_DATA import hsi_data
from Problem2_a_b_PCA import SVD_PCA
import contextlib
import io
import matplotlib

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from Problem1_a_HSI_DATA import hsi_data, bands_float
    from Problem2_a_b_PCA import SVD_PCA

matplotlib.use(original_backend)
plt.close('all')

def recon_error(hsi_data, pcs, mean_vec, std_vec, X_projected, k_values):
    """
    Computes and returns the L2 reconstruction error for a range of k-values.
    Parameters:
    -----------
    hsi_data : The original hyperspectral cube (H, W, D).
    pcs : Principal components (d, d) from PCA.
    mean_vec : The mean of each feature (d,).
    std_vec : The std dev of each feature if standardize=True, otherwise None.
    X_projected : The data projected onto all principal components (n, d), 
        where n = H*W, d = number of bands.
    k_values : The list of k-values to evaluate.
    Returns:
    --------
    errors : Dictionary mapping k -> L2 reconstruction error.
    """
    H, W, d = hsi_data.shape
    X = hsi_data.reshape(-1, d)  # shape (n, d)
    n = X.shape[0]

    # Compute the centered (standardized) version of X
    if std_vec is not None:
        X_centered = (X - mean_vec) / std_vec # Standardized
    else:
        X_centered = X - mean_vec # Mean-centered only
    errors = {}

    for k in k_values:
        V_k = pcs[:, :k] # Partial reconstruction using the first k PCs
        X_proj_k = X_projected[:, :k] # X_proj_k is the first k dimensions from X_projected
        X_recon_k = np.dot(X_proj_k, V_k.T)  # Reconstruct in the centered/standardized domain
        diff = X_centered - X_recon_k # Compute L2 error = || X_centered - X_recon_k ||_F
        error_k = np.linalg.norm(diff, 'fro')  # Frobenius norm
        errors[k] = error_k

    return errors

def main():
    H, W, D = hsi_data.shape
    X = hsi_data.reshape(-1, D)

    # Run PCA (standardize = True or False)
    pcs, eigenvals, mean_vec, std_vec, X_projected = SVD_PCA(X, standardize=True)
    k_values = [1, 10, 50, 100, D]  # Evaluate the reconstruction error (D is # of bands)
    error_L2 = recon_error(hsi_data, pcs, mean_vec, std_vec, X_projected, k_values)

    # Plot the results
    plt.ion()
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [error_L2[k] for k in k_values], marker='^', linestyle='--')
    plt.title("L2 Reconstruction Error vs. Number of PCs")
    plt.xlabel("Number of Principal Components (k)")
    plt.ylabel("L2 Reconstruction Error")
    plt.tight_layout()
    plt.show()
    plt.ioff()

if __name__ == "__main__":
    main()
