# --pca analysis--
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from Problem1_a_HSI_DATA import hsi_data
import contextlib
import io
import matplotlib

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from Problem1_a_HSI_DATA import hsi_data, bands_float

matplotlib.use(original_backend)
plt.close('all')

# SVD and PCA
def SVD_PCA(data, standardize=False):
    """
    Performs PCA on a data matrix data_X of shape (m_samples, d_features),
    using Singular Value Decomposition (SVD).

    Parameters:
    -----------
    X : np.ndarray

    Returns: (np.ndarray, np.ndarray, np.ndarray)
    --------
    pcs : Principal components (eigenvectors) of shape (d, d).
        Each column is a principal component.
    eigenvalues : Eigenvalues corresponding to the principal components,
        sorted in descending order. Shape (d,).
    mean_arr : The mean of each feature used for centering. Shape (d,).
        Otherwise None.
    """
    mean_arr = np.mean(data, axis=0) # Compute mean of each feature

    if standardize: # compute std_dev of each feature
        std_arr = np.std(data, axis=0, ddof=1) 
        
        std_arr[std_arr == 0] = 1.0 # Avoid division by zero
        X_centered = (data - mean_arr) / std_arr         # Standardize

    else:
        std_arr = None
        X_centered = data - mean_arr # Mean-center here

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False) # Compute the SVD 
    V = Vt.T     # Principal components:

    n_samples = X_centered.shape[0]  
    eigenvalues = (S ** 2) / (n_samples - 1) # Compute eigenvalues
    X_projected = np.dot(X_centered, V) # Project the data on PCAs

    return V, eigenvalues, mean_arr, std_arr, X_projected

def main():
    H, W, D = hsi_data.shape
    X = hsi_data.reshape(-1, D)
    pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(X, standardize=True)
    X_proj_10 = X_proj[:, :10]  # shape (H*W, 10)

    # Reshaping back to image form (H, W, 10)
    X_proj_10_img = X_proj_10.reshape(H, W, 10) # transformed data into lower dimension

    # Plotting each of the 10 principal components
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()  # Flatten the 2D array

    for i in range(10):
        axes[i].imshow(X_proj_10_img[:, :, i], cmap=cmocean.cm.thermal)
        axes[i].set_title(f"\nPC{i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plt.ion()
    main()

    # data = hsi_data.reshape(-1, hsi_data.shape[-1])
    # pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(data, standardize=True)

    # print("Principal Components (each column is a component):\n", pcs)
    # print("Eigenvalues:\n", eigenvals)
    # print("Mean Array:\n", mean_vec)
    # # print("Std Dev Array:\n", std_vec)
    # # print("Projected Data:\n", X_proj)
