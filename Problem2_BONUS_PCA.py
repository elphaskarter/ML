import numpy as np
import matplotlib.pyplot as plt
from Problem1_a_HSI_DATA import hsi_data, bands_float
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

def pcs_impact_analysis():
    H, W, D = hsi_data.shape
    X = hsi_data.reshape(-1, D)
    pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(X, standardize=True)

    # Mean signal in original domain
    mean_signal = np.mean(hsi_data, axis=(0, 1))  # shape (D,)

    # First five PCs in standardized domain
    n_pcs = 5
    pcs_first_five_std = pcs[:, :n_pcs]          # shape (D, 5)
    X_proj_first_five = X_proj[:, :n_pcs]        # shape (m, 5)

    # Un-standardize PC vectors to original domain
    pcs_first_five_unstd = pcs_first_five_std * std_vec.reshape(-1, 1)

    # Standard deviation in PC space
    variability_pc = np.std(X_proj_first_five, axis=0)  # shape (5,)

    # Plot impact
    fig, axs = plt.subplots(n_pcs, 1, figsize=(10, 15))
    for i in range(n_pcs):
        pc_unstd_i = pcs_first_five_unstd[:, i]   # shape (D,)
        scale_i = variability_pc[i]

        positive_impact = mean_signal + scale_i * pc_unstd_i
        negative_impact = mean_signal - scale_i * pc_unstd_i

        axs[i].plot(bands_float, mean_signal, label="Mean Signal", color="black", linestyle="--")
        axs[i].plot(bands_float, positive_impact, label="Positive Impact", color="blue")
        axs[i].plot(bands_float, negative_impact, label="Negative Impact", color="red")

        axs[i].set_title(f"Impact Plot for PC {i+1}")
        axs[i].set_xlabel("Wavelength (nm)")
        axs[i].set_ylabel("Magnitude of Impact")
        axs[i].set_ylim(0, 2.6)
        axs[i].set_xlim(bands_float[0], bands_float[-4])
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pcs_impact_analysis()