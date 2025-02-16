import numpy as np
import matplotlib.pyplot as plt
from Problem1_a_HSI_DATA import hsi_data, bands_float
from Problem2_a_b_PCA import SVD_PCA
import matplotlib
import cmocean


original_backend = matplotlib.get_backend()
matplotlib.use('Agg')  # Set backend to non-interactive

from Problem1_a_HSI_DATA import hsi_data, bands_float
from Problem2_a_b_PCA import SVD_PCA

matplotlib.use(original_backend)

def min_max_normalize(spectral_profile):
    min_val = np.min(spectral_profile)
    max_val = np.max(spectral_profile)
    norm_profile = (spectral_profile - min_val) / (max_val - min_val) # min-max normalization
    return norm_profile

def calculate_snr(image, no_data_value=np.nan):
    # SNR for each band
    snr = np.zeros(image.shape[2])  
    for band in range(image.shape[2]):
        band_data = image[:, :, band].flatten()

        # Mask no-data pixels
        if np.isnan(no_data_value):
            valid_pixels = band_data[~np.isnan(band_data)]
        else:
            valid_pixels = band_data[band_data != no_data_value]

        # Calculate mean (µ) and standard deviation (σ)
        mu = np.mean(valid_pixels)
        sigma = np.std(valid_pixels)
        snr[band] = mu / sigma if sigma != 0 else np.nan  # Avoid division by zero
    return snr

def main():
    # Load the hyperspectral data
    H, W, D = hsi_data.shape
    X = hsi_data.reshape(-1, D)  # (m_samples, d_features)
    pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(X, standardize=True) # SVD and PCA
    
    # cumulative variance
    vari_ratio = eigenvals / np.sum(eigenvals)
    cumul_var = np.cumsum(vari_ratio)

    # number of PCs to retain (99% variance)
    n_retained_comp = np.argmax(cumul_var >= 0.99) + 1
    X_proj_retained = X_proj[:, :n_retained_comp]

    # Reconstruction of cleaned PCs
    pcs_retained = X_proj_retained.reshape(H, W, n_retained_comp) 
    print(f"Number of PCs to retain: {n_retained_comp}")
    
    n_rows = int(np.ceil(np.sqrt(n_retained_comp)))
    n_cols = int(np.ceil(n_retained_comp / n_rows))

    plt.ion()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axes = axes.ravel() 

    for i in range(n_retained_comp):
        axes[i].imshow(pcs_retained[:, :, i], cmap=cmocean.cm.thermal)
        axes[i].set_title(f"PC{i+1}")
        axes[i].axis("off")

    for i in range(n_retained_comp, n_rows * n_cols): # Hide unused subplots
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    pixels = [(828, 861), (416, 1006), (432, 569), (763, 471), (960, 277)]
    n_pixels = len(pixels)
    n_rows = int(np.ceil(np.sqrt(n_pixels)))  # Determine number of rows
    n_cols = int(np.ceil(n_pixels / n_rows))  # Determine number of columns
    
    # Original data reconstruction
    X_recons_all_pcs = np.dot(X_proj, pcs.T)
    X_recons_all_pcs_img = X_recons_all_pcs.reshape(H, W, D)

    # Extract and plot spectral profiles for original data
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.ravel()

    for i, (x, y) in enumerate(pixels):
        orig_spec_profile = X_recons_all_pcs_img[y, x, :]  
        orig_norm_profile = min_max_normalize(orig_spec_profile)  
        axes[i].plot(bands_float, orig_norm_profile, color='red', label=f"Pixel ({x}, {y})")
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("Reflectance")
        axes[i].set_title(f"Pixel ({x}, {y})")
        axes[i].grid()
        axes[i].legend()

    # Hide unused subplots
    for i in range(n_pixels, n_rows * n_cols):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
        
    # diagonal matrix of eigenvalues (d x d)
    S_modified = np.diag(eigenvals.copy())
    S_modified[n_retained_comp:, n_retained_comp:] = 0  # Set less significant eigenvalues to zero

    # Reconstructed data using the modified PC matrix
    X_recon_PCs_clean = np.dot(X_proj, S_modified @ pcs.T)
    X_recon_clean_PCs_img = X_recon_PCs_clean.reshape(H, W, D)

    # spectral profile for reconstructed data
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.ravel()

    for i, (x, y) in enumerate(pixels):
        spectral_profile = X_recon_clean_PCs_img[y, x, :]  # Extract spectral profile
        clean_norm_profile = min_max_normalize(spectral_profile)  # Normalize the profile
        axes[i].plot(bands_float, clean_norm_profile, color='red', label=f"Pixel ({x}, {y})")
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("Reflectance")
        axes[i].set_title(f"Pixel ({x}, {y})")
        axes[i].grid()
        axes[i].legend()

    for i in range(n_pixels, n_rows * n_cols): # remove unused plots
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()