import numpy as np
import matplotlib.pyplot as plt
from Problem1_a_HSI_DATA import hsi_data, bands_float
from Problem2_a_b_PCA import SVD_PCA
import matplotlib
import cmocean

original_backend = matplotlib.get_backend()
matplotlib.use('Agg')  # Set backend to non-interactive
matplotlib.use(original_backend)

def min_max_normalize(spectral_profile):
    min_val = np.min(spectral_profile)
    max_val = np.max(spectral_profile)
    norm_profile = (spectral_profile - min_val) / (max_val - min_val) # min-max normalization
    return norm_profile

def std_with_pca_params(spectral_profile, mean_vec, std_vec):
    """
    Standardize a spectral profile using PCA mean and std.
    """
    std_data = (spectral_profile - mean_vec) / std_vec
    return std_data

def calculate_snr(image, no_data_value=np.nan):
    snr = np.zeros(image.shape[2])  
    for band in range(image.shape[2]):
        band_data = image[:, :, band].flatten()
        if np.isnan(no_data_value):
            valid_pixels = band_data[~np.isnan(band_data)]
        else:
            valid_pixels = band_data[band_data != no_data_value]
        mu = np.mean(valid_pixels)
        sigma = np.std(valid_pixels)
        snr[band] = mu / sigma if sigma != 0 else np.nan
    return snr

def main():
    H, W, D = hsi_data.shape
    X = hsi_data.reshape(-1, D)
    pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(X, standardize=True)
    
    # Reconstruct data using all PCs and reverse standardization
    X_recons_all_pcs = np.dot(X_proj, pcs.T) * std_vec + mean_vec  # Reverse standardization
    X_recons_all_pcs_img = X_recons_all_pcs.reshape(H, W, D)

    # Find retained PCs (99% variance)
    vari_ratio = eigenvals / np.sum(eigenvals)
    cumul_var = np.cumsum(vari_ratio)
    n_retained_comp = np.argmax(cumul_var >= 0.99) + 1
    X_proj_retained = X_proj[:, :n_retained_comp]
    pcs_retained = X_proj_retained.reshape(H, W, n_retained_comp) # Reconstruction of cleaned PCs

    # Reconstructing data with retained PCs and reverse standardization (in the original feature space)
    S_modified = np.diag(eigenvals.copy())
    S_modified[n_retained_comp:, n_retained_comp:] = 0
    X_recon_PCs_clean = np.dot(X_proj, S_modified @ pcs.T) * std_vec + mean_vec  # Reverse standardization
    X_recon_clean_PCs_img = X_recon_PCs_clean.reshape(H, W, D)

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
    
    # Define pixels
    pixels = [(828, 861), (416, 1006), (432, 569), (763, 471), (488, 206)]
    n_pixels = len(pixels)
    n_rows = int(np.ceil(np.sqrt(n_pixels)))
    n_cols = int(np.ceil(n_pixels / n_rows))

    # Plot spectral profiles for original data (unstandardized)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.ravel()
    for i, (x, y) in enumerate(pixels):
        spectral_profile = X_recons_all_pcs_img[y, x, :]
        std_profile_all = std_with_pca_params(spectral_profile, mean_vec, std_vec)
        norm_std_profile_all = min_max_normalize(std_profile_all)
        axes[i].plot(bands_float, norm_std_profile_all, color='red', label=f"Pixel ({x}, {y})")
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("Reflectance")
        axes[i].set_title(f"Pixel ({x}, {y})")
        axes[i].grid()
        axes[i].legend()
    for i in range(n_pixels, n_rows * n_cols):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    # Plot spectral profiles for denoised spaces
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.ravel()
    for i, (x, y) in enumerate(pixels):
        # Extract and standardize using PCA mean/std
        spectral_profile = X_recon_clean_PCs_img[y, x, :]
        std_profile_denoised = std_with_pca_params(spectral_profile, mean_vec, std_vec)
        norm_std_profile_denoised = min_max_normalize(std_profile_denoised)
        axes[i].plot(bands_float, norm_std_profile_denoised, color='red', label=f"Pixel ({x}, {y})")
        axes[i].set_xlabel("Wavelength (nm)")
        axes[i].set_ylabel("Reflectance")
        axes[i].set_title(f"Pixel ({x}, {y})")
        axes[i].grid()
        axes[i].legend()
    for i in range(n_pixels, n_rows * n_cols):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    # Calculate SNR on original scale (reversed standardization)
    snr_original = calculate_snr(X_recons_all_pcs_img)
    snr_denoised = calculate_snr(X_recon_clean_PCs_img)

    print("SNR (Original Image):", np.nanmean(snr_original))
    print("SNR (Transformed Image):", np.nanmean(snr_denoised))

    # Plot SNR vs. wavelength
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
    if bands_float is not None and len(bands_float) == D:
        x_axis = bands_float
        x_label = "Wavelength (nm)"
    else:
        x_axis = np.arange(D)
        x_label = "Band Index"

    # Plot SNR for original data (all PCs)
    axs[0].plot(x_axis, snr_original, label="All PCs")
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("SNR (Mean / Std)")
    axs[0].set_title("SNR - All PCs")
    axs[0].grid(True)
    axs[0].legend()

    # Plot SNR for denoised data (99% PCA)
    axs[1].plot(x_axis, snr_denoised, color='orange', label="Truncated (99%)")
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel("SNR (Mean / Std)")  # Keep y-label for clarity
    axs[1].set_title("SNR - 99% PCA")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()