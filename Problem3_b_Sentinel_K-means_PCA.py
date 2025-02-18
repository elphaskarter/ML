#--Problem 3b--
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from load_sentinel_data import sentinel_data, masked_data, cwv_data
from Problem2_a_b_PCA import SVD_PCA
from Problem3_a_K_means import k_means
import matplotlib
import contextlib
import io

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from Problem1_a_HSI_DATA import hsi_data, bands_float

matplotlib.use(original_backend)
plt.close('all')

def plot_bands_2x6(data, cvw):

    assert masked_data.shape[2] == 12
    assert len(cvw) == 12
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for i in range(12):
        row, col = divmod(i, 6)  
        band_array = data[:, :, i]
        im = axes[row, col].imshow(band_array, cmap=cmocean.cm.gray)
        cbar = plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        cbar.set_label('Reflectance', fontsize=10)
        axes[row, col].set_title(f'Band {i+1}: {cwv_data[i]} nm', fontsize=10)        
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

plt.ion()
plot_bands_2x6(masked_data, cwv_data) # visualize sentinel data
H, W, D = masked_data.shape
X = masked_data.reshape(-1, D)
pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(X, standardize=True)

idx = [2, 3, 4, 5] 
X_proj_idx = X_proj[:, idx]  # shape (H*W, 4)

# Reshape back to image form (H, W, 4)
X_proj_idx_img = X_proj_idx.reshape(H, W, 4)

# Run K-means
k = 4
max_iter = 100
tol = 1e-4
random_state = 4

H, W, num_pcs = X_proj_idx_img.shape  # (H, W, 4)
X_pca_reshaped = X_proj_idx_img.reshape(-1, num_pcs)  # (H*W, 4)
centroids, labels = k_means(X_pca_reshaped, k, max_iter, tol, random_state)

# Plot each of the 10 principal components
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes = axes.ravel()  # Flatten the 2D array

for i, pc_col in enumerate(idx):
    axes[i].imshow(X_proj_idx_img[:, :, i], cmap=cmocean.cm.thermal)
    axes[i].set_title(f"PC{pc_col+1}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# Plot lower dimenstionality data
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    X_pca_reshaped[:, 0],  # PC #3 
    X_pca_reshaped[:, 1],  # PC #4
    c=labels, cmap='viridis', s=2, alpha=0.7)
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=40)
ax.set_title(f'K-Means Clusters: PC{idx[0]+1} vs. PC{idx[1]+1}')
ax.set_xlabel(f'PC{idx[0]+1}')
ax.set_ylabel(f'PC{idx[1]+1}')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
plt.show()

# Plot higher dimenstionality data
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    X_pca_reshaped[:, 2],  # PC 5 
    X_pca_reshaped[:, 3],  # PC 6
    c=labels, cmap='viridis', s=2, alpha=0.7)
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=40)
ax.set_title(f'K-Means Clusters: PC{idx[2]+1} vs. PC{idx[3]+1}')
ax.set_xlabel(f'PC{idx[2]+1}')
ax.set_ylabel(f'PC{idx[3]+1}')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
plt.show() 

# Spartial view on the k clusters
labels_2d = labels.reshape(H, W)
labels_2d_float = labels_2d.astype(float)
labels_2d_float[labels_2d_float <= -1] = np.nan
plt.imshow(labels_2d_float, cmap='tab20')
plt.title("K-Means Clusters (Spatial View)")
plt.colorbar(label='Cluster Label')
plt.show()
 