import numpy as np
import matplotlib.pyplot as plt
from Problem1_a_HSI_DATA import hsi_data
from Problem2_a_b_PCA import SVD_PCA
from Problem3_a_K_means import k_means
import io
import contextlib
import matplotlib
import cmocean
from sklearn.cluster import MiniBatchKMeans

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from Problem1_a_HSI_DATA import hsi_data, bands_float
    from Problem2_a_b_PCA import SVD_PCA

matplotlib.use(original_backend)
plt.close('all')

# hyperspectral data to resize
rows, cols, n_bands = hsi_data.shape
print(f"HSI data shape: {rows} x {cols} x {n_bands}")

row_start = 10
col_start = 300
patch_size = 250

patch = hsi_data[row_start:row_start + patch_size, col_start:col_start + patch_size, :]
print(f"Patch shape: {patch.shape}")

# Perfrom PCA
H, W, n_bands = patch.shape
X = patch.reshape(-1, n_bands) # original data
pcs, eigenvals, mean_vec, std_vec, X_proj = SVD_PCA(X, standardize=True)

# Transform data onto components 2, 5, 10, 50 and 100
idx = [1, 4, 9, 49, 99] 
X_proj_idx = X_proj[:, idx]  # shape (H*W, 5)

# Reshape back to image form (H, W, 5)
X_proj_idx_img = X_proj_idx.reshape(H, W, 5)
H, W, num_pcs = X_proj_idx_img.shape  # (H, W, 5)
X_pca_reshaped = X_proj_idx_img.reshape(-1, num_pcs)  # (H*W, 5)
# centroids, labels = k_means(X_pca_reshaped, k, max_iter, tol, random_state)

# MiniBatchKMeans parameters
k = 6
mbk_pca = MiniBatchKMeans(n_clusters=k, random_state=4, batch_size=1000, max_iter=100, tol=1e-10)
mbk_pca.fit(X_pca_reshaped)
labels_pca = mbk_pca.labels_
centroids_pca = mbk_pca.cluster_centers_
miniK_pca_img = labels_pca.reshape(H, W)

# Plot cluster labels (just as an example)
plt.imshow(miniK_pca_img, cmap='tab20')
plt.title('MiniBatchKMeans Clusters (PCA Data)')
plt.colorbar(label='Cluster Label')
plt.show()

# Plot band 1 to visualize 250x250
band_1 = patch[:, :, 0]
plt.imshow(band_1, cmap='gray')
plt.title("Band 1")
plt.colorbar(label='Reflectance')
plt.show()

# Plot each of the 10 principal components
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes = axes.ravel()  # Flatten the 2D array

for i, pc_col in enumerate(idx):
    axes[i].imshow(X_proj_idx_img[:, :, i], cmap=cmocean.cm.thermal)
    axes[i].set_title(f"PC{pc_col+1}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# K_mean on original patch (H*W, n_bands)
mbk_original = MiniBatchKMeans(n_clusters=k, random_state=4, batch_size=2048, max_iter=100, tol=1e-10)
mbk_original.fit(X)
labels_orig = mbk_original.labels_
labels_orig_img = labels_orig.reshape(H, W)
centroids_orig = mbk_original.cluster_centers_

# Spatial plot cluster labels 
plt.imshow(labels_orig_img, cmap='tab20')
plt.title('MiniBatchKMeans Clusters (Original Data)')
plt.colorbar(label='Cluster Label')
plt.show()

# scatter plot of lower dimenstionality data
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    X_pca_reshaped[:, 0],  # PC #2
    X_pca_reshaped[:, 1],  # PC #5
    c=labels_pca, cmap='viridis', s=2, alpha=0.7)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='o', s=40)
ax.set_title(f'K-Means Clusters: PC{idx[0]+1} vs. PC{idx[1]+1}')
ax.set_xlabel(f'PC{idx[0]+1}')
ax.set_ylabel(f'PC{idx[1]+1}')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
plt.show()

# Scatter plot for higher dimenstionality data
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    X[:, 0],  # PC #2
    X[:, 1],  # PC #5
    c=labels_orig, cmap='viridis', s=2, alpha=0.7)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='o', s=40)
ax.set_title(f'K-Means Clusters: PC{idx[0]+1} vs. PC{idx[1]+1}')
ax.set_xlabel(f'PC{idx[0]+1}')
ax.set_ylabel(f'PC{idx[1]+1}')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
plt.show()