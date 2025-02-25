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
plt.ioff()

# hyperspectral data
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

idx_3 = list(range(3))         # first 3 components
idx_4 = list(range(4))         # first 4 components
idx_5 = list(range(5))         # first 5 components
idx_6 = list(range(6))         # first 6 components

X_proj_3 = X_proj[:, idx_3]    # shape (H*W, 3)
X_proj_4 = X_proj[:, idx_4]    # shape (H*W, 4)
X_proj_5 = X_proj[:, idx_5]    # shape (H*W, 5)
X_proj_6 = X_proj[:, idx_6]    # shape (H*W, 6)

X_proj_3_img = X_proj_3.reshape(H, W, 3)
X_proj_4_img = X_proj_4.reshape(H, W, 4)
X_proj_5_img = X_proj_5.reshape(H, W, 5)
X_proj_6_img = X_proj_6.reshape(H, W, 6)

# store all transforms in dictionery
image_dict = {
    3: X_proj_3_img,
    4: X_proj_4_img,
    5: X_proj_5_img,
    6: X_proj_6_img
}

# Dynamically create only 3 subplots
n = 4
current_image = image_dict[n]

# plot
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))  
axes = axes.ravel() 
for i in range(n):  
    axes[i].imshow(current_image[:, :, i], cmap=cmocean.cm.thermal)
    axes[i].set_title(f"PC{i+1}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# # Plot each of the 3 principal components
# fig, axes = plt.subplots(1, 5, figsize=(20, 5))
# axes = axes.ravel()  # Flatten the 2D array

# for i in range(3):  # We have 5 components
#     axes[i].imshow(X_proj_3_img[:, :, i], cmap=cmocean.cm.thermal)
#     axes[i].set_title(f"PC{i+1}")
#     axes[i].axis("off")

# plt.tight_layout()
# plt.show()

# Run K-means
k = 4
max_iter = 100
tol = 1e-4
random_state = 4

H, W, num_pcs = current_image.shape  # (H, W, 4)
X_pca_reshaped = current_image.reshape(-1, num_pcs)  # (H*W, 4)
centroids, labels = k_means(X_pca_reshaped, k, max_iter, tol, random_state)

# Scatter plot for the 2D PCA (first 2 components)
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    X_pca_reshaped[:, 0],  # PC #1
    X_pca_reshaped[:, 1],  # PC #2
    c=labels, cmap='viridis', s=2, alpha=0.7)
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=40)
ax.set_title(f'K-Means Clusters: PC{1} vs. PC{2}')
ax.set_xlabel(f'PC{1}')
ax.set_ylabel(f'PC{2}')
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

if __name__ == "__main__":
    pass
