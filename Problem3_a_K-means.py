# Problem 3(a)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image

def standardize_data(data):
    scaler = StandardScaler()
    stdzd_data = scaler.fit_transform(data)
    return stdzd_data, scaler

def k_means(data, k, max_iter, tol, random_state=None):
    if random_state is not None: # Initialize K clusters
        np.random.seed(random_state)
    n_samples, n_features = data.shape

    # Initial centroids
    random_idx = np.random.choice(n_samples, size=k, replace=False)
    centroids = data[random_idx, :].copy()
    labels = np.zeros(n_samples, dtype=int)
    
    # Iterate until convergence or max_iter
    for iteration in range(max_iter):
        old_centroids = centroids.copy()

        # Assign each point to the nearest centroid
        for i in range(n_samples): 
            distances = np.linalg.norm(data[i] - centroids, axis=1) 
            labels[i] = np.argmin(distances)
        
        # Update centroids
        for cluster_id in range(k):
            cluster_points = data[labels == cluster_id] 
            if len(cluster_points) > 0:
                centroids[cluster_id] = np.mean(cluster_points, axis=0)
        
        # Check for convergence
        centroid_shift = np.linalg.norm(centroids - old_centroids)
        if centroid_shift < tol:
            print(f"Converged after {iteration+1} iterations.")
            break
    return centroids, labels

# Call main
def main():
    # Load image
    image_path = "jellybeans.tiff"
    image = Image.open(image_path)
    image = np.array(image)  # shape: (height, width, channels)
    height, width, channels = image.shape
    data_points = image.reshape(-1, channels)

    # Standardize
    stdzd_data, scaler = standardize_data(data_points) 
    
    # Run K-means
    k = 10
    max_iter = 100
    tol = 1e-4
    random_state = 4
    centroids, labels = k_means(stdzd_data, k, max_iter, tol, random_state)

    # convert to uint8 
    centroids_original = scaler.inverse_transform(centroids)
    segmented_data = centroids_original[labels]
    segmented_image = segmented_data.reshape((height, width, channels))
    segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    ax1 = axes[0]
    scatter = ax1.scatter(stdzd_data[:, 0], stdzd_data[:, 1], c=labels, cmap='viridis')
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=40)
    ax1.set_title('K-Means Clusters')

    ax2 = axes[1]
    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Image')
    ax2.axis('off') 

    ax3 = axes[2]
    ax3.imshow(image)
    ax3.set_title('Original Image')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()