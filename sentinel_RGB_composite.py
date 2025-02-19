# --Sentinel RGB image---
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

def plot_rgb_composite(masked_data):
    red_band = masked_data[:, :, 3].astype(float)  # Band 4 (665 nm)
    green_band = masked_data[:, :, 2].astype(float)  # Band 3 (560 nm)
    blue_band = masked_data[:, :, 1].astype(float)  # Band 2 (490 nm)

    def process_band(band):
        """Process individual band with non-negative enforcement and stretching"""
        valid_pixels = band[~band.mask]
        p_low, p_high = np.percentile(valid_pixels, (2, 98)) if valid_pixels.size > 0 else (0, 1)
        rescaled_img = rescale_intensity(band.data, in_range=(p_low, p_high), out_range=(0, 1))
        return rescaled_img
    
    red = process_band(red_band)
    green = process_band(green_band)
    blue = process_band(blue_band)

    # Create RGB composite
    rgb = np.stack([red, green, blue], axis=-1)

    plt.figure(figsize=(12, 12), dpi=100)
    plt.imshow(rgb)
    plt.title("Sentinel-2 RGB Composite\n(Bands 4-3-2)", fontsize=11)
    # plt.axis('off')
    plt.savefig('Sentinel_2_RGB.png', bbox_inches='tight', dpi=300)
    plt.show()

# Load and preprocess data
sentinel_data = np.load('sentinel2_rochester.npy')
masked_data = np.ma.masked_where(sentinel_data == 0, sentinel_data)
plot_rgb_composite(masked_data)

def plot_4band_composite(masked_data):
    """
    Create a false-color composite using B3, B4, B5, B6
    - Band assignments: R=Band5, G=Band4, B=Band3
    - Band6 used for additional analysis (e.g., NDVI)
    """
    # Extract bands (0-based indices)
    band3 = masked_data[:, :, 2].astype(float)  # Green (560 nm)
    band4 = masked_data[:, :, 3].astype(float)  # Red (665 nm)
    band5 = masked_data[:, :, 4].astype(float)  # Vegetation Red Edge (705 nm)
    band6 = masked_data[:, :, 5].astype(float)  # Vegetation Red Edge (740 nm)

    # Stretch bands using percentiles
    def stretch(band):
        p_low, p_high = np.percentile(band[band > 0], (2, 98))
        return rescale_intensity(band, in_range=(p_low, p_high), out_range=(0, 1))
    
    # Create RGB composite (Band5=Red, Band4=Green, Band3=Blue)
    rgb = np.stack([stretch(band5), stretch(band4), stretch(band3)], axis=-1)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.title("False-Color Composite: B5(R), B4(G), B3(B)\n+ Band6 for Vegetation Analysis")
    # plt.axis('off')
    plt.show()

# Load data
sentinel_data = np.load('sentinel2_rochester.npy')
masked_data = np.ma.masked_where(sentinel_data == 0, sentinel_data)

# Generate composite
plot_4band_composite(masked_data)

if __name__ == "__main__":
    pass
