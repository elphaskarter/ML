"""
Author: Elphas Khata
Date: 2025-02-01
Description: Problem 1 script loads a Sentinel-2 dataset and 
plots each band with its corresponding central wavelength
"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean

def plot_bands(masked_data, cwv_data):
    for band in range(0, 12):
        band_array = masked_data[:, :, band]
        band_array_flipped = np.flipud(band_array)
        image = plt.pcolormesh(band_array_flipped, cmap=cmocean.cm.gray, shading='auto')
        cbar = plt.colorbar(image)
        cbar.set_label('Reflectance', fontsize=11)
        plt.axis('off')
        plt.title(f'Band CWV: {cwv_data[band]} nm')
        plt.show()
    return None

sentinel_data = np.load('sentinel2_rochester.npy')
cwv_data = [443, 490, 560, 665, 705, 740, 783, 842, 940, 1375, 1610, 2190]
masked_data = np.ma.masked_where(sentinel_data == 0, sentinel_data)
# plot_bands(masked_data, cwv_data)

# call main function 
if __name__ == "__main__":
    pass
