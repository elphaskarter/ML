# --Problem 1(a)---

import numpy as np
import matplotlib.pyplot as plt
import spectral as spy
from collections import defaultdict
from mayavi import mlab

# Load the hyperspectral data
hdr_file = r"C:\Users\elpha\OneDrive\Desktop\hsi_data\tait_hsi.hdr"
hsi_image = spy.open_image(hdr_file)
hsi_data = hsi_image.load()

bands = hsi_image.metadata.get("band names", [])
bands_float = []
for band in bands:
    cleaned_band = band.strip().replace(' ms', '').replace('nm', '').replace(',', '')
    try:
        bands_float.append(float(cleaned_band))
    except ValueError:
        bands_float.append(0.0)  # Placeholder for debugging

# Electromagnetic spectrum families (in nanometers)
em_families = {
    "UV": (10, 400),
    "Optical": (400, 700),
    "Infrared": (700, 1000000)  # 1 mm in nanometers
}

# Optical sub-categories
visible_categories = {
    "Blue": (450, 495),
    "Green": (495, 570),
    "Red": (620, 700)
}

# Classify bands into EM families
band_classes = {}
for i, wavelength in enumerate(bands_float):
    for family, (start, end) in em_families.items():
        if start <= wavelength < end:
            if family == "Optical":
                for subcategory, (sub_start, sub_end) in visible_categories.items():
                    if sub_start <= wavelength < sub_end:
                        band_classes[i] = subcategory
                        break
                else:
                    band_classes[i] = "Optical (Other)"
            else:
                band_classes[i] = family
            break

# Count bands in each family
family_counts = defaultdict(int)
for family in band_classes.values():
    family_counts[family] += 1
print("Bands per EM family:", dict(family_counts))

blue_band = next((i for i, fam in band_classes.items() if fam == "Blue"), None)
green_band = next((i for i, fam in band_classes.items() if fam == "Green"), None)
red_band = next((i for i, fam in band_classes.items() if fam == "Red"), None)
nir_band = next(i for i, fam in band_classes.items() if fam == "Infrared")

# Plot one band in blue, green and red regions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Blue Band
if blue_band is not None:
    axes[0].imshow(hsi_data[:, :, blue_band], cmap='gray')
    axes[0].set_title(f"Blue Band {blue_band} ({bands_float[blue_band]:.2f} nm)")
    axes[0].axis("off")

# Plot Green Band
if green_band is not None:
    axes[1].imshow(hsi_data[:, :, green_band], cmap='gray')
    axes[1].set_title(f"Green Band {green_band} ({bands_float[green_band]:.2f} nm)")
    axes[1].axis("off")

# Plot Red Band
if red_band is not None:
    axes[2].imshow(hsi_data[:, :, red_band], cmap='gray')
    axes[2].set_title(f"Red Band {red_band} ({bands_float[red_band]:.2f} nm)")
    axes[2].axis("off")

plt.tight_layout()
plt.show()

# pseudo plot
pseudo_image = spy.get_rgb(hsi_data, bands=(green_band, red_band, nir_band))
plt.figure(figsize=(6, 4))
plt.imshow(pseudo_image)
plt.axis("off")
plt.title(f"Pseudocolor Composite (Bands {green_band}, {red_band}, {nir_band})", fontsize=8)
plt.savefig('Pseudocolor_Composite.png', dpi=300)
plt.show()

if __name__ == "__main__":
    pass