# --correlation analysis--
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import cmocean

from Problem1_a_HSI_DATA import hsi_data, bands_float

# Reshape 3D data to 2D (pixels x bands)
rows, cols, bands = hsi_data.shape
hsi_2d = hsi_data.reshape(-1, bands)

# Create DataFrame with band indices
hsiData_df = pd.DataFrame(hsi_2d, columns=[f'Band_{i+1}' for i in range(bands)])
corr_matrix = hsiData_df.corr(method='pearson') # Calculate correlation matrix

high_corr = np.where(corr_matrix > 0.9)
print(f"Redundant bands: {list(zip(high_corr[0], high_corr[1]))}")

low_corr = np.where((corr_matrix < 0.3) & (corr_matrix != 1))
print(f"Unique information bands: {list(zip(low_corr[0], low_corr[1]))}")

# Plot heatmap
plt.ion()
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, cmap=cmocean.cm.dense, shading='auto', vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.75})
plt.title('HSI Band Correlation Matrix', fontsize=14)
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()
plt.ioff()