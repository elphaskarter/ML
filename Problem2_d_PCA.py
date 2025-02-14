# --Problem 2(b) pca analysis--
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from Problem1_a_HSI_DATA import hsi_data
from Problem2_a_b_PCA import SVD_PCA
import contextlib
import io
import matplotlib

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from Problem1_a_HSI_DATA import hsi_data, bands_float
    from Problem2_a_b_PCA import SVD_PCA

matplotlib.use(original_backend)

