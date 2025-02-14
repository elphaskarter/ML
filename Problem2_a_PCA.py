# --correlation analysis--
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from Problem1_a_HSI_DATA import hsi_data
import contextlib
import io
import matplotlib

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from Problem1_a_HSI_DATA import hsi_data, bands_float

matplotlib.use(original_backend)
