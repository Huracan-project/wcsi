import matplotlib as mpl
import numpy as np


try:
    COLORS = mpl.colormaps["okabe_ito"](np.linspace(0, 1, 8))
except KeyError:
    COLORS = mpl.colormaps["tab10"](np.linspace(0, 1, 8))
