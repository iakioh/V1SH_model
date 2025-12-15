import numpy as np
from scipy.ndimage import convolve
from typing import Optional

from v1sh_model.utils.activations import g_x


def I_o(X: Optional[np.ndarray] = None, original_parametrization: bool = True) -> np.ndarray:
    """Computes normalization term of pyramidal cells

    Parameters:
        X (np.ndarray): state of pyramidal cells, shape (N_y, N_x, K)

    Returns:
        (np.ndarray): normalization term, shape (N_y, N_x, 1), values in [0, inf]
    """
    if X is not None:
        g_X = g_x(X)
        g_X_summed_over_K = g_X.sum(axis=-1, keepdims=True)  # shape (N_y, N_x, 1)

        # neighbors on Manhatten Grid with distance maximal 2
        # neuron itself is included in neighborhood
        neighbors = np.ones((5, 5), dtype=g_X.dtype)

        g_X_normalized = convolve(
            g_X_summed_over_K, neighbors[:, :, np.newaxis], mode="wrap"
        )  # shape (N_y, N_x, 1)
        
        if original_parametrization:
            return 0.85 - 2.0 * (g_X_normalized / 16) ** 2 # normalization actually implemented as "16" instead of "25"
        else:
            return 0.8 - 2.0 * (g_X_normalized / neighbors.sum()) ** 2 # as documented in original paper
    else:
        return 0.85


def I_c(I_top_down=0.0):
    """Computes normalization term of interneurons

    Parameters:
        I_top_down (float): top-down input, default 0.0

    Returns:
        (float): normalization term, value in R
    """
    return 1.0 + I_top_down
