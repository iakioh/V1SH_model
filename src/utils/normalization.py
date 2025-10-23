import numpy as np
from scipy.ndimage import convolve


def I_o(g_X: np.ndarray):
    """Computes normalization term of pyramidal cells

    Parameters:
        g_X (np.ndarray): Acitvated state of pyramidal cells, shape (N_y, N_x, K)

    Returns:
        (np.ndarray): normalization term, shape (N_y, N_x, 1), values in [0, inf]
    """

    g_X_summed_over_K = g_X.sum(axis=-1, keepdims=True)  # shape (N_y, N_x, 1)

    # neighbors on Manhatten Grid with distance maximal 2
    neighbors = np.ones((5, 5), dtype=g_X.dtype)
    # neighbors[2, 2] = 0  # neuron itself is included in neighborhood

    g_X_normalized = convolve(
        g_X_summed_over_K, neighbors[:, :, np.newaxis], mode="wrap"
    )  # shape (N_y, N_x, 1)

    return 0.85 - 2.0 * (g_X_normalized / neighbors.sum()) ** 2


def I_c(I_top_down=0.0):
    """Computes normalization term of interneurons

    Parameters:
        I_top_down (float): top-down input, default 0.0

    Returns:
        (float): normalization term, value in R
    """
    return 1.0 + I_top_down
