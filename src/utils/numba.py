from numba import njit, prange
import numpy as np


@njit(parallel=True)
def convolve_numba(S_padded, kernel, N_y, N_x, K, kernel_size):
    """ Computes the sum over the connection kerne, i.e. (inverse) convolution of the kernel over S_padded across the first two spatial dimension

    Args:
        S_padded (np.ndarray): 3D array of shape (N_y, N_x, K)
        kernel (np.ndarray): 4D array of shape (N_y_k, N_x_k, K, K)
        N_y (int): number of rows of S_padded
        N_x (int): number of columns of S_padded
        K (int): number of channels of S_padded
        kernel_size (int): size of kernel

    Returns:
        result (np.ndarray): 3D array of shape (N_y, N_x, K)
    """
    
    result = np.zeros((N_y, N_x, K))
    for k_post in prange(K):
        for i in range(N_y):
            for j in range(N_x):
                acc = 0.0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        for k_pre in range(K):
                            acc += (
                                S_padded[i + ki, j + kj, k_pre]
                                * kernel[ki, kj, k_post, k_pre]
                            )
                result[i, j, k_post] = acc
    return result
