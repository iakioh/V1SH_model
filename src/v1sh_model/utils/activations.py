import numpy as np


def g_x(x, T_x=1.0):
    """Activation function for pyramidal cells

    Parameters:
        x (np.ndarray): deviation from rersting state (input current), shape (...), values in R
        T_x (float): threshold parameter, default 1.0

    Returns:
        (np.ndarray): activation values, shape (...), values in [0, inf)
    """
    g_of_x = np.zeros_like(x)
    g_of_x[T_x <= x] = (x - T_x)[T_x <= x]
    g_of_x[x > T_x + 1] = 1.0
    return g_of_x


def g_y(y, L_y=1.2, g_1=0.21, g_2=2.5):
    """Activation function for interneurons

    Parameters:
        y (np.ndarray): deviation from resting state (input current), shape (...), values in R
        L_y (float): threshold parameter, default 1.2
        g_1 (float): slope parameter, default 0.21
        g_2 (float): slope parameter, default 2.5

    Returns:
        (np.ndarray): activation values, shape (...), values in [0, inf)
    """
    g_of_y = np.zeros_like(y)
    g_of_y[0 <= y] = g_1 * y[0 <= y]
    g_of_y[y >= L_y] = g_1 * L_y + g_2 * (y[y >= L_y] - L_y)
    return g_of_y
