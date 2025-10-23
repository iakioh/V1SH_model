import numpy as np


def tuning_curve(angle: np.ndarray) -> np.ndarray:
    """Tuning curve function

    Parameters:
        angle (np.ndarray): angle difference (radians), shape (N_y, N_x, ...), values in [-pi/2, +pi/2]

    Returns:
        (np.ndarray): tuning curve values, shape (N_y, N_x, ...), values in [0, 1]
    """
    absolute_angle = np.abs(angle)
    absolute_angle = np.minimum(
        absolute_angle, np.pi - absolute_angle
    )  # wrap to [0, pi/2]
    phi = np.exp(-absolute_angle / (np.pi / 8))
    phi[absolute_angle >= np.pi / 6] = 0
    return phi
