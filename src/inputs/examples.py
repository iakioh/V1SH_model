import numpy as np


def bar_without_surround(N_y=9, N_x=9):
    C = np.zeros((N_y, N_x))
    C[int((N_y - 1) / 2), int((N_x - 1) / 2)] = 3.5
    A = np.zeros((N_y, N_x))
    return A, C


def iso_orientation(N_y=9, N_x=9):
    C = np.full((N_y, N_x), 3.5)
    A = np.zeros((N_y, N_x))
    return A, C


def random_background(N_y=9, N_x=9, seed=None):
    C = np.full((N_y, N_x), 3.5)
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, np.pi, (N_y, N_x))
    A[int((N_y - 1) / 2), int((N_x - 1) / 2)] = 0.0
    return A, C


def cross_orientation(N_y=9, N_x=9):
    C = np.full((N_y, N_x), 3.5)
    A = np.full((N_y, N_x), np.pi / 2)
    A[int((N_y - 1) / 2), int((N_x - 1) / 2)] = 0
    return A, C


def bar_without_surround_low_contrast(N_y=9, N_x=9):
    C = np.zeros((N_y, N_x))
    C[int((N_y - 1) / 2), int((N_x - 1) / 2)] = 1.05
    A = np.zeros((N_y, N_x))
    return A, C


def with_one_flanker(N_y=9, N_x=9):
    C = np.zeros((N_y, N_x))
    C[int((N_y - 1) / 2), int((N_x - 1) / 2)] = 1.05
    C[int((N_y - 1) / 2) + 1, int((N_x - 1) / 2)] = 3.5
    A = np.zeros((N_y, N_x))
    return A, C


def with_two_flankers(N_y=9, N_x=9):
    C = np.zeros((N_y, N_x))
    y_mid, x_mid = int((N_y - 1) / 2), int((N_x - 1) / 2)
    C[y_mid, x_mid] = 1.5
    C[y_mid - 1, x_mid] = 3.5
    C[y_mid + 1, x_mid] = 3.5
    A = np.zeros((N_y, N_x))
    return A, C


def with_flanking_line_and_noise(N_y=9, N_x=9, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, np.pi, (N_y, N_x))
    y_mid, x_mid = int((N_y - 1) / 2), int((N_x - 1) / 2)
    A[:, x_mid] = 0.0
    C = np.full((N_y, N_x), 3.5)
    C[y_mid, x_mid] = 1.5
    return A, C


def neighboring_textures(n_rows=11, n_cols=27, I_input=2.0):
    A = np.zeros((n_rows, n_cols))
    A[:, : n_cols // 2] = np.pi / 2
    C = np.full((n_rows, n_cols), I_input)
    return A, C


def conjuction_features():
    A = np.zeros((11, 27, 2))
    A[:, :14, 1] = np.pi / 2 * 0.5
    C = np.zeros((11, 27, 2))
    C[:, :14, :] = 1.0
    C[:, 14:, 0] = 1.0
    return A, C
