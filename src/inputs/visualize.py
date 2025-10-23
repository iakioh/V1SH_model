import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_bars(
    A: np.ndarray,
    W: np.ndarray,
    l: float = 9,
    r: float = 1.3,
    verbose: bool = True,
    dpi: int = 500,
    axis: Optional[plt.Axes] = None,
    color: str = "k",
) -> plt.Figure:
    """
    Plots a Manhatten grid of bars with given angles A and widths W.

    Parameters:
        A (np.ndarray): array of angles (radians), shape (N_y, N_x, K) or (N_y, N_x). 1. and 2. dimension = indicate y- and x-coordinate of bar. Last dimension allows plotting multiple bars at the same location.
        W (np.ndarray or None): array of bar widths, same shape as A
        l (float): Bar length
        r (float): Grid spacing factor
        verbose (bool): If True, show the plot
        dpi (int): Dots per inch for rendering
        axis (matplotlib axis or None): If provided, plot on this axis instead of creating a new figure.
        color (str): Color of the bars; default: black

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure object
    """
    assert W.shape == A.shape, "W must have the same shape as A"
    if A.ndim == 2:
        A = A[:, :, np.newaxis]
        W = W[:, :, np.newaxis]
    N_y, N_x, K = A.shape

    # Calculate image size in pixels
    d = l * r  # grid spacing
    img_height = int(N_y * d)
    img_width = int(N_x * d)

    # Create figure
    if axis is not None:
        ax = axis
        fig = None
    else:
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=dpi)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_aspect("equal")  # keep x and y scales the same, avoding distortion
    ax.axis("off")

    # Draw bars
    for i in range(N_y):
        for j in range(N_x):
            for k in range(K):
                # compute center of the bar
                cx = (j + 0.5) * d  # center x-coordinate
                cy = (i + 0.5) * d  # invert y-axis for plotting
                # compute bar directions
                angle = A[i, j, k]
                dx = l * np.sin(angle) / 2
                dy = l * np.cos(angle) / 2
                # compute endpoints of the bar
                x0, y0 = cx - dx, cy - dy
                x1, y1 = cx + dx, cy + dy
                # draw the bar
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    linewidth=W[i, j, k],
                    solid_capstyle="butt",
                )

    if verbose:
        plt.show()

    return fig


def visualize_input(
    A: np.ndarray,
    C: np.ndarray,
    scale: float = 3,
    l: float = 9,
    r: float = 1.3,
    verbose: bool = True,
    dpi: int = 500,
    axis: Optional[plt.Axes] = None,
) -> None:
    """
    Visualizes the input angles A and contrasts C as a grid of bars.

    Parameters:
        A (np.ndarray): array of angles (radians), shape (N_y, N_x, K) or (N_y, N_x). 1. and 2. dimension = indicate y- and x-coordinate of bar. Last dimension allows plotting multiple bars at the same location.
        C (np.ndarray): array of bar widths, same shape as A, values in [1, 4]
        scale (float): Scaling factor for bar widths
        l (float): Bar length
        r (float): Grid spacing factor
        verbose (bool): If True, show the plot
        dpi (int): Dots per inch for rendering
        axis (matplotlib axis or None): If provided, plot on this axis instead of creating a new figure.

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure object
    """
    assert np.all((C >= 1) & (C <= 4) | (C == 0)), "C values must 0 or in [1, 4]"
    W = C / scale
    return plot_bars(A, W, l=l, r=r, verbose=verbose, dpi=dpi, axis=axis)


def visualize_output(A, S, scale=1, l=9, r=1.3, verbose=True, dpi=500, axis=None):
    """
    Visualizes the output saliency S as a grid of bars with uniform orientation.

    Parameters:
        A (np.ndarray): array of angles (radians), shape (N_y, N_x, K) or (N_y, N_x). 1. and 2. dimension = indicate y- and x-coordinate of bar.Last dimension allows plotting multiple bars at the same location.
        S (np.ndarray): array of saliency values, same shape as A
        scale (float): Scaling factor for bar widths
        l (float): Bar length
        r (float): Grid spacing factor
        verbose (bool): If True, show the plot
        dpi (int): Dots per inch for rendering

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure object
    """
    assert np.all(S >= 0), "S values must be non-negative"
    W = S / scale
    return plot_bars(A, W, l=l, r=r, verbose=verbose, dpi=dpi, axis=axis)
