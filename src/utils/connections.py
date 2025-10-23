import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from src.inputs.visualize import plot_bars


def compute_psi(theta, K, atol=1e-6):
    """Computes distance metric for orientation angles used in V1 model

    Parameters:
        theta (np.ndarray): Angle differences (radians), shape (...), values in radians

    Returns:
        (np.ndarray): Distance metric, shape (...), values in [0, 1]
    """

    theta = np.abs(theta)
    theta = theta % np.pi  # wrap to [0, pi]
    theta = np.minimum(theta, np.pi - theta)  # wrap to [0, pi/2]

    psi = np.zeros_like(theta)

    where_theta_is_zero = np.isclose(theta, 0.0, atol=atol)
    psi[where_theta_is_zero] = 1

    where_theta_is_pi_over_K = np.isclose(theta, np.pi / K, atol=atol)
    psi[where_theta_is_pi_over_K] = 0.8

    where_theta_is_pi_over_2K = np.isclose(theta, 2 * np.pi / K, atol=atol)
    psi[where_theta_is_pi_over_2K] = 0.7

    return psi


def compute_W_values(d, beta, delta_theta, theta_1, theta_2):
    """Computes the entries of W according to pp. 314, "Understanding Vision" (Li Zhaoping, 2014)

    Parameters:
        d (np.ndarray): Distance between pre- and post-synaptic neurons, values in [0, inf)
        beta (np.ndarray): Angle between preferred orientation of pre-synaptic neuron and line connecting two neurons (radians), values in [0, pi]
        delta_theta (np.ndarray): Angle between preferred orientations of pre- and post-synaptic neurons (radians), values in [0, pi/2]
        theta_1 (np.ndarray): Smallest angle between preferred orientation of neuron and line connecting pre- and post-synaptic neurons (radians), values in [0, pi/2]
        theta_2 (np.ndarray): Largest angle between preferred orientation of neuron and line connecting pre- and post-synaptic neurons (radians), values in [0, pi/2]

    Returns:
        (np.ndarray): entry of W, values in R
    """
    if (
        (d > 0)
        and (d / np.cos(beta / 4)) < 10
        and (beta >= np.pi / 1.1)
        and (np.abs(theta_1) > np.pi / 11.999)
        and (delta_theta < np.pi / 3)
    ):
        # d > 0: only connected to neurons in other hypercolumns
        # d / np.cos(beta/4) < 10: elliptical shaped interaction circumfrence (elongated along line connecting two neurons)
        # delta_theta < np.pi / 3: only connections for iso-oriented neurons
        # beta >= np.pi / 1.1 and |theta_1| > np.pi / 11.999: only connections for bars not colinear, i.e. parallel but orthogonal to line connecting two neurons
        return (
            0.141
            * (1 - np.exp(-0.4 * (beta / d) ** 1.5))
            * np.exp(-((delta_theta / (np.pi / 4)) ** 1.5))
        )
    else:
        return 0.0


def compute_J_values(d, beta, delta_theta, theta_1, theta_2):
    """Computes the entries of J according to pp. 314, "Understanding Vision" (Li Zhaoping, 2014)

    Parameters:
        d (np.ndarray): Distance between pre- and post-synaptic neurons, values in [0, inf)
        beta (np.ndarray): Angle between preferred orientation of pre-synaptic neuron and line connecting two neurons (radians), values in [0, pi]
        delta_theta (np.ndarray): Angle between preferred orientations of pre- and post-synaptic neurons (radians), values in [0, pi/2]
        theta_1 (np.ndarray): Smallest angle between preferred orientation of neuron and line connecting pre- and post-synaptic neurons (radians), values in [0, pi/2]
        theta_2 (np.ndarray): Largest angle between preferred orientation of neuron and line connecting pre- and post-synaptic neurons (radians), values in [0, pi/2]

    Returns:
        (np.ndarray): entry of J, values in R
    """
    if (
        (d > 0)
        and (d <= 10)
        and (
            (beta < np.pi / 2.69)
            or ((beta < np.pi / 1.1) and (np.abs(theta_2) < np.pi / 5.9))
        )
    ):
        # d > 0: only connected to neurons in other hypercolumns
        # d <= 10: circular shaped interaction circumfrence
        # |theta_2| < np.pi / 5.9: only connections for prefered orientations close to line (since |theta_1| <= |theta_2|)
        # beta < np.pi / 2.69: introduces slight asymmetry such that more bars are connected if they are colinear (i.e. along a smooth contour)
        # beta < np.pi / 1.1: unnecessary since always true if |theta_1| <= |theta_2| < np.pi / 5.9
        return 0.126 * np.exp(-((beta / d) ** 2) - 2 * (beta / d) ** 7 - d**2 / 90)
    else:
        return 0.0


def compute_angle_between_bar_and_line(bar_angle, line_angle):
    """Computes the angle between a bar and a line, both defined by their angles (radians)

    Parameters:
        bar_angle (np.ndarray): Angle of the bar (radians), shape (...), values in [0, pi]
        line_angle (np.ndarray): Angle of the line (radians), shape (...), values in [0, pi]

    Returns:
        (np.ndarray): Angle between bar and line (radians), shape (...), values in [0, pi/2]
    """
    angle_diff = line_angle - bar_angle

    if angle_diff >= np.pi / 2:
        angle_diff -= np.pi

    elif angle_diff < -np.pi / 2:
        angle_diff += np.pi

    return angle_diff


def compute_connection_kernel(K=12, verbose=False) -> np.ndarray:
    """Computes intracortical connection kernels J, W and Psi, according to pp. 314, "Understanding Vision" (Li Zhaoping, 2014).
        Note: 3. dimension is post-synaptic, 4. dimension is pre-synaptic orientation channel

    Parameters:
        K (int): Number of orientation channels, default 12

    Returns:
        J (np.ndarray): 3D array of inter-hypercolumn excitatory connection kernel, shape (N_y, N_x, K, K),
        W (np.ndarray): 3D array of inter-hypercolumn inhibitory connection kernel, same shape as J
        Psi (np.ndarray): 2D array of intra-hypercolum connection kernel, shape (1, 1, K, K)

    """

    kernel_size = 10
    N_x, N_y = 2 * kernel_size + 1, 2 * kernel_size + 1
    J = np.zeros((N_y, N_x, K, K))
    W = np.zeros((N_y, N_x, K, K))
    Psi = np.zeros((1, 1, K, K))

    A = np.linspace(0, np.pi, K, endpoint=False, dtype=np.float64)  # shape (K,)
    x, y = (
        np.arange(-kernel_size, kernel_size + 1, 1, dtype=np.float64),
        np.arange(-kernel_size, kernel_size + 1, 1, dtype=np.float64),
    )
    Y, X = np.meshgrid(x, y, indexing="ij")  # shape (21, 21)

    # angle between y-axis and vector to neuron
    alphas = np.arctan2(X, Y) % np.pi  # shape (21, 21), values in [0, pi]

    if verbose:
        # for testing
        plt.imshow(X, cmap="viridis", origin="lower")
        plt.colorbar(label="X", fraction=0.046, pad=0.04)
        plt.title("X coordinate")
        plt.show()

        plt.imshow(Y, cmap="viridis", origin="lower")
        plt.colorbar(label="Y", fraction=0.046, pad=0.04)
        plt.title("Y coordinate")
        plt.show()

        plt.imshow(
            alphas / np.pi * 180,
            cmap="viridis",
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=0,
            vmax=180,
        )
        plt.colorbar(
            label="Angle [°]", fraction=0.046, pad=0.04, ticks=[0, 45, 90, 135, 180]
        )
        plt.title("Angle between y-axis and vector to neuron")
        plt.show()

    for k_pre in range(K):
        for k_post in range(K):
            a = np.abs(A[k_post] - A[k_pre]) % np.pi
            delta_theta = np.minimum(a, np.pi - a)

            # compute non-zero entries of Psi
            Psi[0, 0, k_post, k_pre] = compute_psi(delta_theta, K)
            if k_post == k_pre:
                Psi[0, 0, k_post, k_pre] = 0.0  # no self-connection

            for i in range(0, N_y):
                for j in range(0, N_x):
                    d = np.sqrt(X[i, j] ** 2 + Y[i, j] ** 2)
                    if d > 0:
                        # angle between y-axis and vector to neuron
                        alpha = alphas[i, j]

                        # angle between preferred orientation
                        # of centered neuron and vector to neuron
                        theta_1_dash = compute_angle_between_bar_and_line(
                            A[k_pre], alpha
                        )
                        theta_2_dash = compute_angle_between_bar_and_line(
                            A[k_post], alpha
                        )

                        # name theta_1 and theta_2 correctly
                        if np.abs(theta_1_dash) < np.abs(theta_2_dash):
                            theta_1 = theta_1_dash
                            theta_2 = theta_2_dash
                        else:
                            theta_1 = theta_2_dash
                            theta_2 = theta_1_dash

                        beta = 2 * np.abs(theta_1) + 2 * np.sin(
                            np.abs(theta_1 + theta_2)
                        )

                        # compute non-zero entries of J
                        J[i, j, k_post, k_pre] = compute_J_values(
                            d, beta, delta_theta, theta_1, theta_2
                        )

                        # compute non-zero entries of W
                        W[i, j, k_post, k_pre] = compute_W_values(
                            d, beta, delta_theta, theta_1, theta_2
                        )

    return J, W, Psi


def visualize_weights(
    W, J, Psi, k_pres=[0, 6], K=12, dpi=200, verbose=False, colored=True
):
    N_y, N_x = W.shape[0], W.shape[1]
    A = np.linspace(0, np.pi, K, endpoint=False)
    A = A[np.newaxis, np.newaxis, :]  # shape (1, 1, K)
    A = np.broadcast_to(A, (N_y, N_x, K))  # shape (N_y, N_x, K)

    plt.rcParams.update({"font.size": 8})

    for k_pre in k_pres:  # preferred orientation of presynaptic neuron
        if colored:
            fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True, dpi=dpi)
            blue_line = mlines.Line2D([], [], color="tab:blue", label=r"$J$")
            plot_bars(
                A,
                J[:, :, :, k_pre] * 7.5,
                verbose=False,
                dpi=dpi,
                axis=axis,
                color="tab:blue",
            )

            red_line = mlines.Line2D([], [], color="tab:red", label=r"$W$")
            plot_bars(
                A,
                W[:, :, :, k_pre] * 7.5,
                verbose=False,
                dpi=dpi,
                color="tab:red",
                axis=axis,
            )

            green_line = mlines.Line2D([], [], color="tab:green", label=r"$\psi$")
            Psi_broadcasted = np.zeros_like(W)
            Psi_broadcasted[W.shape[0] // 2, W.shape[1] // 2, :, :] = Psi[0, 0, :, :]
            plot_bars(
                A,
                Psi_broadcasted[:, :, :, k_pre],
                verbose=False,
                dpi=dpi,
                color="tab:green",
                axis=axis,
            )

            center_bar = np.zeros((N_y, N_x, K))
            center_bar[10, 10, k_pre] = 1
            black_line = mlines.Line2D([], [], color="k", label="presynaptic neuron")
            plot_bars(A, center_bar, verbose=False, dpi=dpi, color="k", axis=axis)

            plt.legend(
                handles=[blue_line, red_line, green_line, black_line],
                loc="upper right",
                framealpha=1.0,
            )

            return fig
        else:  # replicate fig. 4 from A Neural Model of Contour Integration, Zhaoping Li, 1998
            center_bar = np.zeros((N_y, N_x, K))
            center_bar[10, 10, k_pre] = 4

            fig_1, axis = plt.subplots(
                figsize=(12, 5), constrained_layout=True, dpi=dpi
            )
            plot_bars(
                A,
                (J[:, :, :, k_pre] > 0) * 0.5,
                verbose=False,
                dpi=dpi,
                axis=axis,
                color="k",
            )
            plot_bars(A, center_bar, verbose=False, dpi=dpi, color="k", axis=axis)
            plt.title("Horizontal connections J\nto excitatory post-synaptic cells")

            fig_2, axis = plt.subplots(
                figsize=(12, 5), constrained_layout=True, dpi=dpi
            )
            plot_bars(
                A,
                (W[:, :, :, k_pre] > 0) * 0.5,
                verbose=False,
                dpi=dpi,
                axis=axis,
                color="k",
            )
            plot_bars(A, center_bar, verbose=False, dpi=dpi, color="k", axis=axis)
            plt.title("Horizontal connections W\nto inhibitory post-synaptic cells")

            return fig_1, fig_2
