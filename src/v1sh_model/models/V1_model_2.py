from numba import njit, prange
import numpy as np
from tqdm import tqdm

from v1sh_model.utils.connections import compute_connection_kernel
from v1sh_model.utils.activations import g_x, g_y
from v1sh_model.utils.normalization import I_o, I_c
from v1sh_model.utils.tuning_curve import tuning_curve
from v1sh_model.inputs.visualize import visualize_input, visualize_output

@njit(parallel=True)
def summation_numba(
    X: np.ndarray,
    Y: np.ndarray,
    g_X: np.ndarray,
    g_Y: np.ndarray,
    g_X_padded: np.ndarray,
    I: np.ndarray,
    I_o: np.ndarray,
    I_c: float,
    Psi: np.ndarray,
    J: np.ndarray,
    W: np.ndarray,
    J_o: float,
    alpha_x: float,
    alpha_y: float,
):  
    N_y, N_x, K = X.shape
    kernel_size = W.shape[0] # odd, square,same for J
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    
    dXdt = np.zeros((N_y, N_x, K), dtype=np.float64)
    dYdt = np.zeros((N_y, N_x, K), dtype=np.float64)

    for n_y in prange(N_y):
        for n_x in prange(N_x):
            for k_post in prange(K):
                # add non-interactiove terms
                dxdt = (
                    - alpha_x * X[n_y, n_x, k_post] 
                    - g_Y[n_y, n_x, k_post]
                    + J_o * g_X[n_y, n_x, k_post]
                    + I[n_y, n_x, k_post]
                    + I_o[n_y, n_x, 0] # orientation unspecific, so same for all k_post
                )

                dydt = (
                    - alpha_y * Y[n_y, n_x, k_post]
                    + g_X[n_y, n_x, k_post]
                    + I_c
                )

                # accumulatively add interactive terms
                acc_dxdt = 0.
                acc_dydt = 0.
                for k_pre in range(K):
                    # add within hypercolumn interaction
                    if k_pre != k_post:
                        acc_dxdt -= (
                            Psi[0, 0, k_post, k_pre] * g_Y[n_y, n_x, k_pre] # Psi same for all locations
                        )
                    for y_pre in range(kernel_size):
                        for x_pre in range(kernel_size):
                            # add between hypercolumn interaction
                            acc_dxdt += (
                                J[y_pre, x_pre, k_post, k_pre]
                                * g_X_padded[n_y + y_pre, n_x + x_pre, k_pre]
                            )
                            acc_dydt += (
                                W[y_pre, x_pre, k_post, k_pre]
                                * g_X_padded[n_y + y_pre, n_x + x_pre, k_pre]
                            )

                dXdt[n_y, n_x, k_post] = dxdt + acc_dxdt
                dYdt[n_y, n_x, k_post] = dydt + acc_dydt

    return dXdt, dYdt

class V1_model_2:
    def __init__(
        self,
        K=12,
        alpha_x=1.0,
        alpha_y=1.0,
        average_noise_height=0.1,
        average_noise_temporal_width=0.1,
        seed=None,
    ):
        """Initializes the full V1 model with pyramidal cells and interneurons

        Parameters:
            K (int): Number of orientation channels, default 12
            alpha_x (float): Time constant of pyramidal cells, default 1.0
            alpha_y (float): Time constant of interneurons, default 1.0
            average_noise_height (float): Standard deviation of noise amplitude, default 0.1
            average_noise_temporal_width (float): Average temporal width of noise, default 0.1
            seed (int or None): Random seed for noise generation, default None
        """
        # Membrane time constants
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        # Precompute preferred orientations per neuron
        self.K = K
        angles = np.linspace(0, np.pi, self.K, endpoint=False, dtype=np.float64)
        self.M = angles[np.newaxis, np.newaxis, :]  # shape (1, 1, K)

        # Connection kernels
        self.J, self.W, self.Psi = compute_connection_kernel(K=K, verbose=False)
        kernel_size = self.W.shape[0]
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert (self.W.shape[0] == self.W.shape[1]), "Kernels must be square"
        assert (self.J.shape[0] == self.W.shape[0]) and (self.J.shape[1] == self.W.shape[1]), "J and W must have the same shape"
        self.J_o = 0.8

        # Normalization terms
        self.I_o = I_o
        self.I_c = I_c

        # Activation functions
        self.g_x = g_x
        self.g_y = g_y

        # Noise parameters
        assert average_noise_height >= 0, "average_noise_height must be non-negative"
        assert average_noise_temporal_width > 0, (
            "average_noise_temporal_width must be positive"
        )
        self.noise_std = average_noise_height
        self.noise_tau = average_noise_temporal_width
        self.rng = np.random.default_rng(seed)

    def get_input(
        self, A: np.ndarray, C: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """Computes model input from visual input

        TODO: extend to multiple input bars per location (i.e. A and C of shape (N_y, N_x, L) where L is number of input bars per location)

        Parameters:
            A (np.ndarray): 2D array of angles (radians) of input bars, shape (N_y, N_x), values in [0, pi]
            C (np.ndarray): 2D array of contrasts of input bars, same shape as A, values in [1, 4] or 0 (no bar)
            verbose (bool): If True, visualize input and output

        Returns:
            I (np.ndarray): 3D array of model input, shape (N_y, N_x, K)

        """
        assert A.ndim == 2, "A must be a 2D array"
        assert C.shape == A.shape, "C must have the same shape as A"
        N_y, N_x = A.shape
        M = np.broadcast_to(self.M, (N_y, N_x, self.K))
        M = M % np.pi  # ensure M in [0, pi]

        A = A % np.pi  # ensure A in [0, pi]
        A = A[:, :, np.newaxis]  # shape (N_y, N_x, 1)
        C = C[:, :, np.newaxis]  # shape (N_y, N_x, 1)

        I = C * tuning_curve(A - M)

        if verbose:
            visualize_input(A, C, verbose=True)
            for k in range(I.shape[2]):
                print("==================================")
                print(f"Neurons {k}, tuned to {M[0, 0, k] / np.pi * 180}°")
                visualize_output(M[:, :, k], I[:, :, k], verbose=True)

        return I

    def update_noise(
        self, I_noise: np.ndarray, noise_duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Defines noise distributions and updates the noise input I_noise based on the remaining noise duration

        Parameters:
            I_noise (np.ndarray): Current noise input, shape (N_y, N_x, K)
            noise_duration (np.ndarray): Remaining duration of current noise input, same shape as I_noise

        Returns:
            I_noise (np.ndarray): Updated noise input, same shape as input
            noise_duration (np.ndarray): Updated remaining duration of current noise input, same shape as I _noise
        """

        # amplitude follows normal distribution
        I_noise[noise_duration <= 0] = self.rng.normal(
            0., self.noise_std, size=I_noise.shape
        )[noise_duration <= 0]

        # temporal width follows exponential distribution,
        noise_duration[noise_duration <= 0] = self.rng.exponential(
            self.noise_tau, size=noise_duration.shape
        )[noise_duration <= 0]

        return I_noise, noise_duration

    def derivative(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        I: np.ndarray,
        I_top_down: float = 0.0,
        mode: str = "symmetric",
    ) -> np.ndarray:
        """Computes the derivative dX/dt and dY/dt of the model at current state X and Y given input I

        Parameters:
            X (np.ndarray): Current state of pyramidal cells, shape (N_y, N_x, K)
            Y (np.ndarray): Current state of interneurons, shape (N_y, N_x, K)
            I (np.ndarray): Input, shape (N_y, N_x, K)

        Returns:
            dXdt (np.ndarray): Derivative of pyramidal cells state, same shape as X
            dYdt (np.ndarray): Derivative of interneurons state, same shape as Y

        """
        # precompute activatied states
        g_X = self.g_x(X)
        g_Y = self.g_y(Y)

        # precompute padded term 
        kernel_size = self.W.shape[0] # assert square kernel
        delta_pad = (kernel_size - 1) // 2 # assert odd kernel size
        g_X_padded = np.pad(g_X, ((delta_pad, delta_pad), (delta_pad, delta_pad), (0, 0)), mode=mode)

        # pre-compute normalization terms 
        I_o_values = self.I_o(X)
        I_c_values = self.I_c(I_top_down)

        # compute derivative fast
        dXdt, dYdt = summation_numba(
            X,
            Y,
            g_X,
            g_Y,
            g_X_padded,
            I,
            I_o_values,
            I_c_values,
            self.Psi,
            self.J,
            self.W,
            self.J_o,
            self.alpha_x,
            self.alpha_y,
        )

        return dXdt, dYdt

    def euler_method(
        self,
        I: np.ndarray,
        dt: float,
        T: float,
        noisy: bool = True,
        mode: str = "symmetric",
    ) -> np.ndarray:
        """Simulates the model over time given input I

        Parameters:
            I (np.ndarray): Input, shape (N_y, N_x, K)
            dt (float): Time step
            T (float): Total simulation time

        Returns:
            X (np.ndarray): Final pyramidal state after simulation, shape (T, N_y, N_x, K)
        """
        N_y, N_x, K = I.shape
        assert K == self.K, (
            "Input and model must have the same number of orientation channels"
        )
        steps = int(T / dt)  # + 1 for initial condition

        # Pyramidal cells state over time
        X = np.zeros((steps, N_y, N_x, K), dtype=np.float64)

        # Interneuon state over time
        Y = np.zeros((steps, N_y, N_x, K), dtype=np.float64)

        # Noise initialization
        if noisy:
            I_noise = np.zeros(
                (N_y, N_x, K, 2), dtype=np.float64
            )  # last dim: 0: noise for X, 1: noise for Y
            noise_duration = np.zeros((N_y, N_x, K, 2), dtype=np.float64)

            # initial state = random
            I_noise, noise_duration = self.update_noise(I_noise, noise_duration)
            X[0] += I_noise[..., 0] * dt
            Y[0] += I_noise[..., 1] * dt

        # Time integration using Euler method
        update_steps = int(0.05 / dt)
        with tqdm(total=steps, desc="Simulating", unit="step") as pbar:
            for t in range(1, steps):
                # start_time_step = time.time()
                dXdt, dYdt = self.derivative(X[t - 1], Y[t - 1], I, mode=mode)
                X[t] = X[t - 1] + dt * dXdt
                Y[t] = Y[t - 1] + dt * dYdt

                if noisy:
                    # add noise
                    noise_duration -= dt
                    I_noise, noise_duration = self.update_noise(I_noise, noise_duration)
                    X[t] += I_noise[..., 0] * dt
                    Y[t] += I_noise[..., 1] * dt

                # Update progress bar every 0.05 seconds of simulated time
                if (t >= 1) and ((t - 1) % update_steps == 0):
                    pbar.update(update_steps)

                # end_time_step = time.time()
                # print(f"Time step {t}/{steps} computation time: {end_time_step - start_time_step:.4f} seconds")

        return X, Y

    def simulate(
        self,
        A: np.ndarray,
        C: np.ndarray,
        dt: float = 0.001,
        T: float = 12.0,
        verbose: bool = False,
        noisy: bool = True,
        mode: str = "symmetric",
    ) -> np.ndarray:
        """Runs the full simulation given angles A and contrasts C

        Parameters:
            A (np.ndarray): 2D array of angles (radians) of input bars, shape (N_y, N_x), values in [0, pi]
            C (np.ndarray): 2D array of contrasts of input bars, same shape as A, values in [1, 4] or 0 (no bar)
            dt (float): Time step
            T (float): Total simulation time
            verbose (bool): If True, visualize input; default False
            noisy (bool): If True, add noise to the simulation; default True
            mode (str): boundary condition of simulation (see np.pad); default "symmetric"

        Returns:
            X (np.ndarray): Final state after simulation, shape (T, N_y, N_x, K)
        """
        I = self.get_input(A, C, verbose=verbose)
        X, Y = self.euler_method(I, dt, T, noisy=noisy, mode=mode)
        return X, Y, I
