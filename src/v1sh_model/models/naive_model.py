import numpy as np

from v1sh_model.inputs.visualize import visualize_input, visualize_output
from v1sh_model.utils.tuning_curve import tuning_curve

class NaiveModel:
    def __init__(self, K=12, alpha=1.0):
        self.alpha = alpha
        self.K = K

        # Precompute preferred orientations per neuron
        angles = np.linspace(0, np.pi, self.K, endpoint=False)
        self.M = angles[np.newaxis, np.newaxis, :]  # shape (1, 1, K)

    def get_input(
        self, A: np.ndarray, C: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """Computes model input from visual input

        TODO: extend to multiple input bars per locationn (i.e. A and C of shape (N_y, N_x, L) where L is number of input bars per location)

        Parameters:
            A (np.ndarray): 2D array of angles (radians) of input bars, shape (N_y, N_x), values in [0, pi]
            C (np.ndarray): 2D array of contrasts of input bars, same shape as A, values in [1, 4] or 0 (no bar)
            verbose (bool): If True, visualize input and output

        Returns:
            I (np.ndarray): 3D array of model input, shape (N_y, N_x, K)

        """
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

    def derivative(self, X: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Computes the derivative dX/dt

        Parameters:
            X (np.ndarray): Current state, shape (N_y, N_x, K)
            I (np.ndarray): Input, shape (N_y, N_x, K)

        Returns:
            (np.ndarray): Derivative dX/dt, shape (N_y, N_x, K)
        """

        return -self.alpha * X + I

    def euler_method(self, I: np.ndarray, dt: float, T: float) -> np.ndarray:
        """Simulates the model over time given input I

        Parameters:
            I (np.ndarray): Input, shape (N_y, N_x, K)
            dt (float): Time step
            T (float): Total simulation time

        Returns:
            X (np.ndarray): Final state after simulation, shape (T, N_y, N_x, K)
        """

        steps = int(T / dt)
        X = np.zeros((steps, *I.shape))

        X[0] = np.zeros_like(I)  # initial state = input
        for t in range(1, steps):
            dXdt = self.derivative(X[t - 1], I)
            X[t] = X[t - 1] + dt * dXdt

        return X

    def simulate(
        self,
        A: np.ndarray,
        C: np.ndarray,
        dt: float = 0.001,
        T: float = 12.0,
        verbose: bool = False,
    ) -> np.ndarray:
        """Runs the full simulation given angles A and contrasts C

        Parameters:
            A (np.ndarray): 2D array of angles (radians) of input bars, shape (N_y, N_x), values in [0, pi]
            C (np.ndarray): 2D array of contrasts of input bars, same shape as A, values in [1, 4] or 0 (no bar)
            dt (float): Time step
            T (float): Total simulation time

        Returns:
            X (np.ndarray): Final state after simulation, shape (T, N_y, N_x, K)
        """

        I = self.get_input(A, C, verbose=verbose)
        X = self.euler_method(I, dt, T)
        return X, I
        

def analytical_solution(t: np.ndarray, I: np.ndarray, alpha: float) -> np.ndarray:
    """Computes the analytical solution of the ODE at time t given input I and parameter alpha.
        Assumes constant input, and initial condition X(t = 0) = 0.

    Parameters:
        t (np.ndarray): Time points, shape (N_t,)
        I (np.ndarray): Input, shape (N_y, N_x, K)
        alpha (float): Model parameter

    Returns:
        (np.ndarray): Analytical solution at time t, shape (N_y, N_x, K)
    """
    t_ = t[:, np.newaxis, np.newaxis, np.newaxis]  # shape (N_t, 1, 1, 1)
    I_ = I[np.newaxis, :, :, :]  # shape (1, N_y, N_x, K)
    return (1 - np.exp(-alpha * t_)) / alpha * I_