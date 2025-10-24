import numpy as np

from v1sh_model.inputs.examples import neighboring_textures
from v1sh_model.models.V1_model_2 import V1_model_2


def test_V1_model():
    # Instantiate and test the FullModel
    seed = 42
    model = V1_model_2(seed=seed, alpha_x=1.0, alpha_y=1.0)
    A, C = neighboring_textures(22, 60, 2.0)

    T = 2.0
    dt = 0.001
    X_gen, Y_gen, I = model.simulate(
        A, C, dt=dt, T=T, verbose=False, noisy=True, mode="wrap"
    )

    assert X_gen.shape == (T / dt, A.shape[0], A.shape[1], model.K)
    assert Y_gen.shape == (T / dt, A.shape[0], A.shape[1], model.K)
    assert I.shape == (A.shape[0], A.shape[1], model.K)


def test_1(N_y=20, N_x=20, K=12, seed=None):
    model = V1_model_2(seed=seed, alpha_x=1.0, alpha_y=1.0)

    X, Y = np.zeros((N_y, N_x, K)), np.zeros((N_y, N_x, K))
    rng = np.random.default_rng(seed)
    I = rng.random((N_y, N_x, K))

    dXdt, dYdt = model.derivative(X, Y, I)

    assert dXdt.shape == (N_y, N_x, K)
    assert dYdt.shape == (N_y, N_x, K)

    np.testing.assert_allclose(dXdt, I + 0.85)
    np.testing.assert_allclose(dYdt, 1.0)


def test_2(N_y=40, N_x=40, K=12, seed=None):
    model = V1_model_2(seed=seed, alpha_x=1.0, alpha_y=1.0)
    model.J = np.zeros_like(model.J)
    model.W = np.zeros_like(model.W)
    model.Psi = np.zeros_like(model.Psi)

    rng = np.random.default_rng(seed)
    X, Y = rng.uniform(1.0, 2.0, (N_y, N_x, K)), rng.uniform(0.0, 1.2, (N_y, N_x, K))
    I = rng.random((N_y, N_x, K))

    dXdt, dYdt = model.derivative(X, Y, I)

    np.testing.assert_allclose(model.J, 0.0)
    np.testing.assert_allclose(model.W, 0.0)
    np.testing.assert_allclose(model.Psi, 0.0)
    np.testing.assert_allclose(dYdt, X - Y)
    np.testing.assert_allclose(dXdt, -0.2 * X - 0.21 * Y + I + model.I_o(X) - 0.8)


def test_3(N_y=40, N_x=40, K=12, seed=None):
    model = V1_model_2(seed=seed, alpha_x=1.0, alpha_y=1.0)
    model.J = np.zeros_like(model.J)
    model.W = np.zeros_like(model.W)
    model.Psi = np.zeros_like(model.Psi)

    rng = np.random.default_rng(seed)
    X, Y = rng.uniform(2.0, 4.0, (N_y, N_x, K)), rng.uniform(1.2, 4, (N_y, N_x, K))
    I = rng.random((N_y, N_x, K))

    dXdt, dYdt = model.derivative(X, Y, I)

    np.testing.assert_allclose(model.J, 0.0)
    np.testing.assert_allclose(model.W, 0.0)
    np.testing.assert_allclose(model.Psi, 0.0)
    np.testing.assert_allclose(dYdt, 2 - Y)
    np.testing.assert_allclose(model.g_y(Y), 0.21 * 1.2 + 2.5 * (Y - 1.2))
    np.testing.assert_allclose(model.g_x(X), 1.0)
    np.testing.assert_allclose(
        dXdt, -X + 0.8 + I - (0.21 * 1.2 + 2.5 * (Y - 1.2)) + model.I_o(X)
    )


def test_4(N_y=40, N_x=40, K=12, seed=None):
    model = V1_model_2(seed=seed, alpha_x=1.0, alpha_y=1.0)
    kernel_size = 5
    rng = np.random.default_rng(seed)
    model.J = np.ones((kernel_size, kernel_size, K, K)) * 0.1
    model.W = np.ones((kernel_size, kernel_size, K, K)) * 0.01
    model.Psi = np.zeros((1, 1, K, K)) * 0.25

    rng = np.random.default_rng(seed)
    X, Y = rng.uniform(2.0, 4.0, (N_y, N_x, K)), rng.uniform(1.2, 4, (N_y, N_x, K))
    I = rng.random((N_y, N_x, K))

    dXdt, dYdt = model.derivative(X, Y, I)

    np.testing.assert_allclose(model.J, 0.1)
    np.testing.assert_allclose(model.W, 0.01)
    np.testing.assert_allclose(model.Psi, 0.0)
    np.testing.assert_allclose(model.g_y(Y), 0.21 * 1.2 + 2.5 * (Y - 1.2))
    np.testing.assert_allclose(model.g_x(X), 1.0)
    np.testing.assert_allclose(
        dXdt, -X + 0.8 + kernel_size**2 * K * 0.1 + I - model.g_y(Y) + model.I_o(X)
    )
    np.testing.assert_allclose(dYdt, 2 - Y + kernel_size**2 * K * 0.01)
