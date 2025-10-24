import numpy as np
from scipy.ndimage import convolve

from v1sh_model.models.V1_model_1 import V1_model_1 as V1_model


def test_1(N_x=40, N_y=40, N_kernel=11, K=12, seed=None):
    rng = np.random.default_rng(seed=None)
    input = rng.normal(1, 1, (N_y, N_x, K))
    kernel = rng.normal(0, 1, (N_kernel, N_kernel, K, K))

    # desired result
    result = np.zeros((N_y, N_x, K, K))
    for k_post in range(K):
        for k_pre in range(K):
            result[:, :, k_post, k_pre] = convolve(
                input[:, :, k_pre], kernel[::-1, ::-1, k_post, k_pre], mode="wrap"
            )
    expected_result = np.sum(result, axis=-1)

    # actual result
    model = V1_model(K=K)
    actual_result = model.sum(kernel, input, mode="wrap")

    np.testing.assert_allclose(expected_result, actual_result)


def test_2(seed=None):
    for N_y in [10, 20]:
        for N_x in [10, 20]:
            for K in [5, 7]:
                for N_kernel in [3, 5, 7]:
                    test_1(N_x=N_x, N_y=N_y, N_kernel=N_kernel, K=K, seed=seed)
