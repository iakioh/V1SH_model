import numpy as np
from scipy.ndimage import convolve

from src.models.V1_model_1 import V1_model_1 as V1_model


def test_1(N_x=30, N_y=10, N_kernel=10, K=12, seed=None):
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
    actual_result = model.summation(kernel, input, mode="wrap")

    np.testing.assert_allclose(expected_result, actual_result)
