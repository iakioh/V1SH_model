import numpy as np
import matplotlib.pyplot as plt
import os

from v1sh_model.utils.normalization import I_o, I_c

def test_I_c():
    topdown_inputs = [0.0, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0]
    for I_td in topdown_inputs:
        expected_value = 1.0 + I_td
        computed_value = I_c(I_top_down=I_td)
        assert computed_value == expected_value, (
            f"For I_top_down={I_td}, expected {expected_value} but got {computed_value}"
        )

def test_I_o_1(verbose=False):
    # 1)
    X = np.zeros((9, 9, 12))
    X[0, 0, 0] = 2.0

    analytical_result = 0.85 - 2 * (1 / 25) ** 2
    I_o_result = I_o(X)
    assert I_o_result[0, 0, 0] == analytical_result, (
        f"Analytical value: {analytical_result} <-> Code: {I_o_result[0, 0, 0]}"
    )

    if verbose:
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        plt.matshow(I_o_result[..., 0], cmap="viridis", vmin=0.82, vmax=0.85)
        plt.grid(True, which="both", color="black", linestyle="-", linewidth=1)
        plt.colorbar(label=r"$I_o(X)$", fraction=0.046, pad=0.04)
        plt.plot(0, 0, marker="o", markersize=15, color="white", mew=2)
        plt.title(r"X(x=0, y=0, k=0) = 2 else 0")
        plt.xlabel("x")
        plt.ylabel("y")

        output_path = "tests/figures/I_o_1.png"
        plt.savefig(output_path)
        plt.close()

        assert os.path.exists(output_path), f"Plot was not saved to {output_path}"


def test_I_o_2(verbose=False):
    X = np.zeros((9, 9, 12))
    X[0, 0, 0] = 2
    X[4, 4, 5] = 2
    X[4, 4, 6] = 2

    analytical_result = 0.85 - 2 * (3 / 25) ** 2
    I_o_result = I_o(X)
    assert I_o_result[2, 2, 0] == analytical_result, (
        f"Analytical value: {analytical_result} <-> Code: {I_o_result[2, 2, 0]}"
    )

    if verbose:
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        plt.matshow(I_o_result[..., 0], cmap="viridis", vmin=0.82, vmax=0.85)
        plt.grid(True, which="both", color="black", linestyle="-", linewidth=1)
        plt.colorbar(label=r"$I_o(X)$", fraction=0.046, pad=0.04)
        plt.plot(0, 0, marker="o", markersize=15, color="white", mew=2)
        plt.plot(4, 4, marker="o", markersize=15, color="white", mew=2)
        plt.title(r"also X(x=4, y=4, k={5, 6}) = 2")
        plt.xlabel("x")
        plt.ylabel("y")

        output_path = "tests/figures/I_o_2.png"
        plt.savefig(output_path)
        plt.close()

        assert os.path.exists(output_path), f"Plot was not saved to {output_path}"
