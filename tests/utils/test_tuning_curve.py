import numpy as np
import matplotlib.pyplot as plt
import os

from src.utils.tuning_curve import tuning_curve

def test_tuning_curve():
    k_post = 10000
    angles_1 = np.linspace(-np.pi / 2 - np.pi / k_post, 0, k_post)
    angles_2 = -angles_1[::-1][1:]
    angles = np.concatenate([angles_1, angles_2])
    tc_values = tuning_curve(angles)
    print(
        np.allclose(tc_values, tc_values[::-1], atol=1e-6)
    )  # Should be True for perfect symmetry

    plt.rcParams.update({"font.size": 17})
    plt.figure(figsize=(6, 4), dpi=1000, constrained_layout=True)
    plt.plot(angles * 180 / np.pi, tc_values, linewidth=4)
    plt.xticks(np.arange(-90, 91, 30))
    plt.xlabel(r"Angle $x$ [°]")
    plt.ylabel(r"$\phi(x)$")
    plt.title(r"Tuning Curve $\phi(x)$")
    plt.grid(True)

    output_path = "tests/figures/tuning_curve.png"
    plt.savefig(output_path)
    plt.close()

    assert os.path.exists(output_path), f"Plot was not saved to {output_path}"