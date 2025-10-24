import numpy as np
import matplotlib.pyplot as plt
import os

from v1sh_model.utils.connections import (
    compute_psi,
    compute_connection_kernel,
    visualize_weights,
)


def test_compute_psi():
    K = 12
    N = 1
    thetas = np.linspace(-N * K, N * K, N * K + 1) * np.pi / 2 / K
    psi_values = compute_psi(thetas, K, atol=1e-6)

    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8, 3), dpi=300, constrained_layout=True)
    plt.plot(thetas / np.pi * 180, psi_values, linewidth=1, marker="o", markersize=8)
    plt.xlabel(r"angle $\theta$ [°]")
    plt.ylabel(r"$\psi(\theta)$")
    plt.title(r"Intracortical connection strength $\psi(\theta)$ for $K = 12$")
    plt.xticks(np.arange(-90, 91, 15))
    plt.xlim(-95, 95)
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    output_path = "tests/figures/Psi.png"
    plt.savefig(output_path)
    plt.close()

    assert os.path.exists(output_path), f"Plot was not saved to {output_path}"


def test_compute_WJ():
    K = 12
    J, W, Psi = compute_connection_kernel(K=K, verbose=False)
    fig1, fig2 = visualize_weights(W, J, Psi, k_pres=[6], K=K, dpi=400, colored=False)

    output_path1 = "tests/figures/W.png"
    plt.savefig(output_path1)
    plt.close()
    assert os.path.exists(output_path1), f"Plot was not saved to {output_path1}"

    output_path2 = "tests/figures/J.png"
    plt.savefig(output_path2)
    plt.close()
    assert os.path.exists(output_path2), f"Plot was not saved to {output_path2}"

def test_compute_WJPsi():
    K = 12
    J, W, Psi = compute_connection_kernel(K=K, verbose=False)
    fig = visualize_weights(W, J, Psi, k_pres=[6], K=K, dpi=400, colored=True)

    output_path = "tests/figures/JWPsi.png"
    plt.savefig(output_path)
    plt.close()

    assert os.path.exists(output_path), f"Plot was not saved to {output_path}"

