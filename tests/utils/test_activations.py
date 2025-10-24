import numpy as np
import matplotlib.pyplot as plt
import os

from v1sh_model.utils.activations import g_x, g_y


def test_g_x():
    x_vals = np.linspace(0, 3, 300)
    y_vals = g_x(x_vals)

    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(6, 4), dpi=300, constrained_layout=True)
    plt.plot(x_vals, y_vals, label=r"$g_x(x)$", linewidth=3)
    plt.xlabel("x")
    plt.ylabel(r"$g_x(x)$")
    plt.title("Activation function $g_x(x)$")
    plt.grid(True)
    
    output_path = "tests/figures/g_x.png"
    plt.savefig(output_path)
    plt.close()

    assert os.path.exists(output_path), f"Plot was not saved to {output_path}"

def test_g_y():
    y_range = np.linspace(-1, 3, 400)
    g_y_vals = g_y(y_range)

    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(6, 4), dpi=300, constrained_layout=True)
    plt.plot(y_range, g_y_vals, label=r"$g_y(y)$", linewidth=3)
    plt.xlabel("y")
    plt.ylabel(r"$g_y(y)$")
    plt.title("Activation function $g_y(y)$")
    plt.grid(True)
    
    output_path = "tests/figures/g_y.png"
    plt.savefig(output_path)
    plt.close()

    assert os.path.exists(output_path), f"Plot was not saved to {output_path}"
