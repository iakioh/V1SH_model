import matplotlib.pyplot as plt
import os

from v1sh_model.inputs.visualize import visualize_input
from v1sh_model.inputs.examples import (
    bar_without_surround,
    iso_orientation,
    random_background,
    cross_orientation,
    bar_without_surround_low_contrast,
    with_one_flanker,
    with_two_flankers,
    with_flanking_line_and_noise,
)


def test_examples():
    dpi = 400
    N_y, N_x = 9, 9
    l = 9
    r = 1.3
    d = l * r  # grid spacing
    img_height = int(N_y * d)
    img_width = int(N_x * d)

    plt.rcParams.update({"font.size": 6})
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(img_width / 100 * 4, img_height / 100 * 2),
        dpi=dpi,
        constrained_layout=True,
    )

    axes = axes.flatten()

    test_cases = [
        ("A: Bar without\nsurround", bar_without_surround),
        ("B: Iso-\norientation", iso_orientation),
        ("C: Random\nbackground", random_background),
        ("D: Cross-\norientation", cross_orientation),
        ("E: Bar without\nsurround", bar_without_surround_low_contrast),
        ("F: With one\nflanker", with_one_flanker),
        ("G: With two\nflankers", with_two_flankers),
        ("E: With flanking\nline and noise", with_flanking_line_and_noise),
    ]

    for ax, (title, func) in zip(axes, test_cases):
        A, C = func()
        visualize_input(A, C, verbose=False, axis=ax)
        ax.set_title(title)

    output_path = "./tests/figures/calibration_inputs.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    assert os.path.exists(output_path), f"Plot was not saved to {output_path}"
