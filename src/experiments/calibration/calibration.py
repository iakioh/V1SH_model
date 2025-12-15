# replicate fig. 5.18 in "Understanding Vision" (Li Zhaoping, 2014)

import numpy as np
import concurrent.futures

from v1sh_model.models.V1_model import V1_model
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

if __name__ == "__main__":
    model = V1_model(average_noise_height = 0.1, average_noise_temporal_width = 0.1, seed = 42)
    g_x = model.g_x

    T = 12.0
    dt = 0.01
    N_y_test, N_x_test = 9 + 2 * 10, 9 + 2 * 10

    test_cases = {
        "A: Bar without\nsurround": bar_without_surround,
        "B: Iso-\norientation": iso_orientation,
        "C: Random\nbackground": random_background,
        "D: Cross-\norientation": cross_orientation,
        "E: Bar without\nsurround": bar_without_surround_low_contrast,
        "F: With one\nflanker": with_one_flanker,
        "G: With two\nflankers": with_two_flankers,
        "E: With flanking\nline and noise": with_flanking_line_and_noise,
    }

    # generate model response for all test cases
    input_and_outputs = {}
    for title, func in test_cases.items():
        # create input images
        A_in, C_in = func(N_y=N_y_test, N_x=N_x_test)  # A, C shape (N_y, N_x)

        # simulate model
        X, _, _ = model.simulate(
            A_in, C_in, dt=dt, T=T, verbose=False, noisy=False, mode="wrap"
        )
        model_output = g_x(X).mean(axis=0)  # N_y x N_x x K
        C_out = model_output.max(axis=-1)  # N_y x N_x
        argmax_angle_indices = model_output.argmax(axis=-1)  # N_y x N_x
        A_out = np.pi / model.K * argmax_angle_indices  # N_y x N_x

        input_and_outputs[title] = (A_in, C_in, A_out, C_out)

    ###### to parallelize the above loop instead, uncomment below: ######
    
    # def run_test_case(args):
    #     # create input images
    #     title, N_y_test, N_x_test, dt, T = args
    #     func = test_cases[title]
    #     A_in, C_in = func(N_y=N_y_test, N_x=N_x_test)

    #     # simulate model
    #     X, _, _ = model.simulate(A_in, C_in, dt=dt, T=T, verbose=False, noisy=True, mode="wrap")
    #     model_output = model.g_x(X).mean(axis=0)
    #     C_out = model_output.max(axis=-1)
    #     argmax_angle_indices = model_output.argmax(axis=-1)
    #     A_out = np.pi / model.K * argmax_angle_indices

    #     return title, (A_in, C_in, A_out, C_out)

    # # Prepare arguments for each test case
    # args_list = [
    #     (title, N_y_test, N_x_test, dt, T)
    #     for title in test_cases.keys()
    # ]

    # input_and_outputs = {}
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # Submit all tasks
    #     futures = [executor.submit(run_test_case, args) for args in args_list]
    #     # Collect results as they complete
    #     for future in concurrent.futures.as_completed(futures):
    #         key, result = future.result()
    #         input_and_outputs[key] = result
    
    ###### end of parallelized version ######

    # save results
    output_path = "data/results/calibration_results.npz"
    np.savez_compressed(output_path, **input_and_outputs)
    print(f"Calibration results saved to {output_path}")
