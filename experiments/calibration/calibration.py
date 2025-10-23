# replicate fig. 5.18 in "Understanding Vision" (Li Zhaoping, 2014)
import numpy as np
import concurrent.futures 
import sys
sys.path.append(r"\\kfs\krothe\Windows Folders\My Documents\V1SH_model")

from src.models.V1_model_1 import V1_model_1 as V1_model
from src.inputs.examples import (
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
    model = V1_model()
    g_x = model.g_x  
    
    T = 12.0
    dt = 0.001
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
    # input_and_outputs = {}
    # for title, func in test_cases.items():
    #     # create input images
    #     A_in, C_in = func(N_y = N_y_test, N_x = N_x_test) # A, C shape (N_y, N_x)
        
    #     # simulate model
    #     X, _, _ = model.simulate(A_in, C_in, dt=dt, T=T, verbose=False, noisy=True, mode="wrap")
    #     model_output = g_x(X).mean(axis=0)  # N_y x N_x x K
    #     C_out = model_output.max(axis=-1) # N_y x N_x
    #     argmax_angle_indices = model_output.argmax(axis=-1) # N_y x N_x
    #     A_out = np.pi / model.K * argmax_angle_indices # N_y x N_x
        
    #     input_and_outputs[title] = (A_in, C_in, A_out, C_out)

    def run_test_case(args):
        # create input images
        title, N_y_test, N_x_test, dt, T = args
        func = test_cases[title]
        A_in, C_in = func(N_y=N_y_test, N_x=N_x_test)
        
        # simulate model
        X, _, _ = model.simulate(A_in, C_in, dt=dt, T=T, verbose=False, noisy=True, mode="wrap")
        model_output = model.g_x(X).mean(axis=0)
        C_out = model_output.max(axis=-1)
        argmax_angle_indices = model_output.argmax(axis=-1)
        A_out = np.pi / model.K * argmax_angle_indices
        
        return title, (A_in, C_in, A_out, C_out)

    # Prepare arguments for each test case
    args_list = [
        (title, N_y_test, N_x_test, dt, T)
        for title in test_cases.keys()
    ]

    input_and_outputs = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(run_test_case, args) for args in args_list]
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            key, result = future.result()
            input_and_outputs[key] = result
            
    # save results
    output_path = "data/calibration_results.npz"
    np.savez_compressed(output_path, **input_and_outputs)
    print(f"Calibration results saved to {output_path}")