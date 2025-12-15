import numpy as np 
import os

from v1sh_model.models.V1_model import V1_model

def stimulus(N_row = None, N_col = None, N_x = 4, N_y = 2, I_input = 1.5, orientation = 0.0):
    if N_row is None:
        N_row = N_y
    if N_col is None:
        N_col = N_x
    
    assert N_x % 2 == 0, "N_x must be even"
    middle_x = int(N_x / 2)
    A = np.full((N_y, N_x), orientation)
    A[:, middle_x:] = orientation + np.pi / 2
    
    C = np.zeros((N_y, N_x))
    C[0 : N_row, 0 : N_col] = I_input
    return A, C

if __name__ == "__main__":
    ################ Parameters ################ 
    T = 12
    dt = 0.01
    seed = 42
    N_rows = np.array([1, 2, 3, 4, 5, 10, 30, 50, 100], dtype = int) # np.array([1, 10, 50, 100], dtype = int)
    I_inputs = np.array([2.0, 3.0, 4.0]) # np.array([2.0, 3.0])
    save_path = "data/results"

    ################### Run #####################
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Save path '{save_path}' does not exist or is not a directory. Please create it before running.")
    save_path = os.path.join(save_path, "border_predictions_T={}_dt={}/".format(T, dt))
    os.makedirs(os.path.join(save_path), exist_ok=False)

    model = V1_model(seed=seed)

    saliency_maps_dict = {}
    for I_input in I_inputs:
        print(f"=== I_input = {I_input} ===")
        
        key = f"I_input={I_input}"
        for N_row in N_rows:
            print(f"--- N_row = {N_row} ---")

            A, C = stimulus(N_row = N_row, N_col = 200, N_x = 200, N_y = 100, I_input=I_input, orientation = 0.0000)
            X_gen, _, _ = model.simulate(A, C, dt=dt, T=T, verbose=False, noisy=False, mode="wrap")
            saliency_map = model.g_x(X_gen).mean(axis=0).max(axis = -1) 
            # save saliency map for predictiong border saliency and bias later
            saliency_maps_dict[f"N_row={N_row}_I_input={I_input}"] = saliency_map
            
    np.savez_compressed(os.path.join(save_path, "border_maps_predictions.npz"), **saliency_maps_dict)

    print(f"Saved files in {save_path}")
            
