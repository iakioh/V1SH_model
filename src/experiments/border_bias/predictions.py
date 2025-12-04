import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from v1sh_model.inputs.visualize import visualize_input, visualize_output
from v1sh_model.models.V1_model_2 import V1_model_2 as V1_model

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
    assert N_y % 2 == 0, "N_y must be even"
    middle_y = int(N_y / 2)
    C[middle_y - N_row // 2 : middle_y + N_row // 2 + 1, middle_x - N_col // 2 : middle_x + N_col // 2 + 1] = I_input
    return A, C

def border_saliency_and_bias(X_gen, n_neighbors = 3, verbose = False):
    C = model.g_x(X_gen).mean(axis=0).max(axis=-1)  # N_y x N_x
    z_score = (C - np.mean(C[C != 0.0])) / np.std(C[C != 0.0])
    N_x = C.shape[1]
    border_idx = N_x // 2 # column right next to the border
    border_saliency_distr = z_score[:, border_idx - n_neighbors:border_idx + n_neighbors].max(axis = 0)
    border_saliency = np.max(border_saliency_distr) 
    max_indices = np.nonzero(border_saliency_distr == border_saliency)[0]
    bias = - (np.mean(max_indices) - n_neighbors + 0.5) # negative because texture parallel to border is left for X_gen
    
    if verbose:
        print(f"Border saliency: {border_saliency}")
        print(f"Border bias: {bias}")
        
        plt.hist(z_score.flatten(), bins=50)
        plt.title("Histogram of output activations")
        plt.axvline(border_saliency, color='red', linestyle='dashed', linewidth=1)
        plt.xlabel("Z-score of output activation")
        plt.ylabel("Frequency")
        plt.title(f"Border saliency: {border_saliency:.2f}")
        plt.yscale('log')
        plt.show()
        
        plt.bar(np.arange(-n_neighbors, + n_neighbors, 1) + 0.5, border_saliency_distr[::-1])
        plt.axvline(0, color = "green", linestyle = "dashed", linewidth = 1, label = "Texture border")
        plt.axvline(bias, color = "red", linestyle = "dashed", linewidth = 1, label = "Predicted bias")
        plt.ylabel("Column saliency (z-score)")
        plt.xlabel("Distance from border towards texture parallel to border (columns)")
        plt.legend()
        plt.title(f"Border bias: {bias}")
        plt.show()
    
    return border_saliency, bias, C

if __name__ == "__main__":
    seed = 42
    model = V1_model(seed=seed)
    T = 12
    dt = 0.01
    
    N_rows = np.array([1, 2, 3, 4, 5, 10, 30, 50, 100], dtype = int) # np.array([1, 10, 50, 100], dtype = int)
    I_inputs = np.array([1.05, 1.5, 2.0, 3.0, 4.0]) # np.array([2.0, 3.0])
    
    bias_dict = {"N_rows": N_rows} 
    saliency_dict = {"N_rows": N_rows}
    saliency_maps_dict = {}
    
    for I_input in I_inputs:
        print(f"=== I_input = {I_input} ===")
        
        key = f"I_input={I_input}"
        saliency_dict[key] = []
        bias_dict[key] = [] 
        for N_row in N_rows:
            print(f"--- N_row = {N_row} ---")

            A, C = stimulus(N_row = N_row, N_col = 48, N_x = 100, N_y = 100, I_input=I_input, orientation = 0.0000)
            X_gen, _, _ = model.simulate(A, C, dt=dt, T=T, verbose=False, noisy=False, mode="wrap")
            
            border_saliency_value, border_bias_value, smap = border_saliency_and_bias(X_gen, verbose=False)
            
            saliency_dict[key].append(border_saliency_value)
            bias_dict[key].append(border_bias_value)
            
            # also save saliency map for explanations later
            saliency_maps_dict[f"N_row={N_row}_I_input={I_input}"] = smap
            
    saliency_df = pd.DataFrame(saliency_dict)
    bias_df = pd.DataFrame(bias_dict)
    saliency_df.to_csv("data/results/border_saliency_predictions.csv", index=False)
    bias_df.to_csv("data/results/border_bias_predictions.csv", index=False)
    np.savez_compressed("data/results/border_maps_predictions.npz", **saliency_maps_dict)
    print("Saved files.")
            
        
